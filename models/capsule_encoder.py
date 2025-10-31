"""
Hierarchical Expandable Semantic Capsules (HESC)

Creates capsules with:
- Sketch: coarse semantic embedding (concept-level)
- Checksum: reconstructability signature  
- Children: fine-grained tokens (expandable on-demand)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsuleEncoder(nn.Module):
    """
    HESC encoder: text → capsules with sketch/checksum/children.
    
    Args:
        hidden_size: TRM hidden dimension (768)
        target_capsules: Number of coarse capsules (k=12)
        children_per_capsule: Fine tokens per capsule (m=4)
        checksum_dim: Reconstructability signature size (32)
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        target_capsules: int = 12,
        children_per_capsule: int = 4,
        checksum_dim: int = 32,
        freeze_encoder: bool = True,
        encoder_model: str = "openai/clip-vit-large-patch14",  # Configurable encoder
    ):
        super().__init__()
        
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("pip install transformers")
        
        self.hidden_size = hidden_size
        self.target_capsules = target_capsules
        self.children_per_capsule = children_per_capsule
        self.checksum_dim = checksum_dim
        self.encoder_model = encoder_model
        
        # Semantic encoder (default: CLIP, but configurable)
        print(f"\n🔧 Initializing CapsuleEncoder:")
        print(f"   Encoder: {encoder_model}")
        print(f"   Target capsules: {target_capsules}")
        print(f"   Children per capsule: {children_per_capsule}")
        
        if "clip" in encoder_model.lower():
            # CLIP models use get_text_features()
            self.encoder = AutoModel.from_pretrained(encoder_model)
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
            self.encoder_type = "clip"
            self.encoder_dim = 768 if "large" in encoder_model else 512
        else:
            # Other encoders (BERT, RoBERTa, etc.)
            self.encoder = AutoModel.from_pretrained(encoder_model)
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
            self.encoder_type = "transformer"
            self.encoder_dim = self.encoder.config.hidden_size
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Capsule components (adapt to encoder dimension)
        self.sketch_projection = nn.Linear(self.encoder_dim, hidden_size)
        self.checksum_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, checksum_dim)
        )
        self.children_projection = nn.Linear(self.encoder_dim, hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.sketch_projection.weight)
        nn.init.zeros_(self.sketch_projection.bias)
        nn.init.xavier_uniform_(self.children_projection.weight)
        nn.init.zeros_(self.children_projection.bias)
        for m in self.checksum_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, texts=None, images=None, return_children: bool = True):
        """
        Encode texts or images to capsules (multimodal support).
        
        Args:
            texts: List of text strings or None
            images: List of PIL Images/tensors or None
            texts: List of strings or pre-encoded tensor
            return_children: Whether to compute children embeddings
        
        Returns:
            dict with 'sketches' [B,k,D], 'checksums' [B,k,R], 'children' [B,k,m,D]
        """
        device = next(self.parameters()).device
        
        # Handle pre-encoded tensors (precomputed mode)
        if isinstance(texts, torch.Tensor):
            device = next(self.encoder.parameters()).device
        
        # Handle vision inputs
        if images is not None:
            return self._encode_images(images, return_children)
        
        # Text encoding path
        if texts is None:
            raise ValueError("Either texts or images must be provided")
        
        # Chunk texts into coarse semantic chunks (constituency-aware)
        all_capsule_chunks = []
        all_children_chunks = []
        capsule_counts = []
        
        for text in texts:
            # Coarse chunking (constituency-based or uniform)
            chunks = self._chunk_text_hierarchical(text, self.target_capsules)
            all_capsule_chunks.extend(chunks)
            
            # Fine chunking (children per capsule)
            if return_children:
                for chunk in chunks:
                    children = self._split_into_children(chunk, self.children_per_capsule)
                    all_children_chunks.append(children)
            
            capsule_counts.append(len(chunks))
        
        # Batch encode coarse sketches
        capsule_inputs = self.tokenizer(
            all_capsule_chunks,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            if self.encoder_type == "clip":
                coarse_embeddings = self.encoder.get_text_features(**capsule_inputs)
            else:
                outputs = self.encoder(**capsule_inputs)
                coarse_embeddings = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Split back into batches
        sketches_list = []
        idx = 0
        for count in capsule_counts:
            sketches_list.append(coarse_embeddings[idx:idx+count])
            idx += count
        
        # Pad to target_capsules
        padded_sketches = []
        for emb in sketches_list:
            if emb.size(0) > self.target_capsules:
                emb = emb[:self.target_capsules]
            elif emb.size(0) < self.target_capsules:
                pad = torch.zeros(self.target_capsules - emb.size(0), 768, device=device)
                emb = torch.cat([emb, pad], dim=0)
            padded_sketches.append(emb)
        
        sketches_raw = torch.stack(padded_sketches, dim=0)  # [B, k, 768]
        sketches = self.sketch_projection(sketches_raw)  # [B, k, hidden_size]
        
        # Compute checksums (reconstructability signals)
        checksums = self.checksum_head(sketches)  # [B, k, checksum_dim]
        
        result = {
            'sketches': sketches,
            'checksums': checksums,
            'capsule_counts': capsule_counts
        }
        
        # Encode children if requested
        if return_children and all_children_chunks:
            children_embeddings = []
            
            for children_list in all_children_chunks:
                if not children_list:
                    # Empty children
                    empty = torch.zeros(self.children_per_capsule, 768, device=device)
                    children_embeddings.append(empty)
                    continue
                
                # Batch encode children
                child_inputs = self.tokenizer(
                    children_list,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(device)
                
                with torch.no_grad() if not self.training else torch.enable_grad():
                    if self.encoder_type == "clip":
                        child_emb = self.encoder.get_text_features(**child_inputs)
                    else:
                        outputs = self.encoder(**child_inputs)
                        child_emb = outputs.last_hidden_state[:, 0]
                
                # Pad to children_per_capsule
                if child_emb.size(0) < self.children_per_capsule:
                    pad = torch.zeros(self.children_per_capsule - child_emb.size(0), self.encoder_dim, device=device)
                    child_emb = torch.cat([child_emb, pad], dim=0)
                elif child_emb.size(0) > self.children_per_capsule:
                    child_emb = child_emb[:self.children_per_capsule]
                
                children_embeddings.append(child_emb)
            
            # Reshape to [B, k, m, 768]
            children_batched = []
            idx = 0
            for count in capsule_counts:
                batch_children = children_embeddings[idx:idx+count]
                
                # Pad to target_capsules
                while len(batch_children) < self.target_capsules:
                    batch_children.append(torch.zeros(self.children_per_capsule, self.encoder_dim, device=device))
                batch_children = batch_children[:self.target_capsules]
                
                children_batched.append(torch.stack(batch_children, dim=0))
                idx += count
            
            children = torch.stack(children_batched, dim=0)  # [B, k, m, encoder_dim]
            children = self.children_projection(children.view(-1, self.encoder_dim)).view(
                children.size(0), children.size(1), children.size(2), self.hidden_size
            )  # [B, k, m, hidden_size]
            
            result['children'] = children
        
        return result
    
    def _encode_images(self, images, return_children: bool):
        """
        Encode images to capsules using CLIP vision encoder.
        
        Args:
            images: List of PIL Images or torch tensors [B, 3, H, W]
            return_children: Whether to compute spatial children
        
        Returns:
            dict with 'sketches', 'checksums', 'children'
        """
        device = next(self.encoder.parameters()).device
        
        # Preprocess images
        if self.encoder_type == "clip":
            # CLIP image preprocessing
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                   std=[0.26862954, 0.26130258, 0.27577711])
            ])
            
            # Process images
            if isinstance(images[0], torch.Tensor):
                image_tensors = torch.stack(images).to(device)
            else:
                # PIL Images
                image_tensors = torch.stack([preprocess(img) for img in images]).to(device)
            
            # Encode with CLIP vision encoder
            with torch.no_grad() if not self.training else torch.enable_grad():
                image_features = self.encoder.get_image_features(pixel_values=image_tensors)
            
            # image_features: [B, encoder_dim]
            # Split into k capsules (spatial or semantic regions)
            batch_size = image_features.shape[0]
            sketches = image_features.unsqueeze(1).repeat(1, self.target_capsules, 1)  # [B, k, D]
            
            # Project to hidden size
            sketches = self.sketch_projection(sketches)  # [B, k, hidden_size]
            
            # Compute checksums
            checksums = self.checksum_head(sketches)  # [B, k, checksum_dim]
            
            result = {
                'sketches': sketches,
                'checksums': checksums
            }
            
            # Children: spatial patches (if requested)
            if return_children:
                # Split image into spatial regions as children
                # For now, duplicate sketches (TODO: implement spatial patching)
                children = sketches.unsqueeze(2).repeat(1, 1, self.children_per_capsule, 1)
                result['children'] = children  # [B, k, m, hidden_size]
            
            return result
        else:
            raise NotImplementedError(f"Image encoding not implemented for {self.encoder_type}")
    
    def _chunk_text_hierarchical(self, text: str, target_chunks: int) -> list:
        """Split text into coarse semantic chunks (constituency-aware)."""
        # Simple word-based chunking (can be improved with spaCy constituency parser)
        words = text.split()
        if len(words) <= target_chunks:
            return [text]
        
        chunk_size = max(1, len(words) // target_chunks)
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks[:target_chunks]
    
    def _split_into_children(self, chunk: str, num_children: int) -> list:
        """Split chunk into finer children."""
        words = chunk.split()
        if len(words) <= num_children:
            return [chunk]
        
        child_size = max(1, len(words) // num_children)
        children = []
        for i in range(0, len(words), child_size):
            child = ' '.join(words[i:i + child_size])
            if child:
                children.append(child)
        
        return children[:num_children]
    
    def expand_capsule(self, capsule_idx: int, children_embeddings: torch.Tensor):
        """
        Expand a capsule by replacing its sketch with children embeddings.
        
        Args:
            capsule_idx: Index of capsule to expand
            children_embeddings: [m, hidden_size] children embeddings
        
        Returns:
            Expanded embeddings to splice into sequence
        """
        return children_embeddings  # [m, hidden_size]
    
    def get_compression_ratio(self, texts: list, bpe_tokenizer) -> float:
        """Measure compression vs BPE."""
        bpe_tokens = sum(len(bpe_tokenizer.encode(text)) for text in texts)
        capsule_tokens = len(texts) * self.target_capsules
        return bpe_tokens / max(capsule_tokens, 1)


