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
        encoder_model: str = "openai/clip-vit-large-patch14",
        use_spatial: bool = True,  # Enable spatial organization
        capsule_grid_shape: tuple = (3, 4),  # 3x4 grid = 12 capsules
        # NEW: Multi-encoder support (from cnn_tokenizer migration)
        use_dinov2: bool = False,  # Add DINOv2 for stronger visual reasoning
        dinov2_model: str = "facebook/dinov2-base",
        use_hybrid: bool = False,  # Concatenate CLIP + DINOv2 features
        output_mode: str = "capsules",  # "capsules" (hierarchical) or "patches" (flat)
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
        self.use_spatial = use_spatial
        self.capsule_grid_shape = capsule_grid_shape
        self.use_dinov2 = use_dinov2
        self.use_hybrid = use_hybrid or (use_dinov2 and "clip" in encoder_model.lower())
        self.output_mode = output_mode
        self.freeze_encoder = freeze_encoder
        
        assert capsule_grid_shape[0] * capsule_grid_shape[1] == target_capsules, \
            f"Grid shape {capsule_grid_shape} must multiply to {target_capsules}"
        
        # Initialize encoders
        print(f"\n🔧 Initializing CapsuleEncoder:")
        print(f"   Primary encoder: {encoder_model}")
        print(f"   DINOv2: {use_dinov2}")
        print(f"   Hybrid mode: {self.use_hybrid}")
        print(f"   Output mode: {output_mode}")
        print(f"   Target capsules: {target_capsules}")
        print(f"   Spatial mode: {use_spatial}")
        if use_spatial:
            print(f"   Capsule grid: {capsule_grid_shape[0]}×{capsule_grid_shape[1]}")
        
        # Initialize primary encoder (CLIP or other)
        if "clip" in encoder_model.lower():
            self.encoder = AutoModel.from_pretrained(encoder_model)
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
            self.encoder_type = "clip"
            clip_dim = 768 if "large" in encoder_model else 512
        else:
            self.encoder = AutoModel.from_pretrained(encoder_model)
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
            self.encoder_type = "transformer"
            clip_dim = self.encoder.config.hidden_size
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Initialize DINOv2 encoder (optional)
        self.dinov2_encoder = None
        dinov2_dim = 0
        if use_dinov2:
            print(f"   Loading DINOv2: {dinov2_model}")
            self.dinov2_encoder = AutoModel.from_pretrained(dinov2_model)
            if freeze_encoder:
                for param in self.dinov2_encoder.parameters():
                    param.requires_grad = False
                self.dinov2_encoder.eval()
            dinov2_dim = self.dinov2_encoder.config.hidden_size
        
        # Set encoder dimension based on mode
        if self.use_hybrid and use_dinov2:
            self.encoder_dim = clip_dim + dinov2_dim
            print(f"   ✓ Hybrid: CLIP({clip_dim}) + DINOv2({dinov2_dim}) = {self.encoder_dim}")
        else:
            self.encoder_dim = dinov2_dim if use_dinov2 and not self.use_hybrid else clip_dim
        
        # Capsule components
        self.sketch_projection = nn.Linear(self.encoder_dim, hidden_size)
        self.checksum_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, checksum_dim)
        )
        self.children_projection = nn.Linear(self.encoder_dim, hidden_size)
        
        # NEW: Spatial components
        if use_spatial:
            # 2D convolutions for spatial feature extraction
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(self.encoder_dim, hidden_size, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
            )
            
            # Spatial positional encoding (3x4 grid)
            self.spatial_pos_embed = nn.Parameter(
                torch.randn(1, target_capsules, hidden_size) * 0.02
            )
            
            # Spatial attention bias (nearby capsules attend more)
            self.register_buffer('spatial_bias', self._build_spatial_bias())
        else:
            self.spatial_conv = None
            self.spatial_pos_embed = None
        
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
    
    def _build_spatial_bias(self):
        """Build attention bias matrix based on capsule spatial positions."""
        H, W = self.capsule_grid_shape
        positions = [(i // W, i % W) for i in range(self.target_capsules)]
        
        # Distance-based bias (Manhattan distance)
        bias = torch.zeros(self.target_capsules, self.target_capsules)
        for i, (r1, c1) in enumerate(positions):
            for j, (r2, c2) in enumerate(positions):
                dist = abs(r1 - r2) + abs(c1 - c2)
                bias[i, j] = -0.1 * dist  # Closer = higher attention
        
        return bias
    
    def _spatial_pool(self, embeddings):
        """Apply spatial convolutions and pool to grid layout.
        
        Args:
            embeddings: [k, D] embeddings (k <= target_capsules)
            
        Returns:
            [k, D] spatially-processed embeddings
        """
        if not self.use_spatial or embeddings.size(0) != self.target_capsules:
            return embeddings
        
        H, W = self.capsule_grid_shape
        
        # Reshape to 2D grid: [k, D] -> [1, D, H, W]
        grid = embeddings.view(H, W, -1).permute(2, 0, 1).unsqueeze(0)
        
        # Apply spatial convolutions (preserves spatial relationships)
        spatial_features = self.spatial_conv(grid)  # [1, hidden_size, H, W]
        
        # Flatten back to sequence: [1, hidden_size, H, W] -> [k, hidden_size]
        output = spatial_features.squeeze(0).permute(1, 2, 0).reshape(self.target_capsules, -1)
        
        return output
    
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
        
        # Pad to target_capsules and apply spatial pooling
        padded_sketches = []
        for emb in sketches_list:
            if emb.size(0) > self.target_capsules:
                emb = emb[:self.target_capsules]
            elif emb.size(0) < self.target_capsules:
                pad = torch.zeros(self.target_capsules - emb.size(0), self.encoder_dim, device=device)
                emb = torch.cat([emb, pad], dim=0)
            
            # NEW: Apply spatial pooling if enabled
            if self.use_spatial:
                emb = self._spatial_pool(emb)
            
            padded_sketches.append(emb)
        
        sketches_raw = torch.stack(padded_sketches, dim=0)  # [B, k, encoder_dim]
        sketches = self.sketch_projection(sketches_raw)  # [B, k, hidden_size]
        
        # NEW: Add spatial positional encoding
        if self.use_spatial:
            sketches = sketches + self.spatial_pos_embed
        
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
            batch_size = image_tensors.shape[0]
            all_features = []
            
            # CLIP features
            with torch.no_grad() if not self.training else torch.enable_grad():
                if self.use_hybrid or not self.use_dinov2:
                    # Use CLIP
                    clip_features = self.encoder.get_image_features(pixel_values=image_tensors)
                    all_features.append(clip_features)
            
            # DINOv2 features
            if self.dinov2_encoder is not None:
                with torch.no_grad() if not self.training else torch.enable_grad():
                    dinov2_features = self.dinov2_encoder(pixel_values=image_tensors).last_hidden_state[:, 0]  # CLS token
                    all_features.append(dinov2_features)
            
            # Combine features
            if self.use_hybrid and len(all_features) > 1:
                image_features = torch.cat(all_features, dim=-1)  # [B, clip_dim + dinov2_dim]
            else:
                image_features = all_features[0]  # [B, encoder_dim]
            
            if self.output_mode == "patches":
                # Flat patches mode (from cnn_tokenizer): return raw patch embeddings
                from transformers import CLIPVisionModel
                all_patch_features = []
                
                # CLIP patches
                if self.use_hybrid or not self.use_dinov2:
                    if isinstance(self.encoder, CLIPVisionModel):
                        with torch.no_grad() if not self.training else torch.enable_grad():
                            outputs = self.encoder.vision_model(pixel_values=image_tensors, output_hidden_states=True)
                            clip_patches = outputs.last_hidden_state  # [B, num_patches+1, clip_dim]
                        all_patch_features.append(clip_patches)
                
                # DINOv2 patches
                if self.dinov2_encoder is not None:
                    with torch.no_grad() if not self.training else torch.enable_grad():
                        dinov2_outputs = self.dinov2_encoder(pixel_values=image_tensors)
                        dinov2_patches = dinov2_outputs.last_hidden_state  # [B, num_patches+1, dinov2_dim]
                    all_patch_features.append(dinov2_patches)
                
                # Combine patches
                if self.use_hybrid and len(all_patch_features) > 1:
                    # Concatenate along feature dimension
                    sketches = torch.cat(all_patch_features, dim=-1)  # [B, seq_len, combined_dim]
                else:
                    sketches = all_patch_features[0]
                
            elif self.use_spatial:
                # Spatial capsules mode: extract spatial patch features
                from transformers import CLIPVisionModel
                all_spatial_features = []
                
                # CLIP spatial features
                if self.use_hybrid or not self.use_dinov2:
                    if isinstance(self.encoder, CLIPVisionModel):
                        # Get patch embeddings before pooling [B, num_patches+1, dim]
                        with torch.no_grad() if not self.training else torch.enable_grad():
                            outputs = self.encoder.vision_model(pixel_values=image_tensors, output_hidden_states=True)
                            patch_embeddings = outputs.last_hidden_state[:, 1:]  # Remove CLS token
                        
                        # Adaptive pooling to 3×4 grid
                        H, W = self.capsule_grid_shape
                        num_patches = patch_embeddings.size(1)
                        patch_grid_size = int(num_patches ** 0.5)  # CLIP uses square grids
                        
                        # Reshape to spatial grid
                        patch_grid = patch_embeddings.view(batch_size, patch_grid_size, patch_grid_size, -1)
                        
                        # Adaptive average pooling to target grid size
                        patch_grid = patch_grid.permute(0, 3, 1, 2)  # [B, D, H_p, W_p]
                        pooled = F.adaptive_avg_pool2d(patch_grid, (H, W))  # [B, D, H, W]
                        
                        # Flatten to capsule sequence
                        clip_spatial = pooled.permute(0, 2, 3, 1).reshape(batch_size, H * W, -1)  # [B, k, clip_dim]
                        all_spatial_features.append(clip_spatial)
                
                # DINOv2 spatial features
                if self.dinov2_encoder is not None:
                    with torch.no_grad() if not self.training else torch.enable_grad():
                        dinov2_outputs = self.dinov2_encoder(pixel_values=image_tensors)
                        dinov2_patches = dinov2_outputs.last_hidden_state[:, 1:]  # Remove CLS
                    
                    # Pool DINOv2 patches to grid
                    H, W = self.capsule_grid_shape
                    num_patches = dinov2_patches.size(1)
                    patch_grid_size = int(num_patches ** 0.5)
                    
                    patch_grid = dinov2_patches.view(batch_size, patch_grid_size, patch_grid_size, -1)
                    patch_grid = patch_grid.permute(0, 3, 1, 2)
                    pooled = F.adaptive_avg_pool2d(patch_grid, (H, W))
                    dinov2_spatial = pooled.permute(0, 2, 3, 1).reshape(batch_size, H * W, -1)
                    all_spatial_features.append(dinov2_spatial)
                
                # Combine spatial features
                if self.use_hybrid and len(all_spatial_features) > 1:
                    sketches = torch.cat(all_spatial_features, dim=-1)  # [B, k, combined_dim]
                else:
                    sketches = all_spatial_features[0] if all_spatial_features else image_features.unsqueeze(1).repeat(1, self.target_capsules, 1)
            else:
                # Non-spatial mode: repeat global feature
                sketches = image_features.unsqueeze(1).repeat(1, self.target_capsules, 1)  # [B, k, encoder_dim]
            
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


