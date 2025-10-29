"""
Text generation script for TRM.
Supports greedy, sampling, and ACT-based adaptive generation.
"""

import torch
import json
from transformers import AutoTokenizer
from pathlib import Path
import argparse

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained TRM model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config from checkpoint directory
    config_path = Path(checkpoint_path).parent / "all_config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    # Create model
    model = TinyRecursiveReasoningModel_ACTV1(config['arch'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    return model, config


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    use_act: bool = True,
    device: str = "cuda"
):
    """
    Generate text using TRM with adaptive computation.
    
    Args:
        model: TRM model
        tokenizer: HuggingFace tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (1.0 = default, <1 = more focused)
        top_p: Nucleus sampling threshold
        use_act: Use ACT halting (adaptive cycles per token)
        device: cuda or cpu
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Initialize carry
    batch = {
        "inputs": input_ids,
        "labels": input_ids,  # Dummy labels
        "puzzle_identifiers": torch.zeros(1, dtype=torch.int32, device=device)
    }
    carry = model.initial_carry(batch)
    
    generated_tokens = input_ids.tolist()[0]
    act_steps = []  # Track ACT cycles per token
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Update batch with current sequence
            current_ids = torch.tensor([generated_tokens], device=device)
            batch_current = {
                "inputs": current_ids,
                "labels": current_ids,
                "puzzle_identifiers": torch.zeros(1, dtype=torch.int32, device=device)
            }
            
            # Forward pass with ACT
            carry_out, outputs = model(carry, batch_current)
            
            # Get logits for last token
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative prob > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Track ACT steps
            if use_act:
                steps = carry_out.steps[0].item()
                act_steps.append(steps)
            
            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token)
            carry = carry_out
    
    # Decode
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    if use_act and act_steps:
        avg_steps = sum(act_steps) / len(act_steps)
        print(f"\n📊 ACT Statistics:")
        print(f"   Average cycles per token: {avg_steps:.2f}")
        print(f"   Total tokens: {len(act_steps)}")
        print(f"   Efficiency gain: {(16 - avg_steps) / 16 * 100:.1f}% (vs max 16 cycles)")
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with TRM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--no-act", action="store_true", help="Disable ACT halting")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    print(f"\n🎯 Prompt: {args.prompt}")
    print(f"⚙️  Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"🔄 ACT: {'Enabled' if not args.no_act else 'Disabled'}\n")
    
    # Generate
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_act=not args.no_act,
        device=args.device
    )
    
    print(f"\n📝 Generated text:\n{generated}\n")


if __name__ == "__main__":
    main()
