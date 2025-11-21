import sys
import os
from pathlib import Path

rearc_path = Path("dataset/re-arc")
print(f"Re-ARC path: {rearc_path.absolute()}")
print(f"Exists: {rearc_path.exists()}")

sys.path.insert(0, str(rearc_path))

try:
    import generators
    print("Successfully imported generators")
    
    gen_names = [name for name in dir(generators) if name.startswith('generate_')]
    print(f"Found {len(gen_names)} generators")
    print(f"First 5 generators: {gen_names[:5]}")
    
    if gen_names:
        first_gen = getattr(generators, gen_names[0])
        print(f"Testing generator: {gen_names[0]}")
        try:
            example = first_gen(diff_lb=0, diff_ub=1)
            print("Successfully generated example")
            print(f"Input shape: {len(example['input'])}")
        except Exception as e:
            print(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()

except ImportError as e:
    print(f"Failed to import generators: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
