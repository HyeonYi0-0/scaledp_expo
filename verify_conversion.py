#!/usr/bin/env python3
"""Simple verification script that doesn't require external dependencies"""

import sys
import os
import inspect

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_file_structure():
    """Verify that all files exist and have the expected structure"""
    print("=== File Structure Verification ===")
    
    files_to_check = [
        "src/policy/EXPO.py",
        "src/distributions/tanh_normal.py", 
        "src/model/mlp.py",
        "src/model/ensemble.py",
        "src/model/state_action_value.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
            
            # Check file content for PyTorch imports
            with open(file_path, 'r') as f:
                content = f.read()
                
            if 'import torch' in content:
                print(f"  ✓ Contains PyTorch imports")
            else:
                print(f"  ⚠ Missing PyTorch imports")
                
            if any(jax_term in content for jax_term in ['import jax', 'jax.', 'jnp.', 'flax']):
                print(f"  ⚠ Still contains JAX/Flax code")
            else:
                print(f"  ✓ JAX/Flax code removed")
        else:
            print(f"✗ {file_path} missing")

def verify_class_definitions():
    """Verify that key classes are properly defined"""
    print("\n=== Class Definition Verification ===")
    
    try:
        # Check if we can read the files and find class definitions
        with open('src/policy/EXPO.py', 'r') as f:
            expo_content = f.read()
            
        # Check for key PyTorch patterns
        pytorch_patterns = [
            'class Agent:',
            'class Temperature(nn.Module):',
            'class Expo(Agent):',
            'torch.nn as nn',
            'torch.optim as optim',
            'def forward(',
            'def __init__(',
            'super().__init__()',
        ]
        
        for pattern in pytorch_patterns:
            if pattern in expo_content:
                print(f"✓ Found pattern: {pattern}")
            else:
                print(f"✗ Missing pattern: {pattern}")
                
        # Check for removed JAX patterns
        jax_patterns = [
            '@partial(jax.jit',
            'struct.PyTreeNode',
            'TrainState',
            'jax.random.split',
            'optax.',
        ]
        
        print("\nJAX patterns (should be removed):")
        for pattern in jax_patterns:
            if pattern in expo_content:
                print(f"⚠ Still contains: {pattern}")
            else:
                print(f"✓ Removed: {pattern}")
                
    except Exception as e:
        print(f"Error reading files: {e}")

def verify_method_signatures():
    """Verify that key methods have the expected signatures"""
    print("\n=== Method Signature Verification ===")
    
    try:
        with open('src/policy/EXPO.py', 'r') as f:
            content = f.read()
            
        expected_methods = [
            'def create(',
            'def update_actor(',
            'def update_critic(',
            'def update_temperature(',
            'def update(',
            'def state_dict(',
            'def load_state_dict(',
        ]
        
        for method in expected_methods:
            if method in content:
                print(f"✓ Method found: {method}")
            else:
                print(f"✗ Method missing: {method}")
                
    except Exception as e:
        print(f"Error checking methods: {e}")

def main():
    """Main verification function"""
    print("PyTorch Conversion Verification (Structure Only)")
    print("=" * 50)
    
    verify_file_structure()
    verify_class_definitions()
    verify_method_signatures()
    
    print("\n" + "=" * 50)
    print("Verification complete!")
    print("\nNote: This verification only checks code structure.")
    print("To test functionality, install PyTorch and run:")
    print("python test_pytorch_conversion.py")

if __name__ == "__main__":
    main()
