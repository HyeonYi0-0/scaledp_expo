#!/usr/bin/env python3
"""Test script to verify PyTorch conversion functionality"""

import sys
import os
import numpy as np
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.policy.EXPO import Expo

class DummySpace:
    """Dummy gym space for testing"""
    def __init__(self, shape):
        self.shape = shape
    
    def sample(self):
        return np.random.randn(*self.shape).astype(np.float32)

class DummyEnv:
    """Dummy environment for testing"""
    def __init__(self, obs_dim=4, action_dim=2):
        self.observation_space = DummySpace((obs_dim,))
        self.action_space = DummySpace((action_dim,))
    
    def reset(self):
        return self.observation_space.sample()

def test_expo_creation():
    """Test if Expo agent can be created successfully"""
    print("Testing Expo agent creation...")
    
    # Create a simple dummy environment for testing
    env = DummyEnv(obs_dim=4, action_dim=2)
    
    try:
        agent = Expo.create(
            seed=42,
            observation_space=env.observation_space,
            action_space=env.action_space,
            hidden_dims=(64, 64),
            device='cpu'
        )
        print("✓ Expo agent created successfully")
        return agent, env
    except Exception as e:
        print(f"✗ Failed to create Expo agent: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_action_sampling(agent, env):
    """Test action sampling functionality"""
    print("\nTesting action sampling...")
    
    try:
        obs = env.observation_space.sample()
        
        # Test evaluation actions
        eval_actions, _ = agent.eval_actions(obs)
        print(f"✓ Eval actions shape: {eval_actions.shape}")
        
        # Test sample actions  
        sample_actions, _ = agent.sample_actions(obs)
        print(f"✓ Sample actions shape: {sample_actions.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Failed action sampling: {e}")
        return False

def test_batch_update(agent):
    """Test batch update functionality"""
    print("\nTesting batch update...")
    
    try:
        # Create dummy batch data
        batch_size = 32
        obs_dim = 4  # Environment observation dimension
        action_dim = 2  # Environment action dimension (continuous version)
        
        batch = {
            "observations": np.random.randn(batch_size, obs_dim).astype(np.float32),
            "actions": np.random.randn(batch_size, action_dim).astype(np.float32),
            "rewards": np.random.randn(batch_size).astype(np.float32),
            "next_observations": np.random.randn(batch_size, obs_dim).astype(np.float32),
            "masks": np.random.randint(0, 2, batch_size).astype(np.float32)
        }
        
        # Test update
        updated_agent, mini_batch, info = agent.update(batch, utd_ratio=1)
        
        print(f"✓ Update completed successfully")
        print(f"  - Actor loss: {info.get('edit_loss', 'N/A')}")
        print(f"  - Critic loss: {info.get('critic_loss', 'N/A')}")
        print(f"  - Temperature: {info.get('temperature', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"✗ Failed batch update: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state_dict(agent):
    """Test state dict save/load functionality"""
    print("\nTesting state dict functionality...")
    
    try:
        # Save state dict
        state_dict = agent.state_dict()
        print(f"✓ State dict saved with keys: {list(state_dict.keys())}")
        
        # Test loading (just verify the method exists and works)
        agent.load_state_dict(state_dict)
        print("✓ State dict loaded successfully")
        
        return True
    except Exception as e:
        print(f"✗ Failed state dict test: {e}")
        return False

def test_model_components():
    """Test individual model components"""
    print("\nTesting model components...")
    
    try:
        from src.model.mlp import MLP
        from src.model.state_action_value import StateActionValue
        from src.distributions.tanh_normal import TanhNormal
        from src.model.ensemble import Ensemble
        
        # Test MLP
        mlp = MLP(hidden_dims=[10, 64, 32])
        x = torch.randn(5, 10)
        output = mlp(x)
        print(f"✓ MLP test passed: input {x.shape} -> output {output.shape}")
        
        # Test StateActionValue
        sav = StateActionValue(
            base_cls=lambda **kwargs: MLP(**kwargs),
            input_dim=14,  # obs_dim + action_dim
            hidden_dims=[64, 32]
        )
        obs = torch.randn(5, 4)
        actions = torch.randn(5, 10)
        q_values = sav(obs, actions)
        print(f"✓ StateActionValue test passed: output shape {q_values.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Failed model components test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=== PyTorch Conversion Verification ===\n")
    
    # Test individual components first
    if not test_model_components():
        print("\n❌ Model components test failed. Stopping.")
        return
    
    # Test main functionality
    agent, env = test_expo_creation()
    if agent is None:
        print("\n❌ Cannot proceed without agent creation.")
        return
    
    success = True
    success &= test_action_sampling(agent, env)
    success &= test_state_dict(agent)
    success &= test_batch_update(agent)  # Now enabled
    
    if success:
        print("\n🎉 All tests passed! PyTorch conversion is successful.")
    else:
        print("\n❌ Some tests failed. Please check the conversion.")

if __name__ == "__main__":
    main()
