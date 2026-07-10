#!/usr/bin/env python3
"""
A2C Policy Validation Test
Validates end-to-end A2C training without testbed dependency.
"""
import sys
import os
import json
import csv
from pathlib import Path

# Add module to path
sys.path.insert(0, str(Path(__file__).parent))

from oranslice_drl.drl_policy import A2CPolicy
from oranslice_drl.drl_training import StateEncoder, ActionDecoder, ReplayBuffer, DRLTrainer
import yaml
import numpy as np

def test_a2c():
    """Test A2C implementation with synthetic data."""
    print("Testing A2C implementation...")
    
    # Initialize policy
    state_dim = 18
    action_dim = 432
    policy = A2CPolicy(state_dim, action_dim, hidden_dim=128, learning_rate=1e-3, gamma=0.99)
    print(f"✓ A2C Policy initialized (state_dim={state_dim}, action_dim={action_dim})")
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=10000)
    print(f"✓ Replay buffer initialized (capacity={replay_buffer.capacity})")
    
    # Generate synthetic transitions
    num_transitions = 100
    for _ in range(num_transitions):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.randint(0, action_dim)
        reward = np.float32(np.random.randn())
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = np.float32(np.random.rand() < 0.1)
        
        replay_buffer.add(state, action, reward, next_state, done)
    print(f"✓ Generated {num_transitions} synthetic transitions")
    
    # Run training steps
    batch_size = 32
    num_train_steps = 10
    metrics_log = []
    
    for step in range(num_train_steps):
        batch = replay_buffer.sample(batch_size)
        metrics = policy.train_step(batch)
        metrics_log.append(metrics)
        
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}: loss={metrics['loss']:.4f}, "
                  f"actor_loss={metrics['actor_loss']:.4f}, "
                  f"critic_loss={metrics['critic_loss']:.4f}, "
                  f"entropy={metrics['entropy']:.4f}")
    
    print(f"✓ A2C training completed ({num_train_steps} steps)")
    
    # Test action selection
    test_state = np.random.randn(state_dim).astype(np.float32)
    action, value = policy.select_action(test_state, training=True)
    assert 0 <= action < action_dim
    assert isinstance(value, float)
    print(f"✓ Action selection works (action={action}, value={value:.4f})")
    
    # Test checkpoint save/load
    ckpt_path = "/tmp/a2c_test_ckpt.pt"
    policy.save_checkpoint(ckpt_path)
    assert os.path.exists(ckpt_path)
    policy.load_checkpoint(ckpt_path)
    os.remove(ckpt_path)
    print(f"✓ Checkpoint save/load works")
    
    # Summary
    print("\n✓ All A2C tests passed!")
    print(f"  Average loss: {np.mean([m['loss'] for m in metrics_log]):.4f}")
    print(f"  Average actor_loss: {np.mean([m['actor_loss'] for m in metrics_log]):.4f}")
    print(f"  Average critic_loss: {np.mean([m['critic_loss'] for m in metrics_log]):.4f}")
    print(f"  Average entropy: {np.mean([m['entropy'] for m in metrics_log]):.4f}")

if __name__ == "__main__":
    test_a2c()
