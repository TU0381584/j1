# Phase 2b: A2C Implementation Roadmap

**Status**: Scaffold ready, training algorithm pending  
**Estimated duration**: 1-2 weeks  
**Timeline**: After Phase 3 pilot DQN results

## Overview

A2C (Advantage Actor-Critic) will provide a policy-gradient alternative to DQN's value-based learning. This enables:
- On-policy learning (potentially faster convergence)
- Natural entropy regularization
- Better exploration-exploitation tradeoff
- Comparison of algorithmic paradigms for paper

## Architecture

### Actor Network (Policy)
```
Input: state [state_dim=18]
Hidden-1: ReLU(18 → 128)
Hidden-2: ReLU(128 → 128)
Output: action_logits [action_dim=432]
         → softmax → π(a|s)
```

### Critic Network (Value)
```
Input: state [state_dim=18]
Shared with actor (same hidden layers)
Output: V(s) (scalar value estimate)
```

### Shared Representation
```
Actor and Critic share hidden layers 1-2
Network outputs: action_logits, value
```

## Training Algorithm: A2C

### Step 1: Collect trajectory
```
Run N episodes (or fixed T timesteps)
Collect: states, actions, rewards, next_states
```

### Step 2: Compute advantages
```
For each step t:
  V_target = reward_t + γ * V(s_{t+1}) if not done
           = reward_t if done
  advantage = V_target - V(s_t)
```

### Step 3: Update actors

**Actor loss** (policy gradient):
```
L_actor = - log π(a_t | s_t) * advantage_t + β_entropy * H[π]
where H[π] = - Σ π(a|s) * log π(a|s)
```

**Critic loss** (value estimation):
```
L_critic = (V_target - V(s_t))^2
```

**Total loss**:
```
L = L_actor + L_critic + L_entropy_regularization
```

### Step 4: Optimize
```
∇_θ = gradient of total loss
θ ← θ - α * ∇_θ
```

## Implementation Checklist

- [ ] Implement `A2CTrainer` class (parallel to `DRLTrainer`)
  - Trajectory collection (buffer for one episode)
  - Advantage computation
  - Policy + value updates
  - Checkpoint save/load

- [ ] Update `A2CPolicy.train_step()` implementation
  - Currently raises `NotImplementedError`
  - Compute actor + critic losses
  - Backward pass & optimization

- [ ] Create A2C configs
  - `rfsim_a2c_training.yaml`
  - `rfsim_a2c_inference.yaml`

- [ ] Add hyperparameter configs
  - Entropy coefficient (β_entropy): try 0.01
  - Learning rate: try 1e-3
  - Batch size: 32 (episodes before update)
  - Discount factor (γ): 0.99

- [ ] Testing & validation
  - Single 50-step run
  - Multi-seed (3 seeds) training
  - Compare convergence vs DQN

## Code Location

Primary changes in:
- `oranslice_drl/drl_policy.py`: Update `A2CPolicy.train_step()`
- `oranslice_drl/drl_training.py`: Add `A2CTrainer` class
- `oranslice_drl/controllers.py`: Update `A2CController` (minor)
- `configs/`: Add `rfsim_a2c_training.yaml` and `rfsim_a2c_inference.yaml`

## Expected Behavior

### Convergence
- DQN: More stable, slower initial learning
- A2C: Faster initial learning, potentially more unstable
- Both should show improving rewards over 300 steps

### Comparison metrics
```
DQN vs A2C:
  - Final reward (300 steps)
  - SLA violation rate
  - Stability (reward variance)
  - Training time
```

## Phase 2b Milestones

### Week 1
- [ ] Implement A2CTrainer trajectory collection
- [ ] Implement advantage computation
- [ ] Test on small episodes (10 steps)

### Week 2
- [ ] Implement policy + value updates
- [ ] Hyperparameter tuning (entropy β, LR)
- [ ] Multi-seed validation (5 seeds, 3 each)
- [ ] Generate convergence plots

### Week 3 (optional)
- [ ] Compare DQN vs A2C results
- [ ] Write comparison section for paper
- [ ] Ablation: entropy coefficient sensitivity

## Starting Point: Copy-Paste Template

When ready to implement, start here:

```python
# In oranslice_drl/drl_training.py

class A2CTrainer(DRLTrainer):
    """A2C-specific trainer with trajectory accumulation."""
    
    def __init__(self, policy, state_encoder, action_decoder, batch_size=32):
        super().__init__(policy, state_encoder, action_decoder, batch_size)
        self.trajectory = {
            'states': [], 'actions': [], 'rewards': [], 
            'values': [], 'next_states': []
        }
    
    def step(self, state, action, reward, next_state, done):
        """Accumulate trajectory."""
        self.trajectory['states'].append(state_encoder.encode(state))
        self.trajectory['actions'].append(action)
        self.trajectory['rewards'].append(reward)
        self.trajectory['next_states'].append(state_encoder.encode(next_state))
        
        if done or len(self.trajectory['states']) >= self.batch_size:
            self._train_on_trajectory()
    
    def _train_on_trajectory(self):
        # Compute advantages, update networks
        pass
```

## Integration with Runner

No changes needed to `runner.py`—A2C will work with existing infrastructure:

```yaml
controller:
  mode: a2c_train    # New mode, same runner
  # config...
```

## Success Criteria

- [x] A2CPolicy loads without error
- [x] Forward pass on dummy state works
- [ ] A2C training updates weights
- [ ] Convergence plot shows improving reward
- [ ] A2C outperforms ≥1 baseline
- [ ] DQN vs A2C comparison meaningful

## Connection to Paper

**Section 4: Proposed Method**
- Describe DQN (Phase 2 ✓)
- Describe A2C (Phase 2b)
- Explain why two algorithms

**Section 5: Evaluation**
- DQN results (Phase 3)
- A2C results (Phase 2b + Phase 3)
- Direct comparison table/figure

**Contribution**: "Comparative study of value-based (DQN) vs. policy-based (A2C) RL for 5G slicing"

---

## Parallelization Strategy

**Don't wait for A2C to finish Phase 3**:
1. Use Phase 3 to run comprehensive DQN evaluation (all scenarios, seeds)
2. Start A2C implementation in parallel
3. When A2C ready, run same Phase 3 matrix for A2C
4. Final comparison with both algorithms on identical scenarios

**Timeline**:
- Weeks 1-2: Phase 3 (DQN comprehensive eval)
- Weeks 1-2 (parallel): Phase 2b (A2C implementation)
- Weeks 3-4: Phase 3 (A2C comprehensive eval)
- Week 5: Compare results, write paper section 5

This parallelization keeps you on track for publication.

---

**Phase 2b is optional for initial paper but recommended for stronger contribution.**
