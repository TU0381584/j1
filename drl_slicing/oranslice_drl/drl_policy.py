"""
DRL Policy implementations: DQN and A2C.
Abstract base class for algorithm-agnostic integration.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RLPolicy(ABC):
    """Abstract base class for RL policies."""

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = False) -> Tuple[int, float]:
        """
        Select action(s) given state.
        
        Returns:
            action: discrete action index or action array
            action_info: auxiliary info (entropy, value, etc.)
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch: Dict) -> Dict:
        """
        Perform one training step on a batch of transitions.
        
        Returns:
            loss_dict: training metrics (loss, grad_norm, etc.)
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        raise NotImplementedError


class QNetwork(nn.Module):
    """Deep Q-Network for slice resource allocation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DQNPolicy(RLPolicy):
    """
    Deep Q-Network (DQN) for discrete action space.
    
    State: continuous, normalized per-slice metrics
    Action: discrete indices for (min_ratio, max_ratio) pairs
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)

        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.train_step_count = 0

    def select_action(self, state: np.ndarray, training: bool = False) -> Tuple[int, float]:
        """
        Epsilon-greedy action selection.
        
        Args:
            state: [state_dim] numpy array
            training: if True, use epsilon-greedy; else greedy
            
        Returns:
            action: discrete index
            q_value: Q-value of selected action
        """
        if training and np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
            q_value = 0.0
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_t)
            action = int(q_values.argmax(dim=1).item())
            q_value = float(q_values.max().item())

        return action, q_value

    def train_step(self, batch: Dict) -> Dict:
        """
        DQN training step.
        
        Batch must contain:
            states: [batch_size, state_dim]
            actions: [batch_size]
            rewards: [batch_size]
            next_states: [batch_size, state_dim]
            dones: [batch_size]
        """
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        # Compute Q(s,a)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q(s',a') using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update target network periodically
        if self.train_step_count % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()),
            "epsilon": self.epsilon,
            "avg_target_q": float(target_q_values.mean().item()),
        }

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_step_count": self.train_step_count,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_network.load_state_dict(ckpt["target_network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.train_step_count = ckpt["train_step_count"]


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for A2C policy (placeholder, implement in phase 2b)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_dim, action_dim)  # policy logits
        self.critic = nn.Linear(hidden_dim, 1)  # value estimate

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(state)
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value


class A2CPolicy(RLPolicy):
    """
    Advantage Actor-Critic (A2C) policy.
    Placeholder for Phase 2b implementation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = torch.device(device)

        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.train_step_count = 0

    def select_action(self, state: np.ndarray, training: bool = False) -> Tuple[int, float]:
        """Sample action from policy distribution."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_logits, value = self.network(state_t)
        probs = torch.softmax(action_logits, dim=1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        return action, float(value.item())

    def train_step(self, batch: Dict) -> Dict:
        """
        A2C training step on batch of transitions.
        
        Algorithm:
        1. Compute value estimates V(s) and V(s')
        2. Compute advantages A(s,a) = r + γ*V(s') - V(s)
        3. Actor loss: -log π(a|s) * A(s,a) (maximize advantage)
        4. Critic loss: MSE(V(s), r + γ*V(s'))
        5. Total loss: actor_loss + critic_loss
        """
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        # Forward pass: get action logits and values
        action_logits, values = self.network(states)
        with torch.no_grad():
            _, next_values = self.network(next_states)

        # Compute advantages: A(s,a) = r + γ*V(s') - V(s)
        target_values = rewards + self.gamma * next_values.squeeze(-1) * (1 - dones)
        advantages = (target_values - values.squeeze(-1)).detach()

        # Actor loss: -log π(a|s) * A(s,a)
        action_dist = torch.distributions.Categorical(torch.softmax(action_logits, dim=1))
        log_probs = action_dist.log_prob(actions)
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss: MSE(V(s), target_V(s))
        critic_loss = nn.functional.mse_loss(values.squeeze(-1), target_values)

        # Total loss with entropy regularization for exploration
        entropy = action_dist.entropy().mean()
        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1

        return {
            "loss": float(total_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
            "grad_norm": float(grad_norm.item()),
            "avg_advantage": float(advantages.mean().item()),
            "avg_value": float(values.mean().item()),
        }

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_step_count": self.train_step_count,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.train_step_count = ckpt["train_step_count"]
