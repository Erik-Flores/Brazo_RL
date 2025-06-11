import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# --- State Normalizer ---
class StateNormalizer:
    def __init__(self, observation_dims):
        self.mean = np.zeros(observation_dims, dtype=np.float32)
        self.std = np.ones(observation_dims, dtype=np.float32)
        self.n_samples = 0
        self.epsilon = 1e-8 # Para evitar división por cero

    def update(self, observations):
        observations = np.array(observations, dtype=np.float32)
        batch_mean = np.mean(observations, axis=0)
        batch_std = np.std(observations, axis=0)
        batch_n = observations.shape[0]

        if self.n_samples == 0:
            self.mean = batch_mean
            self.std = batch_std
        else:
            delta = batch_mean - self.mean
            total_n = self.n_samples + batch_n
            
            self.mean = self.mean + delta * batch_n / total_n
            
            new_var = ((self.n_samples * self.std**2) + (batch_n * batch_std**2) + 
                       (delta**2 * self.n_samples * batch_n / total_n)) / total_n
            self.std = np.sqrt(new_var)

        self.n_samples += batch_n
        self.std = np.maximum(self.std, self.epsilon) # Ensure std is never zero

    def normalize(self, observation):
        normalized_obs = (observation - self.mean) / (self.std + self.epsilon)
        return normalized_obs

# --- Actor-Critic Network ---
class ActorCriticNetwork(nn.Module):
    def __init__(self, observation_space_dims, action_space_dims):
        super(ActorCriticNetwork, self).__init__()

        self.common_fc1 = nn.Linear(observation_space_dims, 256)
        self.common_relu1 = nn.ReLU()
        self.common_fc2 = nn.Linear(256, 256)
        self.common_relu2 = nn.ReLU()

        self.actor_mean_layer = nn.Linear(256, action_space_dims)
        self.log_std = nn.Parameter(torch.full((action_space_dims,), -0.5)) 

        self.critic_value_layer = nn.Linear(256, 1)

    def forward(self, x):
        common_output = self.common_relu1(self.common_fc1(x))
        common_output = self.common_relu2(self.common_fc2(common_output))

        mean = torch.tanh(self.actor_mean_layer(common_output))
        std = torch.exp(self.log_std)

        action_distribution = Normal(mean, std)
        
        value = self.critic_value_layer(common_output)
        
        return action_distribution, value


# --- PPO Agent Class ---
class PPOAgent:
    def __init__(self, observation_space_dims, action_space_dims,
                 clip_param=0.2, ppo_epochs=10, mini_batch_size=64,
                 actor_lr=3e-4, gamma=0.99, gae_lambda=0.95): # Eliminado critic_lr
        
        self.actor_critic = ActorCriticNetwork(observation_space_dims, action_space_dims)
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=actor_lr)

        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.state_normalizer = StateNormalizer(observation_space_dims)

    def get_action_and_value(self, obs):
        normalized_obs = self.state_normalizer.normalize(obs)
        obs_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0)
        
        action_dist, value = self.actor_critic(obs_tensor)
        action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action).sum(dim=-1)
        
        return action.squeeze(0).detach().numpy(), log_prob.item(), value.item()

    def evaluate_action_and_value(self, obs_tensor, action_tensor):
        action_dist, value = self.actor_critic(obs_tensor)
        log_prob = action_dist.log_prob(action_tensor).sum(dim=-1)
        entropy = action_dist.entropy().sum(dim=-1)
        return log_prob, value, entropy

    def update(self, batch_obs, batch_actions, batch_log_probs, batch_returns, batch_advantages):
        batch_obs = torch.FloatTensor(np.array(batch_obs))
        batch_actions = torch.FloatTensor(np.array(batch_actions))
        batch_log_probs = torch.FloatTensor(np.array(batch_log_probs))
        batch_returns = torch.FloatTensor(np.array(batch_returns))
        batch_advantages = torch.FloatTensor(np.array(batch_advantages))

        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

        # Para llevar el registro de las pérdidas promedio del batch
        total_actor_loss = 0
        total_critic_loss = 0
        num_minibatches = 0

        for _ in range(self.ppo_epochs):
            indices = np.arange(len(batch_obs))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(batch_obs), self.mini_batch_size):
                end_idx = start_idx + self.mini_batch_size
                mini_batch_indices = indices[start_idx:end_idx]

                mini_obs = batch_obs[mini_batch_indices]
                mini_actions = batch_actions[mini_batch_indices]
                mini_old_log_probs = batch_log_probs[mini_batch_indices]
                mini_returns = batch_returns[mini_batch_indices]
                mini_advantages = batch_advantages[mini_batch_indices]

                new_log_probs, new_values, entropy = self.evaluate_action_and_value(mini_obs, mini_actions)
                
                critic_loss = (new_values.squeeze() - mini_returns).pow(2).mean()

                ratio = torch.exp(new_log_probs - mini_old_log_probs)
                surr1 = ratio * mini_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mini_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.001 * entropy.mean()

                self.optimizer.zero_grad()
                total_loss = actor_loss + critic_loss
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                num_minibatches += 1
        
        # --- CAMBIO CLAVE: Devolver las pérdidas promedio ---
        return total_actor_loss / num_minibatches, total_critic_loss / num_minibatches