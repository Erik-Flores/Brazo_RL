import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# --- State Normalizer ---
class StateNormalizer:
    def __init__(self, observation_space_dims):
        self.mean = np.zeros(observation_space_dims)
        self.std = np.ones(observation_space_dims)
        self.count = 0
        self.running_sum = np.zeros(observation_space_dims)
        self.running_sum_sq = np.zeros(observation_space_dims)

    def normalize(self, obs):
        if self.count > 0:
            return (obs - self.mean) / (self.std + 1e-8) # Add small epsilon for stability
        return obs

    def update(self, obs_batch):
        # Flatten batch for easier calculation if obs_batch is 2D (batch_size, obs_dim)
        if obs_batch.ndim == 1:
            obs_batch = obs_batch[np.newaxis, :] # Make it 2D if it's a single observation

        self.running_sum += np.sum(obs_batch, axis=0)
        self.running_sum_sq += np.sum(np.square(obs_batch), axis=0)
        self.count += obs_batch.shape[0]

        if self.count > 0:
            self.mean = self.running_sum / self.count
            # Calculate variance as E[X^2] - (E[X])^2
            variance = (self.running_sum_sq / self.count) - np.square(self.mean)
            self.std = np.sqrt(variance + 1e-8) # Add small epsilon for stability
            # Ensure std is never too small to prevent division by zero
            self.std[self.std < 1e-2] = 1e-2 # Minimum standard deviation


# --- Actor-Critic Network ---
class ActorCriticNetwork(nn.Module):
    def __init__(self, observation_space_dims, action_space_dims):
        super(ActorCriticNetwork, self).__init__()

        self.common_fc1 = nn.Linear(observation_space_dims, 256) 
        self.common_relu1 = nn.ReLU()
        self.common_fc2 = nn.Linear(256, 256)
        self.common_relu2 = nn.ReLU()

        self.actor_mean_layer = nn.Linear(256, action_space_dims)
        # Initialize log_std to a small negative value for less initial exploration, or 0.0 for more
        self.log_std = nn.Parameter(torch.full((action_space_dims,), 0.0)) 

        self.critic_value_layer = nn.Linear(256, 1)

    def forward(self, x):
        common_output = self.common_relu1(self.common_fc1(x))
        common_output = self.common_relu2(self.common_fc2(common_output))

        # Use tanh to constrain action means to [-1, 1], matching action_scale
        mean = torch.tanh(self.actor_mean_layer(common_output)) 
        std = torch.exp(self.log_std)
        
        # Clamp std to avoid extremely small or large values
        std = torch.clamp(std, 0.1, 1.0) # Example: std between 0.1 and 1.0

        action_distribution = Normal(mean, std)
        
        value = self.critic_value_layer(common_output)
        
        return action_distribution, value


# --- PPO Agent Class ---
class PPOAgent:
    def __init__(self, observation_space_dims, action_space_dims,
                 clip_param=0.2, ppo_epochs=10, mini_batch_size=64,
                 actor_lr=3e-4, gamma=0.99, gae_lambda=0.95, entropy_coef=0.05):
        
        self.actor_critic = ActorCriticNetwork(observation_space_dims, action_space_dims)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=actor_lr)

        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef

        self.state_normalizer = StateNormalizer(observation_space_dims)

    def get_action_and_value(self, obs):
        self.state_normalizer.update(obs) # Update normalizer with current observation
        normalized_obs = self.state_normalizer.normalize(obs)
        obs_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0)
        
        action_dist, value = self.actor_critic(obs_tensor)
        action = action_dist.sample() # Sample action from distribution
        
        # Calculate log_prob for the sampled action
        log_prob = action_dist.log_prob(action).sum(dim=-1) 
        
        return action.squeeze(0).detach().numpy(), log_prob.item(), value.item()

    def evaluate_action_and_value(self, obs_tensor, action_tensor):
        action_dist, value = self.actor_critic(obs_tensor)
        log_prob = action_dist.log_prob(action_tensor).sum(dim=-1)
        entropy = action_dist.entropy().sum(dim=-1) 
        return log_prob, value, entropy

    # --- MISSING: compute_gae function ---
    def compute_gae(self, rewards, dones, values):
        advantages = []
        returns = []
        last_gae_lambda = 0
        
        # Ensure values includes the final value for the last state in the rollout
        # If the episode is done, the value of the next state is 0.
        # This typically means values should be length (batch_size + 1) or handled carefully.
        # For simplicity here, we assume values[i] is the value of state[i].
        # The very last value in the 'values' list should correspond to the value of the *next* state after the last observation in the rollout.
        # If the rollout ends because the episode is done, this next_value is 0.
        
        # This implementation requires values to include the value of the next state *after* the last observation in the rollout.
        # So if you have N (obs, action, reward, done) tuples, you need N+1 value estimates.
        # The last value is the value of the state *after* the last reward was received.
        # If done is True for the last step, then the last value in 'values' should be 0.
        
        # Let's adjust for the common case where 'values' are for the states *within* the rollout.
        # We need the value of the state *after* the last state in the rollout for the GAE calculation.
        # If the rollout exactly matches batch_size and has a next value, values will be len(rewards) + 1.
        # If it's a list of values for each state in the rollout:
        
        # Assuming values list is [V(s_0), V(s_1), ..., V(s_{N-1}), V(s_N)] where N is batch_size
        # V(s_N) is the value of the state *after* the last action/reward in the rollout.
        # If dones[N-1] is True, V(s_N) should be 0.
        
        # For simplicity, let's pass state_values (list of values for each state in the rollout)
        # and ensure the last_value is also passed.
        # Example: rewards = [r0, r1, r2], dones = [d0, d1, d2], values = [v0, v1, v2]
        # We need the value of the state *after* s2.
        
        # Re-evaluating based on how PPOAgent is called in RLTrainer.
        # RLTrainer collects `batch_obs`, and then calls `self.agent.actor_critic(torch.FloatTensor(obs).unsqueeze(0))[1].item()`
        # for each obs in `batch_obs`. So `values` here is `[V(s_0), ..., V(s_{N-1})]`.
        # We need the value of the last state *after* the last reward, or 0 if terminal.
        
        # For the last step, if `dones[-1]` is True, the next state value is 0.
        # Otherwise, we need to estimate the value of the actual next state.
        # The RLTrainer typically calls `agent.compute_gae(batch_rewards, batch_dones, values_of_states_in_rollout_except_last_one)`
        # This is often handled by computing the value of the final observation in the rollout.

        # Correct GAE implementation (assuming values contains V(s) for each s in rollout, N values)
        # We need V(s_T) as next_value for the last step
        
        # The `values` parameter here should be the values for the states IN the batch.
        # We need the value of the state AFTER the last observation in the batch (or 0 if done).
        
        # Simplified GAE calculation:
        # rewards, dones, values are all of length N (timesteps_per_batch)
        # We need to get value of the state *after* the last reward if not done.
        
        # Let's assume `values` passed to this function is `[V(s_0), V(s_1), ..., V(s_{N-1})]`.
        # And `batch_rewards`, `batch_dones` are also length N.
        
        # The GAE calculation needs the value of the "next state".
        # So we iterate backwards.
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1: # Last step in the rollout
                # If the episode ended, next_value is 0. Else, it's value of current state from critic.
                # Here, we assume `values` provided already represent V(s_t).
                # So if dones[t] is True, the value of the next state (which doesn't exist) is 0.
                # Otherwise, if the episode *continues* beyond the rollout, the next_value would be V(s_t+1)
                # which isn't directly available in this `values` list.
                # A robust GAE implementation would usually pass V(s_t+1) explicitly.
                
                # Let's assume `values` contains V(s_t) for all t in the rollout.
                # If the episode was done at `t`, then `delta` calculation should use 0 for the future term.
                next_value = 0 # If episode is done, or if this is the last step in a truncated rollout
                               # (this often needs a more explicit V(s_T) from the environment)
            else:
                next_value = values[t+1] * (1 - dones[t+1]) # Only use next value if not done

            # If current step is done, future rewards and values don't matter, only current reward.
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda * (1 - dones[t])
            advantages.insert(0, last_gae_lambda) # Insert at beginning to reverse order
            
            returns.insert(0, advantages[0] + values[t]) # Returns = advantages + values

        return np.array(returns), np.array(advantages)


    def update(self, batch_obs, batch_actions, batch_log_probs, batch_returns, batch_advantages):
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

                # Ensure values are consistent types
                mini_obs_tensor = mini_obs
                mini_actions_tensor = mini_actions
                mini_old_log_probs_tensor = mini_old_log_probs
                mini_returns_tensor = mini_returns
                mini_advantages_tensor = mini_advantages

                new_log_probs, new_values, entropy = self.evaluate_action_and_value(mini_obs_tensor, mini_actions_tensor)
                
                critic_loss = (new_values.squeeze() - mini_returns_tensor).pow(2).mean()

                ratio = torch.exp(new_log_probs - mini_old_log_probs_tensor)
                surr1 = ratio * mini_advantages_tensor
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mini_advantages_tensor
                
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean() 

                self.optimizer.zero_grad()
                total_loss = actor_loss + critic_loss
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                num_minibatches += 1
        
        return total_actor_loss / num_minibatches, total_critic_loss / num_minibatches