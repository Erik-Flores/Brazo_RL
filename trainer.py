import pybullet as p
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

from src.robot_env import RobotEnv
from src.agent import PPOAgent

# --- Trainer Class (Main Training Loop) ---
class RLTrainer:
    def __init__(self, env, agent, timesteps_per_batch=2048):
        self.env = env
        self.agent = agent
        self.timesteps_per_batch = timesteps_per_batch

    def compute_gae(self, rewards, values, next_value, done_mask):
        values_tensor = torch.FloatTensor(values)
        rewards_tensor = torch.FloatTensor(rewards)
        done_mask_tensor = torch.FloatTensor(done_mask)

        full_values = torch.cat((values_tensor, torch.FloatTensor([next_value])))

        advantages = torch.zeros_like(rewards_tensor)
        last_gae_lambda = 0
        for t in reversed(range(len(rewards_tensor))):
            delta = rewards_tensor[t] + self.agent.gamma * full_values[t+1] * done_mask_tensor[t] - full_values[t]
            advantages[t] = delta + self.agent.gamma * self.agent.gae_lambda * last_gae_lambda * done_mask_tensor[t]
            last_gae_lambda = advantages[t]
        
        returns = advantages + values_tensor

        return advantages.numpy(), returns.numpy()

    def train(self, num_iterations):
        print(f"Iniciando entrenamiento de RL por {num_iterations} iteraciones (batch updates)...")
        
        if not os.path.exists('saves'):
            os.makedirs('saves')

        global_timestep = 0
        
        for iteration in range(num_iterations):
            batch_obs, batch_actions, batch_log_probs = [], [], []
            batch_advantages = []
            batch_returns = []
            
            current_timesteps_collected = 0
            
            while current_timesteps_collected < self.timesteps_per_batch:
                observation = self.env.reset()
                done = False
                episode_rewards = []
                episode_observations = []
                episode_actions = []
                episode_log_probs = []
                episode_values = []
                episode_dones = []

                self.agent.state_normalizer.update(observation.reshape(1, -1))

                while not done and current_timesteps_collected < self.timesteps_per_batch:
                    action, log_prob, value = self.agent.get_action_and_value(observation)
                    
                    episode_observations.append(observation) 
                    episode_actions.append(action)
                    episode_log_probs.append(log_prob)
                    episode_values.append(value)

                    observation, reward, done, info = self.env.step(action)
                    
                    episode_rewards.append(reward)
                    episode_dones.append(0 if done and self.env.current_episode_steps < self.env.max_episode_steps else 1)
                    
                    current_timesteps_collected += 1
                    global_timestep += 1

                    self.agent.state_normalizer.update(observation.reshape(1, -1))

                    if self.env.render_mode:
                        time.sleep(1./240.)

                next_value = 0.0
                if not done:
                    _, next_value_tensor = self.agent.actor_critic(
                        torch.FloatTensor(self.agent.state_normalizer.normalize(observation)).unsqueeze(0)
                    )
                    next_value = next_value_tensor.item()
                
                current_advantages, current_returns = self.compute_gae(
                    episode_rewards, episode_values, next_value, episode_dones
                )

                batch_obs.extend(episode_observations)
                batch_actions.extend(episode_actions)
                batch_log_probs.extend(episode_log_probs)
                batch_advantages.extend(current_advantages) 
                batch_returns.extend(current_returns)

                total_episode_reward = sum(episode_rewards)
                print(f"  Episodio finalizado. Recompensa: {total_episode_reward:.2f}, Pasos: {len(episode_rewards)}")
                
            actor_loss, critic_loss = self.agent.update(
                batch_obs, batch_actions, batch_log_probs, batch_returns, batch_advantages
            )

            print(f"Iteración {iteration+1}/{num_iterations} (Timestep {global_timestep}). "
                  f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
            
            if (iteration + 1) % 50 == 0:
                model_path = os.path.join('saves', f'ppo_robot_reach_iter_{iteration+1}.pth')
                torch.save(self.agent.actor_critic.state_dict(), model_path)
                print(f"Modelo guardado en {model_path}")

        self.env.close()
        print("Entrenamiento de RL finalizado.")

if __name__ == "__main__":
    env = RobotEnv(render=True)

    observation_space_dims = env.observation_space_dims
    action_space_dims = env.action_space_dims 

    ppo_agent = PPOAgent(
        observation_space_dims=observation_space_dims,
        action_space_dims=action_space_dims,
        actor_lr=1e-4, 
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.3, # Puedes probar a aumentar este valor a 0.3 o 0.4 para más exploración inicial
        ppo_epochs=10,
        mini_batch_size=64,
        entropy_coef=0.05 # Puedes aumentar este valor a 0.05 o 0.1 para más exploración
    )

    trainer = RLTrainer(env, ppo_agent, timesteps_per_batch=2048)
    trainer.train(num_iterations=100)