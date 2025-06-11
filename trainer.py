import pybullet as p
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

from src.robot_env import RobotEnv
from src.agent import PPOAgent

# --- AGREGAR EL DIRECTORIO PARA GUARDAR LOS MODELOS ---
SAVE_DIR = "trained_models"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
# ----------------------------------------------------

class RLTrainer:
    def __init__(self, env, agent, timesteps_per_batch=2048):
        self.env = env
        self.agent = agent
        self.timesteps_per_batch = timesteps_per_batch
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []

    def train(self, num_iterations):
        for iteration in range(num_iterations):
            batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_dones = self._collect_rollout()
            
            # Compute returns and advantages
            batch_returns, batch_advantages = self.agent.compute_gae(batch_rewards, batch_dones, 
                                                                    [self.agent.actor_critic(torch.FloatTensor(obs).unsqueeze(0))[1].item() 
                                                                     for obs in batch_obs]) # Pass value estimates
            
            # Convert to tensors
            batch_obs_tensor = torch.FloatTensor(self.agent.state_normalizer.normalize(batch_obs))
            batch_actions_tensor = torch.FloatTensor(batch_actions)
            batch_log_probs_tensor = torch.FloatTensor(batch_log_probs)
            batch_returns_tensor = torch.FloatTensor(batch_returns)
            batch_advantages_tensor = torch.FloatTensor(batch_advantages)
            
            # Normalize advantages
            batch_advantages_tensor = (batch_advantages_tensor - batch_advantages_tensor.mean()) / (batch_advantages_tensor.std() + 1e-8)

            actor_loss, critic_loss = self.agent.update(
                batch_obs_tensor, batch_actions_tensor, batch_log_probs_tensor, 
                batch_returns_tensor, batch_advantages_tensor
            )
            
            self.actor_losses.append(actor_loss)
            self.critic_losses.append(critic_loss)

            avg_reward = np.mean(self.episode_rewards[-self.env.max_episode_steps:]) if self.episode_rewards else 0 # Example for a recent average
            print(f"Iteración {iteration+1}/{num_iterations} (Timestep {(iteration+1) * self.timesteps_per_batch}). Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
            print(f"Recompensa promedio (últimos {len(self.episode_rewards)} episodios): {np.mean(self.episode_rewards):.2f}")
            
            # --- GUARDAR EL AGENTE ---
            # Guarda el agente cada X iteraciones o si el rendimiento es bueno
            if (iteration + 1) % 50 == 0: # Guardar cada 50 iteraciones
                self.save_agent(f"agent_iteration_{iteration+1}.pth")
            # ------------------------

    def _collect_rollout(self):
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_dones = []
        
        current_timesteps = 0
        while current_timesteps < self.timesteps_per_batch:
            obs = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            while not done:
                action, log_prob, value = self.agent.get_action_and_value(obs)
                next_obs, reward, done, info = self.env.step(action)
                
                batch_obs.append(obs)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_rewards.append(reward)
                batch_dones.append(done)

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                current_timesteps += 1

                if current_timesteps >= self.timesteps_per_batch:
                    done = True # Forzar el fin del rollout para el batch

            self.episode_rewards.append(episode_reward)
            print(f"  Episodio finalizado. Recompensa: {episode_reward:.2f}, Pasos: {episode_steps}")

        return np.array(batch_obs), np.array(batch_actions), np.array(batch_log_probs), np.array(batch_rewards), np.array(batch_dones)

    # --- NUEVA FUNCIÓN PARA GUARDAR EL AGENTE ---
    def save_agent(self, filename):
        filepath = os.path.join(SAVE_DIR, filename)
        torch.save({
            'actor_critic_state_dict': self.agent.actor_critic.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'state_normalizer_mean': self.agent.state_normalizer.mean,
            'state_normalizer_std': self.agent.state_normalizer.std,
            'video_counter': self.env.video_counter # Si quieres guardar el contador de videos
        }, filepath)
        print(f"Agente guardado en: {filepath}")
    # ---------------------------------------------


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
        clip_param=0.3, 
        ppo_epochs=10,
        mini_batch_size=64,
        entropy_coef=0.05 
    )

    trainer = RLTrainer(env, ppo_agent, timesteps_per_batch=2048)
    trainer.train(num_iterations=50)

    # --- GUARDAR EL AGENTE AL FINAL DEL ENTRENAMIENTO ---
    trainer.save_agent("final_agent.pth")
    # ----------------------------------------------------

    env.close()