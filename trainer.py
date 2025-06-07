import pybullet as p
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal # Necesitamos esto para muestrear acciones y obtener log_prob
from src.robot_env import RobotEnv
# from src.agent import RandomAgent # Ya no necesitamos RandomAgent para el entrenamiento de la política
import time
import os

# ... (tus constantes de cámara y entorno) ...

class PolicyNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        
        # Para manejar la parte de la imagen, idealmente aquí habría una CNN.
        # Por ahora, asumimos que 'observation_dim' ya incluye la imagen aplanada.
        
        self.fc1 = nn.Linear(observation_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        
        # MODIFICACIÓN CLAVE: La red debe producir la media y la desviación estándar
        # de una distribución de acciones (ej. Normal).
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim) # Para la desviación estándar (logaritmo)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        
        mean = torch.tanh(self.mean_layer(x)) # Salida de la media entre -1 y 1
        # La desviación estándar debe ser positiva. Usamos exp() del log_std.
        # Clamp para evitar valores muy pequeños o grandes.
        log_std = torch.clamp(self.log_std_layer(x), min=-20, max=2) # Valores de log_std razonables
        std = torch.exp(log_std)
        
        # Creamos una distribución Normal (multivariante)
        dist = Normal(mean, std)
        
        return dist # Retornamos la distribución, no la acción directamente

# --- CLASE PRINCIPAL DEL ENTRENADOR ---
class RLTrainer:
    def __init__(self, env, policy_network, learning_rate=1e-4, gamma=0.99):
        self.env = env
        self.policy_network = policy_network
        self.optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma

    def collect_experience(self):
        observations = []
        actions = []
        rewards = []
        log_probs = [] # Aquí almacenaremos los log_probs que SÍ requieren gradiente

        observation = self.env.reset()
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

            # MODIFICACIÓN CLAVE: Usar la distribución de la red para muestrear la acción
            dist = self.policy_network(obs_tensor) # Obtenemos la distribución
            action_tensor = dist.sample() # Muestreamos una acción de la distribución
            
            # Calculamos el log_prob de la acción muestreada. Esto SÍ requiere gradiente.
            log_prob = dist.log_prob(action_tensor).sum(axis=-1) # Suma sobre las dimensiones de la acción

            action = action_tensor.squeeze(0).detach().numpy() # Convertir a numpy para el entorno

            next_observation, reward, done, info = self.env.step(action)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob) # Guardar el log_prob que es un tensor con gradiente

            observation = next_observation

            if self.env.physicsClient != p.DIRECT:
                time.sleep(1./240.)

        return observations, actions, rewards, log_probs

    def calculate_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        # Convertir a tensor de PyTorch (NO necesita gradiente si solo se usa para calcular el loss)
        return torch.tensor(returns, dtype=torch.float32)

    def train_step(self, observations, actions, rewards, log_probs):
        self.optimizer.zero_grad()

        returns = self.calculate_returns(rewards)
        # Normalizar returns (opcional, pero buena práctica)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Apilar los log_probs en un solo tensor. Estos *ya* tienen requires_grad=True
        # porque provienen de la distribución que se generó a partir de la PolicyNetwork.
        log_probs_tensor = torch.stack(log_probs).squeeze() # squeeze si es necesario para que sea 1D

        # MODIFICACIÓN CLAVE: El loss para un algoritmo de gradiente de política (tipo REINFORCE)
        # Sumamos sobre las experiencias (episodios) y negamos para maximizar la recompensa.
        # Esto asume que estás haciendo REINFORCE puro. PPO tiene un loss más complejo.
        loss = - (log_probs_tensor * returns).mean() # Usar .mean() en lugar de .sum() para estabilidad

        loss.backward() # Ahora 'loss' debería tener un grad_fn
        self.optimizer.step()

        print(f"  Loss: {loss.item():.4f}")

    def train(self, num_episodes):
        print(f"Iniciando entrenamiento de RL por {num_episodes} episodios...")
        for episode in range(num_episodes):
            observations, actions, rewards, log_probs = self.collect_experience()
            
            if not log_probs: # Evitar error si el episodio terminó demasiado rápido sin acciones
                print(f"Episodio {episode+1} muy corto, no se colectaron experiencias para entrenar.")
                continue

            self.train_step(observations, actions, rewards, log_probs)

            total_episode_reward = sum(rewards)
            print(f"Episodio {episode+1}/{num_episodes} terminado. Recompensa: {total_episode_reward:.2f}, Pasos: {len(rewards)}")

        self.env.close()
        print("Entrenamiento de RL finalizado.")


if __name__ == "__main__":
    env = RobotEnv(render=True)

    # La PolicyNetwork ahora necesita action_dim para la media y log_std
    policy_net = PolicyNetwork(env.observation_space_dims, env.action_space_dims)

    trainer = RLTrainer(env, policy_net, learning_rate=3e-4)

    trainer.train(num_episodes=10)