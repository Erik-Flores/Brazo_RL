import pybullet as p
import numpy as np
import torch
import os
import time

from src.robot_env import RobotEnv
from src.agent import PPOAgent, StateNormalizer # Importa StateNormalizer también

# --- CONFIGURACIÓN PARA CARGAR EL AGENTE ---
LOAD_MODEL_PATH = "trained_models/final_agent.pth" # Ajusta al nombre de tu archivo guardado
# ------------------------------------------

def run_trained_agent(env, agent, num_episodes=10):
    print(f"\nCargando agente desde: {LOAD_MODEL_PATH}")
    checkpoint = torch.load(LOAD_MODEL_PATH)

    # Cargar los pesos de la red ActorCritic
    agent.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
    
    # Cargar el estado del optimizador (opcional, si quisieras continuar entrenando)
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
    
    # Cargar los parámetros del normalizador de estado (¡CRÍTICO!)
    agent.state_normalizer.mean = checkpoint['state_normalizer_mean']
    agent.state_normalizer.std = checkpoint['state_normalizer_std']
    
    # (Opcional) Restaurar el contador de videos si lo guardaste
    if 'video_counter' in checkpoint:
        env.video_counter = checkpoint['video_counter']

    print("Agente cargado exitosamente. Ejecutando...")

    # Poner el modelo en modo evaluación (desactiva dropout/batchnorm si los hubiera)
    agent.actor_critic.eval() 

    total_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        # Para grabar el episodio si deseas, puedes descomentar estas líneas
        # video_filename = f"run_episode_{episode:02d}.mp4"
        # log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, os.path.join("videos", video_filename))
        # print(f"Grabando ejecución del episodio {episode} en {video_filename}")

        while not done:
            # Desactivar la aleatoriedad (exploración) del actor para una ejecución determinista
            # Poner el actor en modo evaluación ya ayuda, pero asegurar la acción promedio es más directo.
            # Puedes probar get_action_and_value que ya maneja esto, pero para explotar
            # a veces quieres una acción totalmente determinista sin muestreo.
            # Aquí lo dejamos con get_action_and_value para consistencia, pero su `sample()`
            # aun introduce aleatoriedad. Si quieres determinismo puro, tendrías que
            # acceder directamente al `mean` de la distribución.
            
            action, log_prob, value = agent.get_action_and_value(obs)
            
            # Si quieres una ejecución más determinista, puedes hacer esto (requiere modificar get_action_and_value en agent.py)
            # action = agent.get_deterministic_action(obs) 
            
            next_obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            obs = next_obs
            steps += 1
            # time.sleep(1./240.) # Puedes añadir esto para ralentizar la simulación si la GUI va muy rápido

        total_rewards.append(episode_reward)
        print(f"Episodio {episode+1}/{num_episodes} terminado. Recompensa: {episode_reward:.2f}, Pasos: {steps}")

        # if log_id != -1:
        #     p.stopStateLogging(log_id)
        #     log_id = -1

    print(f"\nRecompensa promedio en {num_episodes} episodios: {np.mean(total_rewards):.2f}")
    env.close()

if __name__ == "__main__":
    env_run = RobotEnv(render=True) # Render en True para ver la ejecución
    
    # Necesitas inicializar el agente con las mismas dimensiones que usaste para entrenar
    observation_space_dims = env_run.observation_space_dims
    action_space_dims = env_run.action_space_dims 

    # Se inicializa con valores por defecto, pero se cargarán del checkpoint
    loaded_agent = PPOAgent(
        observation_space_dims=observation_space_dims,
        action_space_dims=action_space_dims
    )

    run_trained_agent(env_run, loaded_agent, num_episodes=10) # Ejecutar 10 episodios