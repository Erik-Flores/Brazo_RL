from src.robot_env import RobotEnv
from src.agent import RandomAgent
import numpy as np
import pybullet as p
import time

def train(num_episodes=1000):
    """
    Función principal para entrenar al agente.
    """
    # Inicializa el entorno (con visualización para ver el progreso)
    env = RobotEnv(render=True)

    # Inicializa el agente (por ahora, un agente aleatorio)
    # action_space_dims viene del entorno (7 articulaciones + 1 empuje = 8)
    agent = RandomAgent(action_space_dims=env.action_space_dims)

    print(f"Iniciando entrenamiento por {num_episodes} episodios...")
    total_steps = 0
    total_rewards = []

    for episode in range(num_episodes):
        observation = env.reset() # Reinicia el entorno para un nuevo episodio
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done:
            # El agente elige una acción basada en la observación actual
            action = agent.choose_action(observation)

            # El entorno ejecuta la acción y devuelve el nuevo estado, recompensa, si terminó, y info
            next_observation, reward, done, info = env.step(action)

            # Aquí es donde un agente real de RL guardaría la experiencia
            # y, periódicamente, llamaría a agent.learn(...)
            # Por ahora, solo acumulamos recompensa.

            observation = next_observation
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # Pausa para visualización (solo si render=True)
            if env.physicsClient != p.DIRECT: # Si no estamos en modo directo
                time.sleep(1./240.) # Mantén la velocidad de simulación visual

            # DEBUG: Imprimir la recompensa del paso (opcional)
            # print(f"  Step: {episode_steps}, Reward: {reward:.2f}, Done: {done}")

        total_rewards.append(episode_reward)
        print(f"Episodio {episode+1}/{num_episodes} terminado. Recompensa: {episode_reward:.2f}, Pasos: {episode_steps}")

        # Puedes añadir lógica para guardar el modelo del agente cada cierto número de episodios
        # o cuando se alcance una recompensa alta.

    print(f"\nEntrenamiento finalizado. Total de pasos: {total_steps}")
    print(f"Recompensa promedio por episodio: {np.mean(total_rewards):.2f}")
    env.close() # Cierra el simulador al finalizar el entrenamiento

if __name__ == "__main__":
    train(num_episodes=100) # Comienza con pocos episodios para probar
    # Una vez que todo funcione, puedes aumentar num_episodes a 100, 1000, o más.