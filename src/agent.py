import numpy as np

class RandomAgent:
    """
    Un agente de ejemplo que elige acciones aleatorias.
    Esto es útil para probar el entorno.
    """
    def __init__(self, action_space_dims):
        self.action_space_dims = action_space_dims
        # Las acciones serán en el rango [-1, 1] para control continuo
        # La última acción (empuje) también será continua entre -1 y 1.
        # Luego, en el entorno, puedes decidir cuándo un valor > 0.5 activa el empuje.

    def choose_action(self, observation):
        """
        Elige una acción aleatoria del espacio de acción.
        Args:
            observation (np.array): El estado actual del entorno.
        Returns:
            np.array: Un vector de acciones aleatorias entre -1 y 1.
        """
        # Generar acciones aleatorias para cada dimensión del espacio de acción
        action = np.random.uniform(low=-1.0, high=1.0, size=self.action_space_dims)
        return action

    def learn(self, experiences):
        """
        Método de aprendizaje (por ahora, vacío para el agente aleatorio).
        Aquí es donde un algoritmo como PPO actualizaría su política.
        """
        pass