import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import math # Para cálculos angulares si fueran necesarios

# Constantes del entorno
ROBOT_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa", "model.urdf")
CUP_URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "simple_cup.urdf")
PLANE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "plane.urdf")

# Límites del espacio de trabajo y acción
X_MIN = -0.5  # 50 cm a la izquierda del centro
X_MAX = 0.5   # 50 cm a la derecha del centro (total 1 metro)
TABLE_HEIGHT = 0.0 # Altura del plano/mesa donde se coloca el vaso
CUP_HEIGHT_OFFSET = 0.05 # La mitad de la altura del vaso (para que su base esté en el suelo)

MAX_EPISODE_STEPS = 500 # Máximo de pasos por episodio para el agente

class RobotEnv:
    def __init__(self, render=True):
        # --- CAMBIO CLAVE AQUÍ ---
        if render:
            self.physicsClient = p.connect(p.GUI) # Guarda la ID de la conexión
        else:
            self.physicsClient = p.connect(p.DIRECT) # Guarda la ID de la conexión
        # --- FIN DEL CAMBIO CLAVE ---

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.) # Paso de tiempo de simulación

        self.robot_id = None
        self.cup_id = None
        self.plane_id = None

        # --- Definición de espacio de acción y observación (crucial para RL) ---
        # Espacio de Acciones: Movimiento de cada articulación del brazo, activación del empuje.
        # Simplificación: Asumiremos control de posición para las primeras articulaciones del Kuka.
        # Kuka LBR iiwa tiene 7 articulaciones. Podemos controlar las 3 o 4 primeras para simplificar.
        self.joint_indices = [0, 1, 2, 3, 4, 5, 6] # Todas las 7 articulaciones del Kuka

        # Límites de las articulaciones (ajusta según los límites reales del Kuka si los conoces)
        # Esto es muy importante, un robot real tiene límites estrictos.
        # Estos son ejemplos, consulta la documentación del Kuka o el URDF para valores exactos.
        self.joint_limits_low = [-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054] # Radianes
        self.joint_limits_high = [2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054] # Radianes

        # Espacio de Acción: Un valor para cada articulación (ej. cambio en el ángulo)
        # y una acción para "empujar" (binaria o continua).
        # Simplificación: Control de posición directa para N articulaciones + acción de empuje.
        # Acciones: [delta_joint0, delta_joint1, ..., delta_jointN-1, push_action_value]
        # Por ahora, vamos a controlar las 7 articulaciones con un pequeño cambio en el ángulo
        # y una acción de "empuje" que se activa al final.
        # Definiremos 7 acciones continuas para las articulaciones (un valor entre -1 y 1 que se mapea a un delta de ángulo)
        # y una acción discreta para el "empuje" (0 o 1).
        # Para simplificar y usar PPO, podemos hacer que todas las acciones sean continuas.
        # La última acción será el "empuje".
        self.action_space_dims = len(self.joint_indices) + 1 # 7 articulaciones + 1 para empuje
        self.action_scale = 0.05 # Cuánto cambia el ángulo por cada acción continua (-1 a 1)
                                 # 0.05 radianes es ~2.8 grados.

        # Espacio de Observación (Estado):
        # - Posición del vaso (x, y, z)
        # - Ángulos de las articulaciones del brazo (para las 7 articulaciones)
        # - (Opcional, para simular cámara): Imagen (píxeles). Por ahora, simplificaremos esto.
        # Simplificamos la "cámara" a tener acceso a la posición del vaso directamente para este prototipo.
        # Un estado completo sería: [pos_vaso_x, pos_vaso_y, pos_vaso_z, joint0_angle, ..., joint6_angle]
        self.observation_space_dims = 3 + len(self.joint_indices) # Posición (x,y,z) del vaso + 7 ángulos

        self.current_episode_steps = 0
        self.max_episode_steps = MAX_EPISODE_STEPS

        # Posición inicial del robot (útil para resetear)
        self.robot_initial_joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # self.robot_initial_joint_positions = [0.0, -0.9, 0.0, 1.8, 0.0, -0.9, 0.0] # Una posición más "doblada" si prefieres

    def reset(self):
        """
        Reinicia el entorno a un estado inicial para un nuevo episodio.
        Carga el robot y el vaso, y coloca el vaso en una posición aleatoria.
        """
        p.resetSimulation() # Limpia la simulación anterior
        p.setGravity(0, 0, -9.8) # Vuelve a configurar gravedad

        self.plane_id = p.loadURDF(PLANE_URDF_PATH)

        # Cargar el brazo robótico Kuka
        robot_start_pos = [0, 0, 0]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(ROBOT_URDF_PATH, robot_start_pos, robot_start_orientation, useFixedBase=True)

        # Resetear articulaciones a la posición inicial
        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_index, self.robot_initial_joint_positions[i])
            # Aplicar control de posición para que el robot vaya a esa posición de forma estable
            p.setJointMotorControl2(self.robot_id,
                                    joint_index,
                                    p.POSITION_CONTROL,
                                    targetPosition=self.robot_initial_joint_positions[i],
                                    force=500 # Una fuerza razonable para moverlo
                                    )
        # Ejecutar unos pasos para que el robot alcance la posición inicial
        for _ in range(240): # Un segundo de simulación
            p.stepSimulation()


        # Colocar el vaso en una posición aleatoria en el plano de 1 metro (solo eje X)
        # El plano es de 1 metro, así que X_MIN y X_MAX son -0.5 y 0.5
        cup_x_pos = np.random.uniform(X_MIN, X_MAX)
        cup_y_pos = 0.0 # El vaso se mueve solo en el eje X, por lo tanto, Y es fijo.
        cup_z_pos = TABLE_HEIGHT + CUP_HEIGHT_OFFSET # Para que el vaso esté sobre el plano

        cup_start_pos = [cup_x_pos, cup_y_pos, cup_z_pos]
        cup_start_orn = p.getQuaternionFromEuler([0, 0, np.random.uniform(-np.pi, np.pi)]) # Orientación aleatoria en Z

        self.cup_id = p.loadURDF(CUP_URDF_PATH, cup_start_pos, cup_start_orn)

        self.current_episode_steps = 0

        # Obtener el estado inicial y retornarlo
        observation = self._get_observation()
        return observation

    def step(self, action):
        """
        Aplica una acción al entorno, avanza la simulación y calcula la recompensa.
        Args:
            action (np.array): Array de valores de acción.
                               Ej: [delta_joint0, ..., delta_joint6, push_action_value]
        Returns:
            tuple: (observation, reward, done, info)
        """
        self.current_episode_steps += 1

        # Mover las articulaciones del brazo
        # La acción es un vector de 8 elementos. Los primeros 7 son para las articulaciones.
        # El último es para la acción de empuje.
        joint_actions = action[:-1] # Las primeras 7 acciones para las articulaciones
        push_action_value = action[-1] # La última acción para el empuje

        # Obtener las posiciones actuales de las articulaciones
        current_joint_positions = [p.getJointState(self.robot_id, i)[0] for i in self.joint_indices]

        # Calcular las nuevas posiciones objetivo para las articulaciones
        target_joint_positions = []
        for i, delta in enumerate(joint_actions):
            new_pos = current_joint_positions[i] + delta * self.action_scale
            # Limitar la nueva posición a los límites de las articulaciones
            new_pos = np.clip(new_pos, self.joint_limits_low[i], self.joint_limits_high[i])
            target_joint_positions.append(new_pos)

        # Aplicar el control de posición a las articulaciones
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.POSITION_CONTROL,
            targetPositions=target_joint_positions,
            forces=[500] * len(self.joint_indices) # Aplicar una fuerza razonable
        )

        # Simular un paso de tiempo
        p.stepSimulation()

        # --- Calcular Recompensa ---
        reward = 0
        done = False # Indica si el episodio ha terminado

        # Obtener posición actual del efector final del robot (la "mano" o herramienta)
        # Necesitas saber el índice del último link/efector final.
        # Para Kuka LBR iiwa, el efector final suele ser el link 6 (link 7 si contamos desde 1).
        # Podemos obtenerlo mirando el URDF o imprimiendo la información de las articulaciones.
        # p.getJointInfo(robot_id, 6) te daría el link asociado a esa articulación.
        # Para simplificar, asumiremos que el último link es el efector final.
        num_links = p.getNumJoints(self.robot_id)
        # El link "end effector" es el último link del robot, que es el link asociado al último joint.
        # Si el Kuka tiene 7 articulaciones (0 a 6), el link 7 es el último link del brazo.
        # En PyBullet, los links tienen índices que corresponden a los índices de las articulaciones que los preceden.
        # Así que si joint 6 es el último, el link 6 es el efector final.
        effector_link_index = num_links - 1 # Última articulación/link

        link_state = p.getLinkState(self.robot_id, effector_link_index)
        effector_pos = link_state[0] # Posición cartesiana del efector final

        # Obtener posición actual del vaso
        cup_pos, cup_orn = p.getBasePositionAndOrientation(self.cup_id)
        cup_euler = p.getEulerFromQuaternion(cup_orn)

        # Recompensa por aproximación al vaso
        distance_to_cup = np.linalg.norm(np.array(effector_pos[:2]) - np.array(cup_pos[:2])) # Solo en XY, ignorar Z por ahora
        reward += max(0, 1.0 - distance_to_cup) * 0.1 # Recompensa proporcional a la cercanía

        # Detectar colisiones o contacto del brazo con el vaso
        # contact_points = p.getContactPoints(self.robot_id, self.cup_id)
        # if len(contact_points) > 0:
        #     reward += 0.5 # Pequeña recompensa por tocar el vaso

        # Lógica de Recompensa por Volteo / Desplazamiento
        # Para "voltear el vaso", podemos verificar la orientación del vaso (euler_z)
        # o si se ha desplazado significativamente de su posición inicial (más complejo).
        # Simplificación: El vaso se "voltea" si su orientación en Z cambia drásticamente (ej. más de 90 grados).
        # Esto es muy simplificado para el "mecanismo de empuje".
        # En un sistema real, un empuje causaría una fuerza. Aquí, podemos simularlo.

        # Si la acción de empuje es suficientemente alta (ej. > 0.5) y el efector está cerca
        # Asumiremos que el "empuje" voltea el vaso si está cerca.
        # En una implementación real, aplicarías una fuerza al vaso.
        if push_action_value > 0.5: # Si la acción de "empuje" se activó (acción continua > 0.5)
            if distance_to_cup < 0.1: # Si el efector final está cerca del vaso (10 cm)
                # Voltea el vaso de alguna manera simulada para el aprendizaje
                # Podemos darle una rotación aleatoria para simular el volteo
                p.resetBasePositionAndOrientation(self.cup_id, cup_pos,
                                                  p.getQuaternionFromEuler([np.pi/2, 0, np.random.uniform(-np.pi, np.pi)]))
                reward += 100 # Gran recompensa por voltear
                done = True # Episodio terminado por éxito
                print("¡Vaso volteado!")
            else:
                reward -= 5 # Penalización por intentar empujar sin estar cerca

        # Recompensa negativa si el vaso se cae (su posición Z es muy baja, ej. debajo del plano)
        if cup_pos[2] < (TABLE_HEIGHT - CUP_HEIGHT_OFFSET / 2): # Si el vaso cae por debajo del nivel esperado
            reward -= 50
            done = True
            print("¡Vaso caído!")

        # Penalización por cada paso de tiempo (fomentar soluciones rápidas)
        reward -= 0.1

        # Si el episodio alcanza el número máximo de pasos
        if self.current_episode_steps >= self.max_episode_steps:
            done = True
            reward -= 10 # Pequeña penalización por no completar a tiempo
            print("Máximo de pasos alcanzado.")

        # Obtener el nuevo estado
        observation = self._get_observation()

        info = {} # Información adicional (opcional)

        return observation, reward, done, info

    def _get_observation(self):
        """
        Retorna el estado actual del entorno.
        """
        # Posición del vaso
        cup_pos, _ = p.getBasePositionAndOrientation(self.cup_id)

        # Ángulos de las articulaciones del brazo
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_angles = [state[0] for state in joint_states] # state[0] es la posición angular

        # Información visual de la cámara (SIMPLIFICADA: NO IMPLEMENTADA AQUÍ)
        # Para una cámara real:
        # view_matrix = p.computeViewMatrixFromYawPitchRoll(...)
        # proj_matrix = p.computeProjectionMatrixFOV(...)
        # img = p.getCameraImage(width, height, view_matrix, proj_matrix)
        # visual_data = img[2] # Solo los píxeles RGB
        # En un agente real, se podría pasar la imagen (un array grande) o características extraídas de ella.
        # Por simplicidad, en este ejemplo el estado es solo numérico.

        # Combina toda la información en un solo vector de NumPy
        observation = np.concatenate([
            np.array(cup_pos),
            np.array(joint_angles)
        ])
        return observation

    def close(self):
        """Cierra la conexión con el simulador."""
        p.disconnect()

# Para verificar las articulaciones del Kuka y obtener sus límites
# Puedes ejecutar esto una vez para entender el robot
# if __name__ == '__main__':
#     env = RobotEnv(render=True)
#     env.reset()
#     print("\nInformación de las articulaciones del Kuka:")
#     for i in range(p.getNumJoints(env.robot_id)):
#         info = p.getJointInfo(env.robot_id, i)
#         print(f"Joint {info[0]} (Name: {info[1].decode('utf-8')}): Type={info[2]}, LowerLimit={info[8]}, UpperLimit={info[9]}")
#     env.close()