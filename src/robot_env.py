import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import math

# Constantes del entorno
ROBOT_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa", "model.urdf")
CUP_URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "simple_cup.urdf")
PLANE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "plane.urdf")

# Límites del espacio de trabajo y acción
X_MIN = -1.0
X_MAX = 1.0
TABLE_HEIGHT = 0.0
CUP_HEIGHT_OFFSET = 0.05

MAX_EPISODE_STEPS = 1000

# --- NUEVAS CONSTANTES PARA LA CÁMARA ---
CAMERA_WIDTH = 64
CAMERA_HEIGHT = 64
CAMERA_FOV = 60 # Campo de visión en grados
CAMERA_NEAR = 0.1 # Distancia mínima de renderizado
CAMERA_FAR = 10.0 # Distancia máxima de renderizado

# Posición de la cámara (ej. justo encima y mirando hacia abajo)
# Ajusta estas coordenadas para que la cámara vea el espacio de trabajo y el vaso.
CAMERA_POSITION = [0.0, 0.0, 2.5] # X, Y, Z de la cámara
CAMERA_TARGET = [0.0, 0.0, 0.0]  # Punto al que mira la cámara (centro del espacio de trabajo)
CAMERA_UP_VECTOR = [0, 1, 0] # Vector que indica la dirección "arriba" de la cámara
# --- FIN NUEVAS CONSTANTES PARA LA CÁMARA ---


class RobotEnv:
    def __init__(self, render=True):
        if render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.)

        self.robot_id = None
        self.cup_id = None
        self.plane_id = None

        self.joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.joint_limits_low = [-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054]
        self.joint_limits_high = [2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054]

        self.action_space_dims = len(self.joint_indices) + 1 # 7 articulaciones + 1 para empuje
        self.action_scale = 0.05

        # --- MODIFICACIÓN: ESPACIO DE OBSERVACIÓN ---
        # Posición del vaso (3) + Ángulos de las articulaciones (7) + Datos de la cámara (width * height * canales)
        # Asumiremos imágenes en escala de grises para simplificar (1 canal). Si es RGB, sería * 3.
        # Es común pasar solo el canal de profundidad o la imagen en escala de grises a la red neuronal inicial.
        self.observation_space_dims = 3 + len(self.joint_indices) + (CAMERA_WIDTH * CAMERA_HEIGHT * 1) # Asumiendo 1 canal (escala de grises/profundidad)
        # --- FIN MODIFICACIÓN ---

        self.current_episode_steps = 0
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.robot_initial_joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # --- NUEVOS ATRIBUTOS PARA LA CÁMARA ---
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=CAMERA_POSITION,
            cameraTargetPosition=CAMERA_TARGET,
            cameraUpVector=CAMERA_UP_VECTOR
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=CAMERA_FOV,
            aspect=float(CAMERA_WIDTH) / CAMERA_HEIGHT,
            nearVal=CAMERA_NEAR,
            farVal=CAMERA_FAR
        )
        # --- FIN NUEVOS ATRIBUTOS PARA LA CÁMARA ---

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF(PLANE_URDF_PATH)

        robot_start_pos = [0, 0, 0]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(ROBOT_URDF_PATH, robot_start_pos, robot_start_orientation, useFixedBase=True)

        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_index, self.robot_initial_joint_positions[i])
            p.setJointMotorControl2(self.robot_id,
                                    joint_index,
                                    p.POSITION_CONTROL,
                                    targetPosition=self.robot_initial_joint_positions[i],
                                    force=500
                                    )
        for _ in range(240):
            p.stepSimulation()

        cup_x_pos = np.random.uniform(X_MIN, X_MAX)
        cup_y_pos = 0.0
        cup_z_pos = TABLE_HEIGHT + CUP_HEIGHT_OFFSET

        cup_start_pos = [cup_x_pos, cup_y_pos, cup_z_pos]
        cup_start_orn = p.getQuaternionFromEuler([0, 0, np.random.uniform(-np.pi, np.pi)])

        self.cup_id = p.loadURDF(CUP_URDF_PATH, cup_start_pos, cup_start_orn)

        self.current_episode_steps = 0

        observation = self._get_observation()
        return observation

    def step(self, action):
        self.current_episode_steps += 1

        # --- FIX: Initialize reward at the very beginning of the method ---
        reward = 0 # Initialize reward to 0 here
        # --- END FIX ---

        joint_actions = action[:-1]
        push_action_value = action[-1]

        current_joint_positions = [p.getJointState(self.robot_id, i)[0] for i in self.joint_indices]

        target_joint_positions = []
        for i, delta in enumerate(joint_actions):
            new_pos = current_joint_positions[i] + delta * self.action_scale
            new_pos = np.clip(new_pos, self.joint_limits_low[i], self.joint_limits_high[i])
            target_joint_positions.append(new_pos)

        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.POSITION_CONTROL,
            targetPositions=target_joint_positions,
            forces=[500] * len(self.joint_indices)
        )

        # --- MODIFICACIÓN: Lógica de Empuje (más física) ---
        effector_link_index = p.getNumJoints(self.robot_id) - 1

        link_state = p.getLinkState(self.robot_id, effector_link_index)
        effector_pos = link_state[0]

        cup_pos, cup_orn = p.getBasePositionAndOrientation(self.cup_id)
        distance_to_cup = np.linalg.norm(np.array(effector_pos[:2]) - np.array(cup_pos[:2]))

        if push_action_value > 0.5 and distance_to_cup < 0.15:
            push_force_magnitude = 100

            force_direction = np.array(cup_pos) - np.array(effector_pos)
            force_direction[2] = 0
            force_direction_normalized = force_direction / (np.linalg.norm(force_direction) + 1e-6)

            force_application_point = [cup_pos[0], cup_pos[1], cup_pos[2] - CUP_HEIGHT_OFFSET + 0.01]

            p.applyExternalForce(self.cup_id, -1,
                                 list(push_force_magnitude * force_direction_normalized),
                                 force_application_point,
                                 p.WORLD_FRAME)
            reward += 0.5 # Now 'reward' exists and can be modified
            print("Aplicando empuje al vaso.")

        p.stepSimulation()

        # --- Calcular Recompensa ---
        # The 'reward = 0' line was moved to the top.
        done = False

        cup_pos, cup_orn = p.getBasePositionAndOrientation(self.cup_id)
        cup_euler = p.getEulerFromQuaternion(cup_orn)

        link_state = p.getLinkState(self.robot_id, effector_link_index)
        effector_pos = link_state[0]
        distance_to_cup = np.linalg.norm(np.array(effector_pos[:2]) - np.array(cup_pos[:2]))
        reward += max(0, 1.0 - distance_to_cup) * 0.1

        original_cup_z_axis_world = [0,0,1]
        cup_rot_matrix = p.getMatrixFromQuaternion(cup_orn)
        cup_rot_matrix = np.array(cup_rot_matrix).reshape(3,3)
        current_cup_z_axis_world = cup_rot_matrix @ np.array([0,0,1])

        angle_diff = np.arccos(np.dot(original_cup_z_axis_world, current_cup_z_axis_world))

        if angle_diff > (np.pi / 3):
            reward += 100
            done = True
            print("¡Vaso volteado/inclinado significativamente!")
        else:
            reward += max(0, 1.0 - distance_to_cup) * 0.1

        if cup_pos[2] < (TABLE_HEIGHT + CUP_HEIGHT_OFFSET / 4):
            if angle_diff <= (np.pi / 3) and cup_pos[2] < (TABLE_HEIGHT + CUP_HEIGHT_OFFSET / 4):
                 reward -= 50
                 done = True
                 print("¡Vaso caído!")

        reward -= 0.01

        if self.current_episode_steps >= self.max_episode_steps:
            done = True
            reward -= 10
            print("Máximo de pasos alcanzado.")

        observation = self._get_observation()
        info = {}

        return observation, reward, done, info
    
    def _get_observation(self):
        # Posición del vaso
        cup_pos, _ = p.getBasePositionAndOrientation(self.cup_id)

        # Ángulos de las articulaciones del brazo
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_angles = [state[0] for state in joint_states]

        # --- MODIFICACIÓN: CAPTURA DE IMAGEN DE LA CÁMARA ---
        img_arr = p.getCameraImage(
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL # Usa este renderizador si está disponible para mejor rendimiento
                                                 # O p.ER_TINY_RENDERER para CPU si no hay GPU
        )
        # img_arr[0] es la anchura, img_arr[1] la altura, img_arr[2] son los valores RGB, img_arr[3] es la profundidad.
        # Queremos los píxeles de la imagen. La documentación de PyBullet dice que img_arr[2] son los píxeles RGB.
        # Convertimos a escala de grises para simplificar el estado, y normalizamos a [0, 1]
        rgb_pixels = np.array(img_arr[2], dtype=np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4)) # RGBA
        # Convertir a escala de grises (promedio de RGB) y aplanarla
        gray_image = np.mean(rgb_pixels[:, :, :3], axis=2) # Ignorar canal alfa, promedio de RGB
        normalized_gray_image = gray_image / 255.0 # Normalizar a [0, 1]
        flat_image = normalized_gray_image.flatten() # Aplanar para incluir en el vector de estado

        # --- FIN MODIFICACIÓN ---

        # Combina toda la información en un solo vector de NumPy
        observation = np.concatenate([
            np.array(cup_pos),
            np.array(joint_angles),
            flat_image # Añadir los datos de la imagen aplanados
        ])
        return observation

    def close(self):
        p.disconnect(self.physicsClient)