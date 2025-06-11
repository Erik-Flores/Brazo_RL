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
X_MIN = 0.5
X_MAX = 1.0 
TABLE_HEIGHT = 0.0

# Posición OBJETIVO FIJA para el vaso
TARGET_CUP_POS = [0.5, 0.5, 0.0] # Fija en [0.5, 0.5, 0.0]

MAX_EPISODE_STEPS = 1000

# --- CONSTANTES DE CÁMARA (solo para visualización GUI) ---
CAMERA_WIDTH = 64
CAMERA_HEIGHT = 64
CAMERA_FOV = 60
CAMERA_NEAR = 0.1
CAMERA_FAR = 10.0
CAMERA_POSITION = [0.5, 0.0, 2.5]
CAMERA_TARGET = [0.5, 0.5, 0.0] # Apunta al vaso fijo
CAMERA_UP_VECTOR = [0, 1, 0]
# --- FIN CONSTANTES DE CÁMARA ---


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

        self.action_space_dims = len(self.joint_indices)
        self.action_scale = 0.1 # Aumentado para movimientos más rápidos

        # Posición del efector final (3D) + Ángulos de las articulaciones (7) + Velocidades de las articulaciones (7)
        self.observation_space_dims = 3 + len(self.joint_indices) + len(self.joint_indices)

        self.current_episode_steps = 0
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.robot_initial_joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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
        self.render_mode = render

        self.target_cup_pos = np.array(TARGET_CUP_POS)

        # Definir un umbral de velocidad angular máxima aceptable
        # Ajusta este valor según cuánto quieras penalizar las velocidades altas.
        # Un valor más bajo penalizará más, uno más alto será más permisivo.
        self.max_angular_velocity_threshold = 0.5 # Radianes por segundo (ejemplo)

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

        cup_start_pos = TARGET_CUP_POS
        cup_start_orn = p.getQuaternionFromEuler([0, 0, 0])

        self.cup_id = p.loadURDF(CUP_URDF_PATH, cup_start_pos, cup_start_orn, useFixedBase=False)

        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.cup_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=cup_start_pos
        )
        for _ in range(50):
            p.stepSimulation()
        p.removeConstraint(constraint_id)

        self.current_episode_steps = 0
        observation = self._get_observation()
        return observation

    def step(self, action):
        self.current_episode_steps += 1

        reward = 0.0
        done = False
        info = {}

        joint_actions = action 

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

        p.stepSimulation()

        # Obtener el estado actual para la observación y las velocidades
        observation = self._get_observation()
        effector_pos = observation[0:3] # Los primeros 3 elementos son la posición del efector final
        joint_velocities = observation[10:17] # Del elemento 10 al 16 son las velocidades angulares de las 7 articulaciones

        distance_to_target = np.linalg.norm(effector_pos - self.target_cup_pos)

        # Recompensa Densa por proximidad
        reward = 1.0 / (distance_to_target + 0.01)

        # Recompensa grande si alcanza el objetivo con un umbral muy pequeño
        if distance_to_target < 0.05:
            reward += 1000.0
            done = True
            print(f"¡Objetivo alcanzado! Distancia final: {distance_to_target:.4f}")

        # Penalización por velocidad angular excesiva
        # Calcula la magnitud de la velocidad angular para cada articulación.
        # Si alguna articulación excede el umbral, aplica una penalización.
        angular_velocity_penalty = 0.0
        for vel in joint_velocities:
            if abs(vel) > self.max_angular_velocity_threshold:
                # Penalización cuadrática para castigar más las desviaciones grandes
                # Puedes ajustar el factor 10.0 según lo severo que quieras que sea.
                angular_velocity_penalty += 10.0 * (abs(vel) - self.max_angular_velocity_threshold)**2 
        
        # Resta la penalización a la recompensa total
        reward -= angular_velocity_penalty
        
        # Penalización suave por cada paso de tiempo para fomentar la eficiencia
        reward -= 0.01 

        # Fin de episodio por máximo de pasos
        if self.current_episode_steps >= self.max_episode_steps:
            done = True
            reward -= 10.0 # Penalización por fallar en el tiempo
            print("Máximo de pasos alcanzado. No se alcanzó el objetivo.")

        info['distance_to_target'] = distance_to_target
        info['angular_velocity_penalty'] = angular_velocity_penalty # Para depuración
        
        return observation, reward, done, info
    
    def _get_observation(self):
        effector_link_index = p.getNumJoints(self.robot_id) - 1
        link_state = p.getLinkState(self.robot_id, effector_link_index)
        effector_pos = link_state[0]

        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_angles = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        observation = np.concatenate([
            np.array(effector_pos),
            np.array(joint_angles),
            np.array(joint_velocities)
        ])
        return observation

    def close(self):
        p.disconnect(self.physicsClient)