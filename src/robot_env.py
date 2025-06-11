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

        # Posición del efector final (3D) + Ángulos de las articulaciones (7) + Velocidades de las articulaciones (7) + Posición del objetivo (3D)
        self.observation_space_dims = 3 + len(self.joint_indices) + len(self.joint_indices) + 3 

        self.current_episode_steps = 0
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.robot_initial_joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.render_mode = render

        self.target_cup_pos = np.array(TARGET_CUP_POS)
        self.initial_cup_pos = None # Para almacenar la posición inicial del vaso

        # Definir un umbral de velocidad angular máxima aceptable
        self.max_angular_velocity_threshold = 0.5 
        
        # Umbral de velocidad angular mínima para penalizar el estancamiento
        self.min_angular_velocity_threshold = 0.05 

        # Umbral para detectar si el vaso ha sido movido/volteado
        self.cup_movement_threshold = 0.05 # Si el vaso se mueve más de 5cm, penalizar/terminar

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
        for _ in range(240): # Pequeña simulación para que el robot se asiente
            p.stepSimulation()

        cup_start_pos = TARGET_CUP_POS
        cup_start_orn = p.getQuaternionFromEuler([0, 0, 0])

        self.cup_id = p.loadURDF(CUP_URDF_PATH, cup_start_pos, cup_start_orn, useFixedBase=False)

        # Para asegurar que el vaso se asiente correctamente al inicio
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

        # Guardar la posición inicial real del vaso después de asentarse
        self.initial_cup_pos = np.array(p.getBasePositionAndOrientation(self.cup_id)[0])

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
        
        # Desempaquetar la observación actualizada
        effector_pos = observation[0:3] 
        joint_velocities = observation[10:17] 

        # Obtener la posición actual del vaso
        current_cup_pos = np.array(p.getBasePositionAndOrientation(self.cup_id)[0])
        distance_cup_moved = np.linalg.norm(current_cup_pos - self.initial_cup_pos)


        # --- CÁLCULO DE RECOMPENSAS Y PENALIZACIONES ---

        # Recompensa Densa por proximidad del efector al vaso
        # Ahora el objetivo es "llegar al vaso", no "a la posición inicial del vaso"
        # ¡IMPORTANTE! Si el vaso se mueve, la recompensa por distancia debería ser a la posición actual del vaso
        # o a la posición objetivo original si el vaso está fijo.
        # Dado que el vaso es fijo en TARGET_CUP_POS, mantenemos la distancia a esa posición.
        distance_to_target = np.linalg.norm(effector_pos - self.target_cup_pos)
        reward += 10.0 / (distance_to_target + 0.1) # Factor aumentado a 10.0 para mayor relevancia


        # Penalización si el vaso se mueve significativamente (se golpea o voltea)
        if distance_cup_moved > self.cup_movement_threshold:
            reward += 20000.0 # Penalización severa por mover el vaso
            done = True # El episodio termina si el vaso es golpeado/movido
            print(f"¡Vaso movido/golpeado! Distancia movida: {distance_cup_moved:.4f}. Episodio terminado.")


        # Condición de éxito: Efector final cerca del objetivo Y vaso no se ha movido
        # ¡Aumentamos la recompensa de éxito para que sea más relevante!
        # if distance_cup_moved < self.cup_movement_threshold:
        #     reward += 10000.0 # Recompensa de éxito mucho más alta
        #     done = True
        #     print(f"¡Objetivo alcanzado limpiamente! Distancia final: {distance_to_target:.4f}, Vaso movido: {distance_cup_moved:.4f}")


        # # Penalización por velocidad angular excesiva (movimientos muy bruscos)
        # angular_velocity_high_penalty = 0.0
        # for vel in joint_velocities:
        #     if abs(vel) > self.max_angular_velocity_threshold:
        #         angular_velocity_high_penalty += 2.0 * (abs(vel) - self.max_angular_velocity_threshold)**2 # Aumentado a 20.0
        # reward -= angular_velocity_high_penalty*0.5


        # # Penalización por velocidad angular muy baja (estancamiento)
        # angular_velocity_low_penalty = 0.0
        # # Solo penalizamos si no estamos muy cerca del objetivo
        # if not done and distance_to_target > 0.1: 
        #     for vel in joint_velocities:
        #         if abs(vel) < self.min_angular_velocity_threshold:
        #             angular_velocity_low_penalty += 1.0 # Aumentado a 10.0
        # reward -= angular_velocity_low_penalty*0.5
        
        # Penalización suave por cada paso de tiempo para fomentar la eficiencia
        # Reducimos la penalización por paso para que la recompensa por distancia y éxito sean más relevantes
        reward -= 0.001 # Reducida de 0.1 a 0.05


        # Fin de episodio por máximo de pasos
        if self.current_episode_steps >= self.max_episode_steps:
            done = True
            reward -= 10.0 # Penalización por fallar en el tiempo (aumentada un poco)
            print("Máximo de pasos alcanzado. No se alcanzó el objetivo.")


        info['distance_to_target'] = distance_to_target
        info['distance_cup_moved'] = distance_cup_moved
        info['angular_velocity_high_penalty'] = angular_velocity_high_penalty 
        info['angular_velocity_low_penalty'] = angular_velocity_low_penalty 
        
        return observation, reward, done, info
    
    def _get_observation(self):
        effector_link_index = p.getNumJoints(self.robot_id) - 1
        link_state = p.getLinkState(self.robot_id, effector_link_index)
        effector_pos = link_state[0]

        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_angles = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        observation = np.concatenate([
            np.array(effector_pos),      # 3 valores (X, Y, Z del efector)
            np.array(joint_angles),     # 7 valores (posición angular de cada articulación)
            np.array(joint_velocities), # 7 valores (velocidad angular de cada articulación)
            self.target_cup_pos          # 3 valores (posición X, Y, Z del vaso objetivo)
        ])
        return observation

    def close(self):
        p.disconnect(self.physicsClient)