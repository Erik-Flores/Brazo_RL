import pybullet as p
import pybullet_data
import numpy as np
import os

RUTA_URDF_ROBOT = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa", "model.urdf")
RUTA_URDF_VASO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "simple_cup.urdf")
RUTA_URDF_PLANO = os.path.join(pybullet_data.getDataPath(), "plane.urdf")

X_MIN = 0.5
X_MAX = 1.0 
ALTURA_MESA = 0.0

POSICION_OBJETIVO_VASO = [0.5, 0.5, 0.0]
PASOS_MAXIMOS_EPISODIO = 1000

class EntornoRobot:
    def __init__(self, renderizar=True):
        if renderizar:
            self.cliente_fisica = p.connect(p.GUI)
        else:
            self.cliente_fisica = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.)

        self.id_robot = None
        self.id_vaso = None
        self.id_plano = None

        self.indices_articulaciones = [0, 1, 2, 3, 4, 5, 6]
        self.limites_articulaciones_bajo = [-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054]
        self.limites_articulaciones_alto = [2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054]

        self.dims_espacio_accion = len(self.indices_articulaciones)
        self.escala_accion = 0.1

        self.dims_espacio_observacion = 3 + len(self.indices_articulaciones) + len(self.indices_articulaciones) + 3 

        self.pasos_episodio_actual = 0
        self.pasos_maximos_episodio = PASOS_MAXIMOS_EPISODIO
        self.posiciones_iniciales_articulaciones_robot = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.posicion_objetivo_vaso = np.array(POSICION_OBJETIVO_VASO)
        self.posicion_inicial_vaso = None
        self.vaso_golpeado = False

        self.umbral_golpe_vaso = 0.05

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.id_plano = p.loadURDF(RUTA_URDF_PLANO)

        posicion_inicio_robot = [0, 0, 0]
        orientacion_inicio_robot = p.getQuaternionFromEuler([0, 0, 0])
        self.id_robot = p.loadURDF(RUTA_URDF_ROBOT, posicion_inicio_robot, orientacion_inicio_robot, useFixedBase=True)

        for i, indice_articulacion in enumerate(self.indices_articulaciones):
            p.resetJointState(self.id_robot, indice_articulacion, self.posiciones_iniciales_articulaciones_robot[i])
            p.setJointMotorControl2(self.id_robot,
                                    indice_articulacion,
                                    p.POSITION_CONTROL,
                                    targetPosition=self.posiciones_iniciales_articulaciones_robot[i],
                                    force=500)
        
        for _ in range(240):
            p.stepSimulation()

        posicion_inicio_vaso = POSICION_OBJETIVO_VASO
        orientacion_inicio_vaso = p.getQuaternionFromEuler([0, 0, 0])

        self.id_vaso = p.loadURDF(RUTA_URDF_VASO, posicion_inicio_vaso, orientacion_inicio_vaso, useFixedBase=False)

        id_restriccion = p.createConstraint(
            parentBodyUniqueId=self.id_vaso,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=posicion_inicio_vaso
        )
        
        for _ in range(50):
            p.stepSimulation()
        p.removeConstraint(id_restriccion)

        self.posicion_inicial_vaso = np.array(p.getBasePositionAndOrientation(self.id_vaso)[0])
        self.pasos_episodio_actual = 0
        self.vaso_golpeado = False
        self.distancia_anterior = None
        
        observacion = self._obtener_observacion()
        return observacion

    def step(self, accion):
        self.pasos_episodio_actual += 1

        recompensa = 0.0
        terminado = False
        info = {}

        posiciones_articulaciones_actuales = [p.getJointState(self.id_robot, i)[0] for i in self.indices_articulaciones]

        posiciones_objetivo_articulaciones = []
        for i, delta in enumerate(accion):
            nueva_pos = posiciones_articulaciones_actuales[i] + delta * self.escala_accion
            nueva_pos = np.clip(nueva_pos, self.limites_articulaciones_bajo[i], self.limites_articulaciones_alto[i])
            posiciones_objetivo_articulaciones.append(nueva_pos)

        p.setJointMotorControlArray(
            self.id_robot,
            self.indices_articulaciones,
            p.POSITION_CONTROL,
            targetPositions=posiciones_objetivo_articulaciones,
            forces=[500] * len(self.indices_articulaciones)
        )

        p.stepSimulation()

        observacion = self._obtener_observacion()
        posicion_efector = observacion[0:3]
        velocidades_articulaciones = observacion[10:17]
        posicion_vaso_actual = np.array(p.getBasePositionAndOrientation(self.id_vaso)[0])
        distancia_vaso_movido = np.linalg.norm(posicion_vaso_actual - self.posicion_inicial_vaso)

        distancia_efector_a_vaso = np.linalg.norm(posicion_efector - posicion_vaso_actual)
        
        if self.distancia_anterior is None:
            self.distancia_anterior = distancia_efector_a_vaso
        
        mejora_distancia = self.distancia_anterior - distancia_efector_a_vaso
        self.distancia_anterior = distancia_efector_a_vaso
        
        recompensa += mejora_distancia * 10.0
        recompensa += 1.0 / (distancia_efector_a_vaso + 0.1)

        velocidad_total = sum(abs(v) for v in velocidades_articulaciones)
        if velocidad_total < 0.1:
            recompensa -= 0.5

        if distancia_vaso_movido > self.umbral_golpe_vaso and not self.vaso_golpeado:
            recompensa += 100.0
            self.vaso_golpeado = True
            terminado = True
            print(f"¡Vaso golpeado! Distancia movida: {distancia_vaso_movido:.4f}")

        orientacion_vaso = p.getBasePositionAndOrientation(self.id_vaso)[1]
        euler_vaso = p.getEulerFromQuaternion(orientacion_vaso)
        inclinacion_vaso = abs(euler_vaso[0]) + abs(euler_vaso[1])
        
        if inclinacion_vaso > 0.5:
            recompensa += 200.0
            terminado = True
            print(f"¡Vaso tirado! Inclinación: {inclinacion_vaso:.4f}")

        if distancia_vaso_movido > 0.3:
            recompensa += 150.0
            terminado = True
            print(f"¡Vaso alejado! Distancia: {distancia_vaso_movido:.4f}")

        recompensa -= 0.1

        if self.pasos_episodio_actual >= self.pasos_maximos_episodio:
            terminado = True
            recompensa -= 10.0

        info['distancia_efector_a_vaso'] = distancia_efector_a_vaso
        info['distancia_vaso_movido'] = distancia_vaso_movido
        info['inclinacion_vaso'] = inclinacion_vaso
        
        return observacion, recompensa, terminado, info
    
    def _obtener_observacion(self):
        indice_enlace_efector = p.getNumJoints(self.id_robot) - 1
        estado_enlace = p.getLinkState(self.id_robot, indice_enlace_efector)
        posicion_efector = estado_enlace[0]

        estados_articulaciones = p.getJointStates(self.id_robot, self.indices_articulaciones)
        angulos_articulaciones = [estado[0] for estado in estados_articulaciones]
        velocidades_articulaciones = [estado[1] for estado in estados_articulaciones]

        observacion = np.concatenate([
            np.array(posicion_efector),
            np.array(angulos_articulaciones),
            np.array(velocidades_articulaciones),
            np.array(p.getBasePositionAndOrientation(self.id_vaso)[0])
        ])
        return observacion

    def close(self):
        p.disconnect(self.cliente_fisica)