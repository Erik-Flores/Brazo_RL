import numpy as np
import torch
import os

from src.robot_env import EntornoRobot
from src.agent import AgentePPO

DIRECTORIO_GUARDADO = "modelos_entrenados"
if not os.path.exists(DIRECTORIO_GUARDADO):
    os.makedirs(DIRECTORIO_GUARDADO)

class EntrenadorRL:
    def __init__(self, entorno, agente, pasos_por_lote=2048):
        self.entorno = entorno
        self.agente = agente
        self.pasos_por_lote = pasos_por_lote
        self.recompensas_episodio = []
        self.perdidas_actor = []
        self.perdidas_critico = []

    def entrenar(self, num_iteraciones):
        for iteracion in range(num_iteraciones):
            lote_obs, lote_acciones, lote_log_probs, lote_recompensas, lote_terminados = self._recopilar_rollout()
            
            valores = [self.agente.actor_critico(torch.FloatTensor(obs).unsqueeze(0))[1].item() for obs in lote_obs]
            lote_retornos, lote_ventajas = self.agente.calcular_gae(lote_recompensas, lote_terminados, valores)
            
            tensor_lote_obs = torch.FloatTensor(self.agente.normalizador_estado.normalizar(lote_obs))
            tensor_lote_acciones = torch.FloatTensor(lote_acciones)
            tensor_lote_log_probs = torch.FloatTensor(lote_log_probs)
            tensor_lote_retornos = torch.FloatTensor(lote_retornos)
            tensor_lote_ventajas = torch.FloatTensor(lote_ventajas)
            
            tensor_lote_ventajas = (tensor_lote_ventajas - tensor_lote_ventajas.mean()) / (tensor_lote_ventajas.std() + 1e-8)

            perdida_actor, perdida_critico = self.agente.actualizar(
                tensor_lote_obs, tensor_lote_acciones, tensor_lote_log_probs, 
                tensor_lote_retornos, tensor_lote_ventajas
            )
            
            self.perdidas_actor.append(perdida_actor)
            self.perdidas_critico.append(perdida_critico)

            print(f"Iteración {iteracion+1}/{num_iteraciones} (Paso {(iteracion+1) * self.pasos_por_lote})")
            print(f"Pérdida Actor: {perdida_actor:.4f}, Pérdida Crítico: {perdida_critico:.4f}")
            print(f"Recompensa promedio: {np.mean(self.recompensas_episodio):.2f}")
            
            if (iteracion + 1) % 50 == 0:
                self.guardar_agente(f"agente_iteracion_{iteracion+1}.pth")

    def _recopilar_rollout(self):
        lote_obs = []
        lote_acciones = []
        lote_log_probs = []
        lote_recompensas = []
        lote_terminados = []
        
        pasos_actuales = 0
        while pasos_actuales < self.pasos_por_lote:
            obs = self.entorno.reset()
            recompensa_episodio = 0
            pasos_episodio = 0
            terminado = False
            
            while not terminado:
                accion, log_prob, valor = self.agente.obtener_accion_y_valor(obs)
                siguiente_obs, recompensa, terminado, info = self.entorno.step(accion)
                
                lote_obs.append(obs)
                lote_acciones.append(accion)
                lote_log_probs.append(log_prob)
                lote_recompensas.append(recompensa)
                lote_terminados.append(terminado)

                obs = siguiente_obs
                recompensa_episodio += recompensa
                pasos_episodio += 1
                pasos_actuales += 1

                if pasos_actuales >= self.pasos_por_lote:
                    terminado = True

            self.recompensas_episodio.append(recompensa_episodio)
            print(f"  Episodio finalizado. Recompensa: {recompensa_episodio:.2f}, Pasos: {pasos_episodio}")

        return np.array(lote_obs), np.array(lote_acciones), np.array(lote_log_probs), np.array(lote_recompensas), np.array(lote_terminados)

    def guardar_agente(self, nombre_archivo):
        ruta_archivo = os.path.join(DIRECTORIO_GUARDADO, nombre_archivo)
        torch.save({
            'estado_dict_actor_critico': self.agente.actor_critico.state_dict(),
            'estado_dict_optimizador': self.agente.optimizador.state_dict(),
            'media_normalizador_estado': self.agente.normalizador_estado.media,
            'std_normalizador_estado': self.agente.normalizador_estado.std,
        }, ruta_archivo)
        print(f"Agente guardado en: {ruta_archivo}")

if __name__ == "__main__":
    entorno = EntornoRobot(renderizar=True)

    dims_espacio_observacion = entorno.dims_espacio_observacion
    dims_espacio_accion = entorno.dims_espacio_accion

    agente_ppo = AgentePPO(
        dims_espacio_observacion=dims_espacio_observacion,
        dims_espacio_accion=dims_espacio_accion,
        lr_actor=1e-4, 
        gamma=0.99,
        lambda_gae=0.95,
        parametro_clip=0.3, 
        epocas_ppo=10,
        tamano_mini_lote=64,
        coef_entropia=0.05 
    )

    entrenador = EntrenadorRL(entorno, agente_ppo, pasos_por_lote=2048)
    entrenador.entrenar(num_iteraciones=100)

    entrenador.guardar_agente("agente_final.pth")
    entorno.close()