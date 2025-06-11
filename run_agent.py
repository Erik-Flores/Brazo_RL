import torch
import numpy as np
import time
import os
import glob
from src.robot_env import EntornoRobot
from src.agent import AgentePPO

class ProbadorAgente:
    def __init__(self, ruta_modelo, renderizar=True):
        self.entorno = EntornoRobot(renderizar=renderizar)
        
        dims_espacio_observacion = self.entorno.dims_espacio_observacion
        dims_espacio_accion = self.entorno.dims_espacio_accion
        
        self.agente = AgentePPO(
            dims_espacio_observacion=dims_espacio_observacion,
            dims_espacio_accion=dims_espacio_accion
        )
        
        self.cargar_agente(ruta_modelo)
        
    def cargar_agente(self, ruta_modelo):
        checkpoint = torch.load(ruta_modelo, map_location='cpu', weights_only=False)
        
        self.agente.actor_critico.load_state_dict(checkpoint['estado_dict_actor_critico'])
        self.agente.normalizador_estado.media = checkpoint['media_normalizador_estado']
        self.agente.normalizador_estado.std = checkpoint['std_normalizador_estado']
        
        print(f"Agente cargado desde: {ruta_modelo}")
    
    def obtener_accion_determinista(self, obs):
        obs_normalizada = self.agente.normalizador_estado.normalizar(obs)
        tensor_obs = torch.FloatTensor(obs_normalizada).unsqueeze(0)
        
        with torch.no_grad():
            dist_accion, valor = self.agente.actor_critico(tensor_obs)
            accion = dist_accion.mean
            
        return accion.squeeze(0).numpy(), valor.item()
    
    def encontrar_mejor_modelo(self, directorio="modelos_entrenados"):
        archivos_modelo = glob.glob(os.path.join(directorio, "agente_iteracion_*.pth"))
        
        if not archivos_modelo:
            print("No se encontraron modelos de iteración")
            return None
        
        mejor_modelo = None
        mejor_recompensa = float('-inf')
        
        print("Evaluando modelos disponibles...")
        
        for archivo in archivos_modelo:
            print(f"Probando: {os.path.basename(archivo)}")
            self.cargar_agente(archivo)
            
            recompensas = []
            for _ in range(3):
                recompensa, _, _ = self.probar_episodio(mostrar_info=False, pausa=0.001)
                recompensas.append(recompensa)
            
            recompensa_promedio = np.mean(recompensas)
            print(f"  Recompensa promedio: {recompensa_promedio:.2f}")
            
            if recompensa_promedio > mejor_recompensa:
                mejor_recompensa = recompensa_promedio
                mejor_modelo = archivo
        
        print(f"\nMejor modelo encontrado: {os.path.basename(mejor_modelo)}")
        print(f"Recompensa promedio: {mejor_recompensa:.2f}")
        
        return mejor_modelo
    def probar_episodio(self, mostrar_info=True, pausa=0.01):
        obs = self.entorno.reset()
        recompensa_total = 0
        pasos = 0
        terminado = False
        
        print("Iniciando episodio de prueba...")
        
        while not terminado:
            accion, valor = self.obtener_accion_determinista(obs)
            obs, recompensa, terminado, info = self.entorno.step(accion)
            
            recompensa_total += recompensa
            pasos += 1
            
            if mostrar_info and pasos % 50 == 0:
                distancia = info.get('distancia_efector_a_vaso', 0)
                print(f"Paso {pasos}: Distancia al vaso: {distancia:.4f}, Valor: {valor:.2f}")
            
            time.sleep(pausa)
            
            if pasos >= 2000:
                print("Límite de pasos alcanzado")
                break
        
        print(f"\nEpisodio terminado:")
        print(f"Recompensa total: {recompensa_total:.2f}")
        print(f"Pasos totales: {pasos}")
        print(f"Información final: {info}")
        
        return recompensa_total, pasos, info
    
    #def probar_episodio(self, mostrar_info=True, pausa=0.01):
    def probar_multiples_episodios(self, num_episodios=5):
        resultados = []
        
        for i in range(num_episodios):
            print(f"\n=== EPISODIO {i+1}/{num_episodios} ===")
            recompensa, pasos, info = self.probar_episodio(mostrar_info=False, pausa=0.005)
            resultados.append((recompensa, pasos, info))
        
        recompensas = [r[0] for r in resultados]
        pasos_totales = [r[1] for r in resultados]
        
        print(f"\n=== RESUMEN DE {num_episodios} EPISODIOS ===")
        print(f"Recompensa promedio: {np.mean(recompensas):.2f} ± {np.std(recompensas):.2f}")
        print(f"Pasos promedio: {np.mean(pasos_totales):.1f} ± {np.std(pasos_totales):.1f}")
        print(f"Mejor recompensa: {np.max(recompensas):.2f}")
        print(f"Peor recompensa: {np.min(recompensas):.2f}")
        
        return resultados
    
    def probar_multiples_episodios(self, num_episodios=5):
        self.entorno.close()

    def cerrar(self):
        self.entorno.close()

if __name__ == "__main__":
    ruta_modelo = "modelos_entrenados/agente_final.pth"
    
    probador = ProbadorAgente(ruta_modelo, renderizar=True)
    
    print("Opciones de prueba:")
    print("1. Probar un episodio")
    print("2. Probar múltiples episodios")
    
    opcion = input("Selecciona opción (1 o 2): ")
    
    if opcion == "1":
        probador.probar_episodio()
    elif opcion == "2":
        num_episodios = int(input("¿Cuántos episodios? (por defecto 5): ") or "5")
        probador.probar_multiples_episodios(num_episodios)
    else:
        print("Opción no válida")
    
    probador.cerrar()