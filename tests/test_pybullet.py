import pybullet as p
import pybullet_data # Todavía lo necesitamos para el plano, etc.
import time
import os

print("Iniciando PyBullet...")

# Conectar al simulador (GUI para visualización)
physicsClient = p.connect(p.GUI)

# Añadir la ruta base de datos de PyBullet.
# Esto es bueno para que encuentre "plane.urdf" y otros archivos directamente en esa raíz.
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Cargar el plano (el suelo)
planeId = p.loadURDF("plane.urdf")
print(f"Plano cargado con ID: {planeId}")

# --- ¡MODIFICACIÓN CLAVE PARA CARGAR EL KUKA! ---
# Construimos la ruta absoluta al modelo del Kuka usando la información que encontraste.
# Es crucial usar 'os.path.join' para que funcione correctamente en Windows.
# Nota: La carpeta es 'kuka_iiwa', no 'kuka_lbr_iiwa' como en algunos ejemplos.
kuka_urdf_file = os.path.join(
    pybullet_data.getDataPath(), # Esto suele ser 'env\Lib\site-packages\pybullet_data'
    "kuka_iiwa",                 # ¡Esta es la subcarpeta que encontraste!
    "model.urdf"                 # El nombre del archivo URDF
)

# Opcional: Verificación para depuración
print(f"Intentando cargar Kuka desde: {kuka_urdf_file}")
if not os.path.exists(kuka_urdf_file):
    print(f"ERROR: El archivo URDF del Kuka NO SE ENCONTRÓ en la ruta construida: {kuka_urdf_file}")
    print("Por favor, verifica la ruta de 'pybullet_data.getDataPath()' y la existencia de 'kuka_iiwa/model.urdf' dentro.")
    p.disconnect()
    exit()
# --- FIN DE LA MODIFICACIÓN CLAVE ---

# Cargar el brazo robótico Kuka usando la ruta absoluta
robotStartPos = [0, 0, 0]
robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
kukaId = p.loadURDF(kuka_urdf_file, robotStartPos, robotStartOrientation, useFixedBase=True)
print(f"Robot Kuka cargado con ID: {kukaId}")


# Configurar la gravedad (eje Z hacia abajo)
p.setGravity(0, 0, -9.8)

# Establecer el paso de tiempo de la simulación
p.setTimeStep(1./240.) # 240 Hz es un valor común

# Bucle de simulación
print("Iniciando bucle de simulación... La ventana del simulador debería aparecer.")
print("Presiona Ctrl+C en la terminal para detener el script.")

try:
    for i in range(10000):
        p.stepSimulation()
        time.sleep(1./240.)

except KeyboardInterrupt:
    print("\nSimulación detenida por el usuario.")
finally:
    p.disconnect()
    print("PyBullet desconectado.")