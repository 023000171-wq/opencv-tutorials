import cv2
import numpy as np
import os

# ================================
# Obtener la ruta del proyecto
# ================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
out_dir = os.path.join(project_root, "out")

# Crear la carpeta out si no existe
os.makedirs(out_dir, exist_ok=True)

print("===================================")
print("Archivo:", __file__)
print("Carpeta del script:", script_dir)
print("Carpeta del proyecto:", project_root)
print("Carpeta OUT:", out_dir)
print("===================================")

# ================================
# Abrir la cámara
# ================================
cap = cv2.VideoCapture(0)

ret, img = cap.read()

if not ret:
    print("No se pudo abrir la cámara.")
    exit()

# Convertir el primer frame a float
averageValue1 = np.float32(img)

print("Presiona 'S' para guardar.")
print("Presiona 'ESC' para salir.")

while True:

    ret, img = cap.read()

    if not ret:
        break

    # Actualizar el fondo
    cv2.accumulateWeighted(img, averageValue1, 0.02)

    # Convertir a imagen
    resultingFrames1 = cv2.convertScaleAbs(averageValue1)

    # Mostrar ventanas
    cv2.imshow("Original Frame", img)
    cv2.imshow("Background (Running Average)", resultingFrames1)

    key = cv2.waitKey(30) & 0xFF

    # Guardar con la tecla S
    if key == ord('s'):

        original_path = os.path.join(out_dir, "imagen_original.jpg")
        background_path = os.path.join(out_dir, "imagen_difuminada.jpg")

        print("\nIntentando guardar...")

        ok1 = cv2.imwrite(original_path, img)
        ok2 = cv2.imwrite(background_path, resultingFrames1)

        print("Original:", original_path)
        print("Difuminada:", background_path)

        print("Resultado imagen original :", ok1)
        print("Resultado imagen fondo    :", ok2)

        if ok1 and ok2:
            print("✓ Imágenes guardadas correctamente.")
        else:
            print("✗ Error al guardar las imágenes.")

    # Salir con ESC
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()