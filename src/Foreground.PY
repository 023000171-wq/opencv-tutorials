# organize imports
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Ruta a la imagen de entrada (dentro de la carpeta data)
input_path = os.path.join('data', 'VACA.png')
image = cv2.imread(input_path)

if image is None:
    print(f"Error: No se pudo cargar la imagen desde {input_path}")
    exit()

# create a simple mask image similar to the loaded image
mask = np.zeros(image.shape[:2], np.uint8)

# specify the background and foreground model
backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)

# 2. Definir la Región de Interés (ROI) para encuadrar a la vaca
# Conseguimos el ancho y alto de tu imagen para hacer un recuadro estimado
h, w = image.shape[:2]
# El formato es (X_inicio, Y_inicio, ancho, alto)
# Dejamos un pequeño margen del 5% alrededor de los bordes para asegurar que encierre a la vaca
rectangle = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))

# apply the grabcut algorithm
cv2.grabCut(image, mask, rectangle,  
            backgroundModel, foregroundModel,
            5, cv2.GC_INIT_WITH_RECT) # Subimos a 5 iteraciones para mejor precisión

# Post-procesamiento de la máscara
mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')

# Multiplicar la máscara por la imagen original
image_segmented = image * mask2[:, :, np.newaxis]

# 3. GUARDAR EL RESULTADO EN LA CARPETA 'out'
# Nos aseguramos de que la carpeta 'out' exista
os.makedirs('out', exist_ok=True)

output_path = os.path.join('out', 'VACA_segmentada.png')
cv2.imwrite(output_path, image_segmented)
print(f"¡Imagen segmentada guardada con éxito en: {output_path}!")

# Mostrar visualización en pantalla (opcional)
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(cv2.cvtColor(image_segmented, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()