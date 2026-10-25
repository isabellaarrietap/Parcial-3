# Parcial 3 - Informática II
  ## Integrantes
   - Isabella Arrieta
   - Isabela López
 ## Descripción
   Programa en Python con POO para analizar imágenes DICOM del Hospital San Vicente Fundación usando PyDicom y OpenCV.
## Requisitos
- Python 3
- pydicom
- opencv-python
- numpy
- matplotlib
## Estructura del proyecto
- clases.py: clases VolumenDicom y GestorImagenes.
- main.py: menu de uso e implementacion.
- resultados/: graficos e imagenes generados.
- datos/: carpetas PPMI, Sarcoma y T2 (no se suben a GitHub).
## Como ejecutar
1. Copiar las carpetas PPMI, Sarcoma y T2 dentro de una carpeta llamada datos.
2. Instalar las librerias: pip install pydicom opencv-python matplotlib numpy
3. Ejecutar: py main.py
## Clase VolumenDicom
Carga una carpeta DICOM y guarda como atributos las filas, columnas, nombre del paciente, modalidad, tamano del voxel en mm y la matriz de pixeles. Escala cada corte al doble de su tamano y reconstruye el volumen 3D.
## Metodo corregirAngulo
Recibe un angulo en grados, calcula la matriz de rotacion tomando como centro el punto medio de la imagen, aplica la transformacion a todos los cortes y guarda el resultado como atributo del objeto.
