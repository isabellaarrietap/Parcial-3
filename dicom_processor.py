import os
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt

class DicomDatasetLoader:
    """Clase encargada de cargar los archivos DICOM, extraer metadatos y la matriz de píxeles."""
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.slices = []
        self.pixel_matrix = None
        self.rows = 0
        self.cols = 0
        self.patient_name = ""
        self.modality = ""
        self.voxel_spacing = {}  # alto, largo, espesor (slice thickness)
        
        self._load_dicom_files()

    def _load_dicom_files(self):
        files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.dcm') or len(os.path.splitext(f)[1]) == 0]
        dataset_list = []
        for f in files:
            try:
                ds = pydicom.dcmread(f)
                dataset_list.append(ds)
            except Exception as e:
                continue
                
        if not dataset_list:
            raise ValueError(f"No se encontraron archivos DICOM válidos en la ruta: {self.folder_path}")

        try:
            dataset_list.sort(key=lambda x: float(x.SliceLocation))
        except AttributeError:
            try:
                dataset_list.sort(key=lambda x: int(x.InstanceNumber))
            except AttributeError:
                pass

        first_ds = dataset_list[0]
        self.rows = int(first_ds.Rows)
        self.cols = int(first_ds.Columns)
        self.patient_name = str(getattr(first_ds, 'PatientName', 'Desconocido'))
        self.modality = str(getattr(first_ds, 'Modality', ''))
        
        pixel_spacing = getattr(first_ds, 'PixelSpacing', [1.0, 1.0])
        slice_thickness = getattr(first_ds, 'SliceThickness', 1.0)
        
        self.voxel_spacing = {
            'alto': float(pixel_spacing[0]),
            'largo': float(pixel_spacing[1]),
            'espesor': float(slice_thickness)
        }

        self.slices = [ds.pixel_array for ds in dataset_list]
        self.pixel_matrix = np.stack(self.slices, axis=0)


class Dicom3DProcessor(DicomDatasetLoader):
    """Clase heredera para procesamiento avanzado: escalado 3D, rotación, ecualización y segmentación."""
    def __init__(self, folder_path):
        super().__init__(folder_path)
        self.volume_scaled = None
        self.volume_rotated = None
        self._scale_volume()

    def _scale_volume(self):
        scaled_slices = []
        for s in self.slices:
            new_dims = (s.shape[1] * 2, s.shape[0] * 2)
            scaled_slice = cv2.resize(s.astype(np.float32), new_dims, interpolation=cv2.INTER_LINEAR)
            scaled_slices.append(scaled_slice)
        self.volume_scaled = np.stack(scaled_slices, axis=0)
        self.volume_rotated = self.volume_scaled.copy()

    def corregir_angulo(self, angulo_grados):
        rotated_slices = []
        num_slices, h, w = self.volume_scaled.shape
        center = (w // 2, h // 2)
        
        rot_matrix = cv2.getRotationMatrix2D(center, angulo_grados, 1.0)
        
        for i in range(num_slices):
            slice_img = self.volume_scaled[i]
            rotated = cv2.warpAffine(slice_img, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
            rotated_slices.append(rotated)
            
        self.volume_rotated = np.stack(rotated_slices, axis=0)
        print(f"Ángulo corregido en {angulo_grados}° para todo el volumen.")

    def procesar_ecualizacion(self, indice_corte, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        if indice_corte >= self.volume_rotated.shape[0]:
            indice_corte = 0
            
        corte = self.volume_rotated[indice_corte]
        corte_norm = cv2.normalize(corte, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        corte_eq = cv2.equalizeHist(corte_norm)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 2, 1)
        plt.imshow(corte_norm, cmap='gray')
        plt.title(f"Corte Normalizado ({indice_corte})")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.hist(corte_norm.ravel(), bins=256, range=[0, 256], color='gray')
        plt.title("Histograma Normalizado")
        
        plt.subplot(2, 2, 3)
        plt.imshow(corte_eq, cmap='gray')
        plt.title("Corte Ecualizado")
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.hist(corte_eq.ravel(), bins=256, range=[0, 256], color='black')
        plt.title("Histograma Ecualizado")
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"ecualizacion_corte_{indice_corte}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Gráfica e imagen de ecualización guardadas en: {plot_path}")

    def segmentar_y_analizar(self, plano='transversal', indice_corte=0, umbral=100, kernel_size=3, iteraciones=1, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        
        if plano == 'transversal':
            corte = self.volume_rotated[indice_corte]
        elif plano == 'sagital':
            corte = self.volume_rotated[:, :, indice_corte]
        elif plano == 'coronal':
            corte = self.volume_rotated[:, indice_corte, :]
        else:
            raise ValueError("Plano no válido. Use: 'transversal', 'sagital' o 'coronal'.")

        corte_uint8 = cv2.normalize(corte, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binaria = cv2.threshold(corte_uint8, umbral, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        apertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=iteraciones)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(apertura, connectivity=8)
        img_rgb = cv2.cvtColor(corte_uint8, cv2.COLOR_GRAY2RGB)
        
        area_mm2 = 0.0
        if num_labels > 1:
            valid_labels = np.argsort(stats[1:, cv2.CC_STAT_AREA]) + 1
            target_label = valid_labels[-1]
            
            x = stats[target_label, cv2.CC_STAT_LEFT]
            y = stats[target_label, cv2.CC_STAT_TOP]
            w = stats[target_label, cv2.CC_STAT_WIDTH]
            h = stats[target_label, cv2.CC_STAT_HEIGHT]
            area_pixels = stats[target_label, cv2.CC_STAT_AREA]
            
            area_mm2 = area_pixels * self.voxel_spacing['alto'] * self.voxel_spacing['largo']
            
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            radio = int(max(w, h) / 2) + 20
            
            cv2.circle(img_rgb, (center_x, center_y), radio, (42, 72, 230), 3)
            texto = f"Area: {area_mm2:.2f} mm2"
            cv2.putText(img_rgb, texto, (center_x - radio, center_y - radio - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (42, 72, 230), 2, cv2.LINE_AA)

        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        plt.title(f"Segmentación y Círculo ({plano.capitalize()}) - Área: {area_mm2:.2f} mm²")
        plt.axis('on')
        
        save_path = os.path.join(output_dir, f"segmentacion_{plano}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Imagen segmentada guardada en: {save_path} | Área calculada: {area_mm2:.2f} mm²")