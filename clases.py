"""
Parcial 3 - Informática II 2026-2 - Universidad de Antioquia, Bioingeniería
Clases para la gestión de volúmenes DICOM (PyDicom + OpenCV).

Clases:
    - VolumenDicom:   representa UNA carpeta DICOM (un estudio). Guarda los datos
                      del dataset, el volumen reconstruido (cortes x2) y procesa.
    - GestorImagenes: administra varios objetos VolumenDicom (PPMI, Sarcoma, T2...).
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom


class VolumenDicom:
    """Estudio DICOM completo cargado desde una carpeta."""

    PLANOS = ("transversal", "coronal", "sagital")

    def __init__(self, carpeta, nombre=None, dirResultados="resultados"):
        # --- Identificación ---
        self.carpeta = carpeta
        self.nombre = nombre if nombre else os.path.basename(os.path.normpath(carpeta))
        self.dirResultados = dirResultados

        # --- Información extraída directamente del dataset ---
        self.filas = None
        self.columnas = None
        self.nombrePaciente = None
        self.modalidad = None
        self.pixelSpacing = None      # (alto, largo) en mm
        self.sliceThickness = None    # ancho en mm
        self.voxel = None             # (alto, largo, ancho) en mm del volumen original
        self.matrizPixeles = None     # cortes originales, arreglo (Z, filas, columnas)
        self.numCortes = 0

        # --- Volumen reconstruido (cada corte escalado al doble) ---
        self.volumen = None           # arreglo (Z, 2*filas, 2*columnas)
        self.voxelEscalado = None     # (alto, largo, ancho) tras escalar x2 en el plano

        # --- Resultados de procesos ---
        self.volumenCorregido = None  # resultado de corregirAngulo
        self.anguloAplicado = 0.0
        self.matrizRotacion = None
        self.volumenCopia = None      # copia usada en la ecualización
        self.imagenEcualizada = None
        self.imagenSegmentada = None

        self.cargar()
        self.reconstruir3D()

    # ------------------------------------------------------------------ #
    #  Carga y reconstrucción
    # ------------------------------------------------------------------ #
    def _listarArchivos(self):
        """Archivos que están directamente en la carpeta (sin entrar a subcarpetas),
        para que cada estudio sea un objeto y no se mezclen varios pacientes."""
        archivos = []
        for n in os.listdir(self.carpeta):
            ruta = os.path.join(self.carpeta, n)
            if os.path.isfile(ruta):
                archivos.append(ruta)
        return sorted(archivos)

    @staticmethod
    def _claveOrden(ds, ruta):
        """Orden anatómico: posición z, luego InstanceNumber, luego nombre."""
        try:
            return (0, float(ds.ImagePositionPatient[2]), ruta)
        except Exception:
            pass
        try:
            return (1, float(ds.InstanceNumber), ruta)
        except Exception:
            return (2, 0.0, ruta)

    def cargar(self):
        """Lee los archivos DICOM de la carpeta y llena los atributos."""
        if not os.path.isdir(self.carpeta):
            raise FileNotFoundError(f"No existe la carpeta: {self.carpeta}")

        lecturas = []
        for ruta in self._listarArchivos():
            try:
                ds = pydicom.dcmread(ruta)
                if "PixelData" not in ds:
                    continue
                lecturas.append((self._claveOrden(ds, ruta), ds))
            except Exception:
                continue  # no es DICOM, se ignora

        if not lecturas:
            raise ValueError(f"No se encontraron archivos DICOM con imagen en {self.carpeta}")

        lecturas.sort(key=lambda t: t[0])
        datasets = [ds for _, ds in lecturas]

        # Cortes (un archivo puede traer varios cortes si es multiframe)
        cortes = []
        for ds in datasets:
            px = ds.pixel_array.astype(np.float32)
            if px.ndim == 2:
                cortes.append(px)
            elif px.ndim == 3 and px.shape[-1] not in (3, 4):
                cortes.extend(list(px))          # multiframe
            elif px.ndim == 3:
                cortes.append(cv2.cvtColor(px.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32))

        # Conservar sólo los cortes con la forma más frecuente (descarta localizers)
        formas = [c.shape for c in cortes]
        formaComun = max(set(formas), key=formas.count)
        cortes = [c for c in cortes if c.shape == formaComun]

        ds0 = datasets[0]
        self.matrizPixeles = np.stack(cortes, axis=0)
        self.numCortes = self.matrizPixeles.shape[0]
        self.filas = int(getattr(ds0, "Rows", formaComun[0]))
        self.columnas = int(getattr(ds0, "Columns", formaComun[1]))
        self.nombrePaciente = str(getattr(ds0, "PatientName", "Desconocido"))
        self.modalidad = str(getattr(ds0, "Modality", "N/D"))

        ps = getattr(ds0, "PixelSpacing", None)
        if ps is None:
            ps = getattr(ds0, "ImagerPixelSpacing", [1.0, 1.0])
        self.pixelSpacing = (float(ps[0]), float(ps[1]))
        self.sliceThickness = float(getattr(ds0, "SliceThickness", 1.0))
        # voxel: (alto, largo, ancho) en mm
        self.voxel = (self.pixelSpacing[0], self.pixelSpacing[1], self.sliceThickness)

    def reconstruir3D(self):
        """Escala cada corte al doble de su tamaño y apila el volumen 3D."""
        escalados = [
            cv2.resize(c, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            for c in self.matrizPixeles
        ]
        self.volumen = np.stack(escalados, axis=0)
        # al duplicar filas/columnas, el pixel mide la mitad; el grosor de corte no cambia
        self.voxelEscalado = (self.voxel[0] / 2, self.voxel[1] / 2, self.voxel[2])

    # ------------------------------------------------------------------ #
    #  Utilidades
    # ------------------------------------------------------------------ #
    def volumenActivo(self):
        """Volumen con la corrección de ángulo si existe; si no, el reconstruido."""
        return self.volumenCorregido if self.volumenCorregido is not None else self.volumen

    @staticmethod
    def _aUint8(img):
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _guardarEn(self, nombreArchivo):
        os.makedirs(self.dirResultados, exist_ok=True)
        return os.path.join(self.dirResultados, nombreArchivo)

    def maxIndice(self, plano):
        z, h, w = self.volumenActivo().shape
        return {"transversal": z, "coronal": h, "sagital": w}[plano] - 1

    def obtenerCorte(self, plano, indice):
        """
        Devuelve (corte2D, espaciadoFilas_mm, espaciadoColumnas_mm).
        El volumen tiene forma (Z, alto, largo):
            transversal -> vol[z]      filas=alto,  columnas=largo
            coronal     -> vol[:, y, :] filas=Z(ancho), columnas=largo
            sagital     -> vol[:, :, x] filas=Z(ancho), columnas=alto
        """
        if plano not in self.PLANOS:
            raise ValueError("Plano inválido")
        vol = self.volumenActivo()
        alto, largo, ancho = self.voxelEscalado
        if not 0 <= indice <= self.maxIndice(plano):
            raise IndexError(f"Índice fuera de rango (0 - {self.maxIndice(plano)})")
        if plano == "transversal":
            return vol[indice], alto, largo
        if plano == "coronal":
            return vol[:, indice, :], ancho, largo
        return vol[:, :, indice], ancho, alto

    def info(self):
        sep = "-" * 52
        return (
            f"{sep}\n"
            f"Objeto        : {self.nombre}\n"
            f"Carpeta       : {self.carpeta}\n"
            f"Paciente      : {self.nombrePaciente}\n"
            f"Modalidad     : {self.modalidad}\n"
            f"Filas x Cols  : {self.filas} x {self.columnas}\n"
            f"N° de cortes  : {self.numCortes}\n"
            f"Voxel original: alto={self.voxel[0]:.3f} largo={self.voxel[1]:.3f} "
            f"ancho={self.voxel[2]:.3f} mm\n"
            f"Voxel x2      : alto={self.voxelEscalado[0]:.3f} largo={self.voxelEscalado[1]:.3f} "
            f"ancho={self.voxelEscalado[2]:.3f} mm\n"
            f"Volumen 3D    : {self.volumen.shape} (Z, filas, columnas)\n"
            f"Ángulo corregido: {self.anguloAplicado}°\n"
            f"{sep}"
        )

    # ------------------------------------------------------------------ #
    #  4. Corrección de ángulo
    # ------------------------------------------------------------------ #
    def corregirAngulo(self, angulo):
        """
        Rota cada corte transversal 'angulo' grados alrededor del punto medio
        de la imagen. El resultado queda en self.volumenCorregido.
        """
        angulo = float(angulo)
        _, h, w = self.volumen.shape
        centro = (w / 2.0, h / 2.0)
        self.matrizRotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)  # 2x3

        rotados = [
            cv2.warpAffine(c, self.matrizRotacion, (w, h), flags=cv2.INTER_LINEAR)
            for c in self.volumen
        ]
        self.volumenCorregido = np.stack(rotados, axis=0)
        self.anguloAplicado = angulo
        return self.volumenCorregido

    # ------------------------------------------------------------------ #
    #  5. Normalización + ecualización
    # ------------------------------------------------------------------ #
    def ecualizarCorte(self, indice, mostrar=True):
        """
        Copia el volumen, normaliza a uint8 el corte transversal 'indice',
        lo ecualiza, grafica histogramas + imagen y guarda gráfico e imagen.
        """
        self.volumenCopia = self.volumenActivo().copy()
        if not 0 <= indice < self.volumenCopia.shape[0]:
            raise IndexError(f"Corte fuera de rango (0 - {self.volumenCopia.shape[0] - 1})")

        original = self._aUint8(self.volumenCopia[indice])
        ecualizada = cv2.equalizeHist(original)
        self.imagenEcualizada = ecualizada

        fig, ax = plt.subplots(2, 2, figsize=(11, 9))
        ax[0, 0].imshow(original, cmap="gray")
        ax[0, 0].set_title("Corte normalizado (uint8)")
        ax[0, 1].imshow(ecualizada, cmap="gray")
        ax[0, 1].set_title("Imagen ecualizada")
        ax[1, 0].hist(original.ravel(), bins=256, range=(0, 256), color="gray")
        ax[1, 0].set_title("Histograma original")
        ax[1, 1].hist(ecualizada.ravel(), bins=256, range=(0, 256), color="steelblue")
        ax[1, 1].set_title("Histograma ecualizado")
        for a in ax[0]:
            a.axis("off")
        fig.suptitle(f"{self.nombre} - corte {indice}")
        fig.tight_layout()

        rutaGrafico = self._guardarEn(f"{self.nombre}_ecualizacion_corte{indice}_grafico.png")
        rutaImagen = self._guardarEn(f"{self.nombre}_ecualizacion_corte{indice}_imagen.png")
        fig.savefig(rutaGrafico, dpi=120)
        cv2.imwrite(rutaImagen, ecualizada)
        if mostrar:
            plt.show()
        plt.close(fig)
        return rutaGrafico, rutaImagen

    # ------------------------------------------------------------------ #
    #  6. Segmentación de huecos anatómicos
    # ------------------------------------------------------------------ #
    def segmentarCorte(self, plano, indice, umbral=None, kernel=5, iteraciones=1, mostrar=True):
        """
        Binariza/umbraliza un corte (sagital, coronal o transversal), aplica
        apertura (kernel e iteraciones definidos por el usuario), encuentra el
        elemento de interés con connectedComponentsWithStats, dibuja un círculo
        en color y calcula su área en mm². Grafica y guarda la imagen resultante.

        umbral: valor 0-255 o None para usar Otsu.
        Los huecos (zonas oscuras) quedan en blanco: THRESH_BINARY_INV.
        """
        corte, espFilas, espCols = self.obtenerCorte(plano, indice)
        img8 = self._aUint8(corte)

        # Binarización / umbralización
        if umbral is None:
            valorUmbral, binaria = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            valorUmbral, binaria = cv2.threshold(img8, float(umbral), 255, cv2.THRESH_BINARY_INV)

        # Apertura
        k = max(1, int(kernel))
        nucleo = np.ones((k, k), np.uint8)
        abierta = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, nucleo, iterations=max(1, int(iteraciones)))

        # Componentes conectados
        n, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(abierta, connectivity=8)
        if n <= 1:
            print("No se encontró ningún elemento de interés con esos parámetros.")
            return None

        alto, ancho = abierta.shape
        candidatos = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            tocaBorde = x == 0 or y == 0 or x + w >= ancho or y + h >= alto
            candidatos.append((tocaBorde, int(area), i))
        # Se prefieren regiones interiores (huecos), no el fondo que toca los bordes
        interiores = [c for c in candidatos if not c[0]]
        elegido = max(interiores or candidatos, key=lambda c: c[1])[2]

        x, y, w, h, areaPx = stats[elegido]
        cx, cy = centroides[elegido]
        areaMM2 = float(areaPx) * espFilas * espCols
        anchoMM = float(w) * espCols
        altoMM = float(h) * espFilas

        # Imagen en RGB para dibujar en color
        rgb = cv2.cvtColor(abierta, cv2.COLOR_GRAY2RGB)
        radio = int(np.hypot(w, h) / 2 + 0.05 * max(alto, ancho))
        grosor = max(2, min(alto, ancho) // 120)
        cv2.circle(rgb, (int(cx), int(cy)), radio, (255, 128, 0), grosor)
        self.imagenSegmentada = rgb

        # Región recortada (alrededor del elemento)
        m = int(0.25 * max(w, h)) + 5
        y0, y1 = max(0, y - m), min(alto, y + h + m)
        x0, x1 = max(0, x - m), min(ancho, x + w + m)
        recorte = img8[y0:y1, x0:x1]

        aspecto = espFilas / espCols
        fig, ax = plt.subplots(2, 2, figsize=(11, 10))
        ax[0, 0].imshow(img8, cmap="gray", aspect=aspecto)
        ax[0, 0].set_title(f"Corte {plano} #{indice} (uint8)")
        ax[0, 1].imshow(abierta, cmap="gray", aspect=aspecto)
        ax[0, 1].set_title(f"Binaria (umbral={valorUmbral:.0f}) + apertura k={k}, it={int(iteraciones)}")
        ax[1, 0].imshow(rgb, aspect=aspecto)
        ax[1, 0].set_title("Elemento de interés")
        ax[1, 1].imshow(recorte, cmap="gray", aspect=aspecto)
        ax[1, 1].set_title("Región recortada")
        ax[1, 1].set_xlabel(
            f"Ancho: {anchoMM:.2f} mm | Alto: {altoMM:.2f} mm | Área: {areaMM2:.2f} mm²"
        )
        fig.suptitle(f"{self.nombre} - Área aproximada: {areaMM2:.2f} mm²")
        fig.tight_layout()

        base = f"{self.nombre}_segmentacion_{plano}{indice}"
        rutaGrafico = self._guardarEn(base + "_grafico.png")
        rutaImagen = self._guardarEn(base + "_imagen.png")
        fig.savefig(rutaGrafico, dpi=120)
        cv2.imwrite(rutaImagen, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if mostrar:
            plt.show()
        plt.close(fig)

        print(f"Área aproximada: {areaMM2:.2f} mm² | {anchoMM:.2f} x {altoMM:.2f} mm")
        return {"areaMM2": areaMM2, "anchoMM": anchoMM, "altoMM": altoMM,
                "grafico": rutaGrafico, "imagen": rutaImagen}


class GestorImagenes:
    """Administra varios objetos VolumenDicom (uno por carpeta)."""

    def __init__(self, dirResultados="resultados"):
        self.dirResultados = dirResultados
        self.volumenes = {}

    def agregarCarpeta(self, carpeta, nombre=None):
        vol = VolumenDicom(carpeta, nombre, self.dirResultados)
        self.volumenes[vol.nombre] = vol
        return vol

    def cargarVarias(self, rutaBase, nombres):
        """
        Carga rutaBase/nombre para cada nombre. Si la carpeta tiene subcarpetas
        con estudios (ej. PPMI/3105, PPMI/3128), cada una se crea como un objeto
        aparte llamado 'PPMI_3105', 'PPMI_3128', etc. Los archivos que no son
        DICOM (como .zip sueltos) se ignoran.
        """
        for nombre in nombres:
            raiz = os.path.join(rutaBase, nombre)
            if not os.path.isdir(raiz):
                print(f"[ERROR] {nombre}: no existe la carpeta {raiz}")
                continue
            cargados = 0
            for ruta, _, archivos in sorted(os.walk(raiz)):
                if not archivos:
                    continue
                rel = os.path.relpath(ruta, raiz)
                nombreObj = nombre if rel == "." else f"{nombre}_{rel.replace(os.sep, '_')}"
                try:
                    self.agregarCarpeta(ruta, nombreObj)
                    print(f"[OK] {nombreObj} cargado.")
                    cargados += 1
                except Exception:
                    continue  # carpeta sin DICOM (por ejemplo, solo .zip)
            if cargados == 0:
                print(f"[ERROR] {nombre}: no se encontraron imágenes DICOM.")

    def obtener(self, nombre):
        return self.volumenes.get(nombre)

    def nombres(self):
        return list(self.volumenes.keys())

    def listar(self):
        if not self.volumenes:
            print("No hay objetos cargados.")
        for v in self.volumenes.values():
            print(v.info())

