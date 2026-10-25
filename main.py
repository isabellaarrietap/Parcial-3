"""
Parcial 3 - Informática II 2026-2
Implementación: menú para trabajar con los objetos PPMI, Sarcoma y T2.
Requiere clases.py en la misma carpeta y las imágenes en ./datos/<PPMI|Sarcoma|T2>
"""

from clases import GestorImagenes, VolumenDicom

RUTA_DATOS = "datos"
CARPETAS = ["PPMI", "Sarcoma", "T2"]


def pedirNumero(texto, tipo=float, defecto=None):
    while True:
        valor = input(texto).strip()
        if valor == "" and defecto is not None:
            return defecto
        try:
            return tipo(valor)
        except ValueError:
            print("Valor inválido, intente de nuevo.")


def elegirObjeto(gestor):
    nombres = gestor.nombres()
    if not nombres:
        print("Primero cargue las carpetas (opción 1).")
        return None
    print("Objetos disponibles:", ", ".join(nombres))
    nombre = input("Nombre del objeto: ").strip()
    obj = gestor.obtener(nombre)
    if obj is None:
        print("Objeto no encontrado.")
    return obj


def opcionCargar(gestor):
    print("1) Cargar PPMI, Sarcoma y T2 desde ./datos")
    print("2) Cargar otra carpeta")
    op = input("Opción: ").strip()
    if op == "1":
        gestor.cargarVarias(RUTA_DATOS, CARPETAS)
    elif op == "2":
        ruta = input("Ruta de la carpeta DICOM: ").strip()
        try:
            v = gestor.agregarCarpeta(ruta)
            print(f"[OK] {v.nombre} cargado.")
        except Exception as e:
            print(f"[ERROR] {e}")


def opcionAngulo(gestor):
    obj = elegirObjeto(gestor)
    if obj:
        ang = pedirNumero("Ángulo de inclinación en grados: ", float)
        obj.corregirAngulo(ang)
        print(f"Corrección de {ang}° aplicada a {obj.nombre}.")


def opcionEcualizar(gestor):
    obj = elegirObjeto(gestor)
    if obj:
        maxi = obj.maxIndice("transversal")
        i = pedirNumero(f"Corte transversal (0-{maxi}): ", int)
        try:
            g, im = obj.ecualizarCorte(i)
            print(f"Guardados:\n  {g}\n  {im}")
        except Exception as e:
            print(f"[ERROR] {e}")


def opcionSegmentar(gestor):
    obj = elegirObjeto(gestor)
    if not obj:
        return
    plano = input("Plano (transversal / coronal / sagital): ").strip().lower()
    if plano not in VolumenDicom.PLANOS:
        print("Plano inválido.")
        return
    maxi = obj.maxIndice(plano)
    i = pedirNumero(f"Índice del corte (0-{maxi}): ", int)
    umbral = pedirNumero("Umbral 0-255 (Enter = Otsu automático): ", float, defecto=-1)
    umbral = None if umbral < 0 else umbral
    k = pedirNumero("Tamaño del kernel (ej. 5): ", int, defecto=5)
    it = pedirNumero("Iteraciones de la apertura (ej. 1): ", int, defecto=1)
    try:
        res = obj.segmentarCorte(plano, i, umbral, k, it)
        if res:
            print(f"Guardados:\n  {res['grafico']}\n  {res['imagen']}")
    except Exception as e:
        print(f"[ERROR] {e}")


def main():
    gestor = GestorImagenes(dirResultados="resultados")
    acciones = {
        "1": ("Cargar carpetas DICOM", opcionCargar),
        "2": ("Ver información de los objetos", lambda g: g.listar()),
        "3": ("Corregir ángulo (corregirAngulo)", opcionAngulo),
        "4": ("Normalizar + ecualizar un corte", opcionEcualizar),
        "5": ("Segmentar huecos y medir área", opcionSegmentar),
    }
    while True:
        print("\n===== ANÁLISIS DE IMÁGENES DICOM =====")
        for k, (texto, _) in acciones.items():
            print(f"{k}. {texto}")
        print("0. Salir")
        op = input("Seleccione: ").strip()
        if op == "0":
            print("Hasta luego.")
            break
        if op in acciones:
            acciones[op][1](gestor)
        else:
            print("Opción inválida.")


if __name__ == "__main__":
    main()
