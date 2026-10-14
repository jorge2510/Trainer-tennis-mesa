# -*- coding: utf-8 -*-
"""
==================================================================
  ENTRENADOR - GOLPE DE DERECHA (Forehand)
  Tenis de mesa - trt_pose (TensorRT) + OpenCV - Jetson Nano
==================================================================
  Version adaptada de forehand_trainer.py (PC / MediaPipe)
  para usar trt_pose (18 keypoints COCO) en el Jetson Nano.

  Uso:
      sudo nvpmodel -m 0 && sudo jetson_clocks      (recomendado)
      python3 forehand_trainer_jetson_trt.py
"""

import os
import csv
import time
import math
import threading
import unicodedata
import statistics as _st
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch

import trt_pose.coco
from trt_pose.parse_objects import ParseObjects
from torch2trt import TRTModule

try:                                   # solo para el texto del resumen
    from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont
    _HAY_PIL = True
except Exception:                      # pragma: no cover
    _HAY_PIL = False


# ==============================================================
# RUTAS Y CAMARA - ajusta estas lineas si hace falta
# ==============================================================

MODEL_WEIGHTS   = "resnet18_baseline_att_224x224_A_epoch_249.pth"
OPTIMIZED_MODEL = "resnet18_baseline_att_224x224_A_epoch_249_trt.pth"

WIDTH, HEIGHT = 224, 224          # tamano de entrada del modelo
CAP_WIDTH, CAP_HEIGHT = 640, 480  # resolucion de captura de camara
CAP_FPS = 30
CAMARA_INDICE = 0                 # camara USB (/dev/video0)
CAMARA_MJPG = True                # si la camara da error o imagen negra -> False


def log(msg) -> None:
    """print() seguro: si el programa arranca sin locale UTF-8 (por ejemplo
    como servicio al encender la Jetson), un print con tildes revienta."""
    print(texto_ascii(msg))

# ==============================================================
# TOPOLOGIA - 18 keypoints COCO usados por trt_pose
# ==============================================================

HUMAN_POSE = {
    "keypoints": [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle", "neck",
    ],
    "skeleton": [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
        [1, 3], [2, 4], [3, 5], [4, 6], [17, 0], [17, 5], [17, 6],
        [17, 11], [17, 12],
    ],
}

# Conexiones para dibujar el esqueleto (indices 0-based, listos para cv2)
_CONEXIONES = [(a, b) for a, b in HUMAN_POSE["skeleton"]]

# Puntos de la cabeza (nariz, ojos, orejas) para el filtro de oclusion
_CABEZA = (0, 1, 2, 3, 4)


# ==============================================================
# TEXTO EN PANTALLA - solo ASCII
# Las fuentes Hershey de OpenCV no dibujan tildes, la letra n con
# tilde ni el simbolo de grados: salen como "??". Todo texto pasa
# por texto_ascii() antes de dibujarse.
# ==============================================================

_REEMPLAZOS_TXT = {
    "\u00b0": "", "\u2192": "->", "\u2190": "<-", "\u2013": "-",
    "\u2014": "-", "\u2713": "OK", "\u2717": "X", "\u2248": "~",
    "\u00f1": "n", "\u00d1": "N",
}


def texto_ascii(s) -> str:
    s = str(s)
    for k, v in _REEMPLAZOS_TXT.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii")


def poner_texto(frame, texto, org, escala, color, grosor=1):
    cv2.putText(frame, texto_ascii(texto), org, cv2.FONT_HERSHEY_SIMPLEX,
                escala, color, grosor, cv2.LINE_AA)


# ==============================================================
# REGLAS - GOLPE DE DERECHA (drive basico)
# ==============================================================

@dataclass
class Reglas:
    """
    Umbrales en angulo INTERNO 2D (180 = articulacion recta), medidos
    en el instante del golpe. Son limites de partida basados en:

      [1] Bankosz & Winiarski (2020), J Sports Sci Med 19:138-148.
          Topspin de elite (7 jugadores, sensores inerciales). Flexion
          media del codo en el contacto 47.1 (DE 30.1) -> ~133 interno.
          Flexion de rodillas en el contacto ~50-56 -> ~124-130 interno.
          Conclusion clave: alta variabilidad entre jugadores; se
          recomienda individualizar (por eso existe el Calibrador).
      [2] Guia tecnica drive basico: codo ~90-100, rodillas flexionadas
          15-20 grados (~160-165 interno).
      [3] Lyu et al. (2025), Front Sports Act Living 7:1635581.
          Estudio con MediaPipe y 1 camara frontal-lateral: el golpe es
          mas rapido con menos movimiento de rodillas durante el swing.

    IMPORTANTE: la camara ve una proyeccion 2D. Con la camara ya fija en
    su posicion final, captura ~20 golpes con SPACE y ajusta estos
    valores con el resumen del Calibrador.
    """
    # Codo (hombro-codo-muneca)
    codo_ideal_min:     float = 100.0
    codo_ideal_max:     float = 135.0
    codo_warn_min:      float = 90.0     # por debajo -> error
    codo_warn_max:      float = 150.0    # por encima -> error

    # Caida de muneca: (y_muneca - y_codo) / largo del brazo superior
    # 0.0 = muneca a la altura del codo; positivo = muneca mas baja
    muneca_caida_warn:  float = 0.30
    muneca_caida_error: float = 0.70

    # Rodilla dominante (cadera-rodilla-tobillo)
    rodilla_ideal_min:  float = 135.0
    rodilla_ideal_max:  float = 165.0
    rodilla_error_min:  float = 120.0    # muy agachado
    rodilla_error_max:  float = 170.0    # piernas rectas

    # Estabilidad de rodilla durante el golpe (rango en grados, solo aviso).
    # Bankosz & Winiarski (2018) reportan rangos de flexo-extension de
    # rodilla de mas de 40 grados en la fase de golpe del topspin de elite,
    # asi que un rango moderado es normal; solo avisamos si es muy grande.
    rodilla_rango_warn:  float = 35.0

    # Filtro de oclusion del brazo
    antebrazo_ratio_min: float = 0.35    # antebrazo / brazo superior
    antebrazo_ratio_max: float = 2.20
    codo_min_fisico:     float = 40.0    # por debajo es lectura imposible
    muneca_cabeza_min:   float = 0.50    # distancia muneca-cabeza / brazo sup.
    brazo_sup_min_px:    float = 12.0    # brazo demasiado pequeno = muy lejos


# ==============================================================
# INDICES DE KEYPOINTS (formato trt_pose / COCO-18)
# ==============================================================

_IDX = {
    "derecha": {
        "hombro":      6,   # right_shoulder
        "codo":        8,   # right_elbow
        "muneca":      10,  # right_wrist
        "cadera":      12,  # right_hip
        "rodilla":     14,  # right_knee
        "tobillo":     16,  # right_ankle
        "cadera_opp":  11,  # left_hip
        "rodilla_opp": 13,  # left_knee
        "tobillo_opp": 15,  # left_ankle
    },
    "zurdo": {
        "hombro":      5,   # left_shoulder
        "codo":        7,   # left_elbow
        "muneca":      9,   # left_wrist
        "cadera":      11,  # left_hip
        "rodilla":     13,  # left_knee
        "tobillo":     15,  # left_ankle
        "cadera_opp":  12,  # right_hip
        "rodilla_opp": 14,  # right_knee
        "tobillo_opp": 16,  # right_ankle
    },
}


# ==============================================================
# PUNTO - coordenadas normalizadas (0-1) de un keypoint
# ==============================================================

class Punto:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


# ==============================================================
# GEOMETRIA - todo en PIXELES para no deformar los angulos
# ==============================================================

def angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angulo en B (A-B-C). Retorna grados [0, 180]."""
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def pt_arr(p: Punto, W: int, H: int) -> np.ndarray:
    return np.array([p.x * W, p.y * H], dtype=np.float64)


def pt_px(p: Punto, W: int, H: int) -> Tuple[int, int]:
    return (int(p.x * W), int(p.y * H))


# ==============================================================
# SELECTOR DE JUGADOR - evita analizar a una segunda persona
# ==============================================================

class SelectorJugador:
    """
    1. La primera vez elige a la persona con mas puntos detectados
       (desempate: la mas alta en pantalla = la mas cercana).
    2. En los frames siguientes se queda con la persona mas cercana a la
       posicion anterior del jugador, aunque otra tenga mas puntos.
    3. Si el jugador desaparece o salta demasiado lejos, vuelve a elegir.
    4. Detecta INTERFERENCIA: si otra persona se superpone con el jugador
       (pasa por delante o esta pegada detras), el analisis se pausa,
       porque trt_pose puede mezclar brazos de las dos personas.
    """
    MIN_PUNTOS     = 5      # minimo de keypoints para considerar jugador
    MIN_PUNTOS_INT = 3      # minimo para contar como posible interferencia
    SALTO_MAX      = 0.25   # distancia normalizada maxima entre frames
    FRAMES_OLVIDO  = 15     # frames sin ver al jugador antes de re-elegir
    SOLAPE_MAX     = 0.05   # fraccion del recuadro del jugador cubierta por otro
    MARGEN_BOX     = 0.03   # margen alrededor del recuadro del jugador

    def __init__(self):
        self.centro_prev: Optional[Tuple[float, float]] = None
        self.frames_perdido = 0
        self.n_personas = 0
        self.interferencia = False

    @staticmethod
    def _solape(box_j, box_o) -> float:
        x1 = max(box_j[0], box_o[0])
        y1 = max(box_j[1], box_o[1])
        x2 = min(box_j[2], box_o[2])
        y2 = min(box_j[3], box_o[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        area_j = max((box_j[2] - box_j[0]) * (box_j[3] - box_j[1]), 1e-6)
        return (x2 - x1) * (y2 - y1) / area_j

    def elegir(self, count: int, objs: np.ndarray, pks: np.ndarray) -> int:
        """objs y pks ya en numpy (se convierten una sola vez por frame)."""
        self.n_personas = count
        self.interferencia = False
        if count == 0:
            self._perdido()
            return -1

        personas = []   # (indice, n_puntos, alto, cx, cy, box)
        for i in range(count):
            obj = objs[i]
            xs, ys = [], []
            for j in range(obj.shape[0]):
                k = int(obj[j])
                if k >= 0:
                    ys.append(float(pks[j][k][0]))
                    xs.append(float(pks[j][k][1]))
            n = len(xs)
            if n < self.MIN_PUNTOS_INT:
                continue
            box = (min(xs), min(ys), max(xs), max(ys))
            personas.append((i, n, box[3] - box[1], sum(xs) / n, sum(ys) / n, box))

        candidatos = [p for p in personas if p[1] >= self.MIN_PUNTOS]
        self.n_personas = len(candidatos)
        if not candidatos:
            self._perdido()
            return -1

        elegido = None
        if self.centro_prev is not None:
            px, py = self.centro_prev
            cercano = min(candidatos,
                          key=lambda c: (c[3] - px) ** 2 + (c[4] - py) ** 2)
            dist = ((cercano[3] - px) ** 2 + (cercano[4] - py) ** 2) ** 0.5
            if dist <= self.SALTO_MAX:
                elegido = cercano

        if elegido is None:
            if self.centro_prev is not None and self.frames_perdido < self.FRAMES_OLVIDO:
                # El jugador pudo quedar tapado un momento: no cambiar de persona
                self.frames_perdido += 1
                return -1
            elegido = max(candidatos, key=lambda c: (c[1], c[2]))

        self.centro_prev = (elegido[3], elegido[4])
        self.frames_perdido = 0

        m = self.MARGEN_BOX
        bj = elegido[5]
        box_j = (bj[0] - m, bj[1] - m, bj[2] + m, bj[3] + m)
        for p in personas:
            if p[0] != elegido[0] and self._solape(box_j, p[5]) > self.SOLAPE_MAX:
                self.interferencia = True
                break
        return elegido[0]

    def _perdido(self):
        self.frames_perdido += 1
        if self.frames_perdido >= self.FRAMES_OLVIDO:
            self.centro_prev = None


def obtener_keypoints(objs: np.ndarray, pks: np.ndarray, obj_idx: int) -> Dict[int, Punto]:
    """
    Devuelve {indice_parte: Punto(x_norm, y_norm)} solo para las partes
    detectadas de la persona obj_idx. Los peaks de trt_pose vienen como
    (fila, columna) = (y, x) normalizado.
    """
    kp: Dict[int, Punto] = {}
    if obj_idx < 0:
        return kp
    obj = objs[obj_idx]
    for j in range(obj.shape[0]):
        k = int(obj[j])
        if k >= 0:
            kp[j] = Punto(float(pks[j][k][1]), float(pks[j][k][0]))
    return kp


def cuerpo_completo(kp: Dict[int, Punto]) -> bool:
    """True si se ven la cabeza y al menos un tobillo, sin tocar los bordes."""
    if not any(c in kp for c in _CABEZA):
        return False
    if 15 not in kp and 16 not in kp:
        return False
    for p in kp.values():
        if p.y < 0.02 or p.y > 0.98:
            return False
    return True


# ==============================================================
# FOTO DEL IMPACTO - recorte del jugador guardado como JPEG
# ==============================================================

FOTO_W, FOTO_H = 300, 400           # recorte vertical 3:4
FOTO_CALIDAD_JPG = 88


def capturar_foto(frame: np.ndarray, kp: Dict[int, "Punto"]):
    """
    Recorta al jugador (cuerpo completo + margen) en formato 3:4 y lo
    devuelve como JPEG (bytes), junto con sus keypoints en pixeles DEL
    RECORTE. Se llama una vez por golpe: el costo es ~2-3 ms.
    """
    if frame is None or not kp:
        return None, {}
    H, W = frame.shape[:2]
    xs = [p.x * W for p in kp.values()]
    ys = [p.y * H for p in kp.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    aspecto = FOTO_W / float(FOTO_H)
    alto = max((y1 - y0) * 1.24, (x1 - x0) * 1.40 / aspecto, 120.0)
    ancho = alto * aspecto
    cy -= alto * 0.03                    # un poco mas de aire sobre la cabeza
    rx0, ry0 = int(round(cx - ancho / 2)), int(round(cy - alto / 2))
    rw, rh = int(round(ancho)), int(round(alto))
    # si el recorte cabe en la imagen, se desliza adentro (sin bandas negras)
    if rw <= W:
        rx0 = min(max(rx0, 0), W - rw)
    if rh <= H:
        ry0 = min(max(ry0, 0), H - rh)

    lienzo = np.full((rh, rw, 3), 28, dtype=np.uint8)
    sx0, sy0 = max(rx0, 0), max(ry0, 0)
    sx1, sy1 = min(rx0 + rw, W), min(ry0 + rh, H)
    if sx1 > sx0 and sy1 > sy0:
        lienzo[sy0 - ry0:sy1 - ry0, sx0 - rx0:sx1 - rx0] = frame[sy0:sy1, sx0:sx1]
    foto = cv2.resize(lienzo, (FOTO_W, FOTO_H), interpolation=cv2.INTER_AREA)

    ex, ey = FOTO_W / float(rw), FOTO_H / float(rh)
    kpf = {i: ((p.x * W - rx0) * ex, (p.y * H - ry0) * ey) for i, p in kp.items()}
    ok, jpg = cv2.imencode(".jpg", foto, [int(cv2.IMWRITE_JPEG_QUALITY), FOTO_CALIDAD_JPG])
    return (jpg.tobytes() if ok else None), kpf


def foto_de(golpe: "Golpe") -> Optional[np.ndarray]:
    if golpe.foto is None:
        return None
    return cv2.imdecode(np.frombuffer(golpe.foto, np.uint8), cv2.IMREAD_COLOR)


# ==============================================================
# DETECTOR DE GOLPE - evalua la postura SOLO en el impacto
# ==============================================================

@dataclass
class Golpe:
    n:             int
    resultado:     "Resultado"       # postura en el frame de maxima velocidad
    vel_pico:      float             # largos de brazo superior por frame
    rango_rodilla: Optional[float]
    frames:        int
    foto:          Optional[bytes] = None                 # JPEG del impacto
    kp_foto:       Dict[int, Tuple[float, float]] = field(default_factory=dict)


class DetectorGolpe:
    """
    Maquina de estados sobre la velocidad de la muneca, medida en
    "largos de brazo superior por frame" (asi no depende de la distancia
    a la camara ni confunde caminar con golpear).

      reposo --(vel > VEL_INICIO)--> golpe --(vel < VEL_FIN x FRAMES_FIN)--> fin
    Al terminar, el golpe es valido si la muneca recorrio al menos
    MIN_RECORRIDO largos de brazo. El frame de maxima velocidad se toma
    como el impacto (a ~12 FPS es una aproximacion de +/- 1 frame), igual
    que el criterio de velocidad de muneca usado por Lyu et al. (2025).
    v5: guarda una copia del frame LIMPIO de maxima velocidad (solo cuando
    aparece un nuevo maximo, 2-5 copias por golpe) para la foto del resumen.
    """
    VEL_INICIO    = 0.35
    VEL_FIN       = 0.20
    FRAMES_FIN    = 2
    MAX_FRAMES    = 24
    MIN_RECORRIDO = 1.5
    MAX_SIN_BRAZO = 3

    def __init__(self):
        self.n = 0
        self.reiniciar()

    @property
    def activo(self) -> bool:
        return self._activo

    def reiniciar(self):
        self._prev = None
        self._activo = False
        self._buf = []
        self._quietos = 0
        self._recorrido = 0.0
        self._sin_brazo = 0
        self._d_max = -1.0
        self._frame_max = None

    def frame_sin_brazo(self):
        """Frame con el brazo oculto: no corta el golpe de inmediato."""
        self._prev = None
        if self._activo:
            self._sin_brazo += 1
            if self._sin_brazo > self.MAX_SIN_BRAZO:
                self.reiniciar()

    def _nuevo_maximo(self, d: float, frame):
        if d > self._d_max:
            self._d_max = d
            self._frame_max = frame.copy() if frame is not None else None

    def actualizar(self, muneca_px: np.ndarray, largo_sup: float,
                   res: "Resultado", frame: Optional[np.ndarray] = None) -> Optional[Golpe]:
        """frame debe estar LIMPIO (antes de dibujar el HUD encima)."""
        if self._prev is None:
            self._prev = muneca_px
            return None
        d = float(np.linalg.norm(muneca_px - self._prev)) / max(largo_sup, 1e-6)
        self._prev = muneca_px
        self._sin_brazo = 0

        if not self._activo:
            if d > self.VEL_INICIO:
                self._activo = True
                self._buf = [(d, res)]
                self._recorrido = d
                self._quietos = 0
                self._nuevo_maximo(d, frame)
            return None

        self._buf.append((d, res))
        self._recorrido += d
        self._nuevo_maximo(d, frame)
        self._quietos = self._quietos + 1 if d < self.VEL_FIN else 0
        if self._quietos >= self.FRAMES_FIN or len(self._buf) >= self.MAX_FRAMES:
            return self._cerrar()
        return None

    def _cerrar(self) -> Optional[Golpe]:
        buf, recorrido, frame_imp = self._buf, self._recorrido, self._frame_max
        prev = self._prev
        self.reiniciar()
        self._prev = prev
        if recorrido < self.MIN_RECORRIDO or len(buf) < 2:
            return None

        d_pico, res_imp = max(buf, key=lambda b: b[0])

        # Rango de rodilla durante el golpe, con mediana de 3 para quitar ruido
        rod = [b[1].angulos["rodilla"] for b in buf if "rodilla" in b[1].angulos]
        rango = None
        if len(rod) >= 3:
            suav = [sorted(rod[i - 1:i + 2])[1] for i in range(1, len(rod) - 1)]
            rango = max(suav) - min(suav) if len(suav) >= 2 else None

        foto, kpf = capturar_foto(frame_imp, res_imp.kp)
        self.n += 1
        return Golpe(self.n, res_imp, d_pico, rango, len(buf), foto, kpf)


# ==============================================================
# ANALISIS DE POSTURA
# ==============================================================

@dataclass
class Resultado:
    angulos:      Dict[str, float] = field(default_factory=dict)
    errores:      List[str]        = field(default_factory=list)
    advertencias: List[str]        = field(default_factory=list)
    avisos:       List[str]        = field(default_factory=list)  # de vision
    brazo_ok:     bool             = True
    pierna_ok:    bool             = True
    # (codigo, severidad): 2 = error, 1 = advertencia. Se usan para el
    # consejo principal en pantalla y para el resumen de la sesion.
    codigos:      List[Tuple[str, int]] = field(default_factory=list)
    kp:           Dict[int, "Punto"]    = field(default_factory=dict)

    @property
    def es_correcto(self) -> bool:
        return len(self.errores) == 0

    @property
    def es_optimo(self) -> bool:
        return len(self.errores) == 0 and len(self.advertencias) == 0


def _brazo_confiable(kp, idx, hombro, codo, muneca, a_codo, reglas, W, H):
    """Retorna (True, "") si la lectura del brazo es creible, o (False, motivo)."""
    largo_sup = float(np.linalg.norm(hombro - codo))
    largo_ant = float(np.linalg.norm(codo - muneca))

    if largo_sup < reglas.brazo_sup_min_px:
        return False, "Jugador muy lejos de la camara"

    ratio = largo_ant / largo_sup
    if ratio < reglas.antebrazo_ratio_min or ratio > reglas.antebrazo_ratio_max:
        return False, "Antebrazo tapado o mal detectado"

    if a_codo < reglas.codo_min_fisico:
        return False, "Lectura de codo imposible (brazo tapado por el cuerpo)"

    for c in _CABEZA:
        if c in kp:
            d = float(np.linalg.norm(muneca - pt_arr(kp[c], W, H)))
            if d < reglas.muneca_cabeza_min * largo_sup:
                return False, "Muneca confundida con la cabeza"

    return True, ""


def analizar(kp: Dict[int, Punto], mano: str, reglas: Reglas,
             W: int, H: int) -> Optional[Resultado]:
    """
    Retorna None solo si no se ve el hombro del lado activo (no hay
    jugador util). Si el brazo o la pierna no se ven bien, lo marca
    en el Resultado en vez de inventar un angulo.
    """
    idx = _IDX[mano]
    if idx["hombro"] not in kp:
        return None

    res = Resultado()
    res.kp = kp
    hombro = pt_arr(kp[idx["hombro"]], W, H)

    # 1. BRAZO: codo + caida de muneca
    if idx["codo"] in kp and idx["muneca"] in kp:
        codo = pt_arr(kp[idx["codo"]], W, H)
        muneca = pt_arr(kp[idx["muneca"]], W, H)
        a_codo = angulo(hombro, codo, muneca)
        ok, motivo = _brazo_confiable(kp, idx, hombro, codo, muneca,
                                      a_codo, reglas, W, H)
        if ok:
            res.angulos["codo"] = a_codo
            if a_codo < reglas.codo_warn_min:
                res.errores.append("Codo muy cerrado (%.0f) - abre el brazo" % a_codo)
                res.codigos.append(("codo_cerrado", 2))
            elif a_codo > reglas.codo_warn_max:
                res.errores.append("Brazo casi recto (%.0f) - dobla el codo" % a_codo)
                res.codigos.append(("codo_recto", 2))
            elif a_codo < reglas.codo_ideal_min:
                res.advertencias.append("Codo %.0f - abrelo un poco mas" % a_codo)
                res.codigos.append(("codo_abrir", 1))
            elif a_codo > reglas.codo_ideal_max:
                res.advertencias.append("Codo %.0f - cierralo un poco al impactar" % a_codo)
                res.codigos.append(("codo_cerrar", 1))

            largo_sup = float(np.linalg.norm(hombro - codo))
            caida = (muneca[1] - codo[1]) / largo_sup
            res.angulos["caida_muneca_%"] = caida * 100.0
            if caida > reglas.muneca_caida_error:
                res.errores.append("Muneca caida - sube el brazo al golpear")
                res.codigos.append(("muneca_caida", 2))
            elif caida > reglas.muneca_caida_warn:
                res.advertencias.append("Muneca ligeramente baja")
                res.codigos.append(("muneca_baja", 1))
        else:
            res.brazo_ok = False
            res.avisos.append(motivo)
    else:
        res.brazo_ok = False
        res.avisos.append("No se ve la muneca del brazo que golpea")

    # 2. RODILLA DOMINANTE
    if all(idx[n] in kp for n in ("cadera", "rodilla", "tobillo")):
        a_rod = angulo(pt_arr(kp[idx["cadera"]], W, H),
                       pt_arr(kp[idx["rodilla"]], W, H),
                       pt_arr(kp[idx["tobillo"]], W, H))
        res.angulos["rodilla"] = a_rod
        if a_rod > reglas.rodilla_error_max:
            res.errores.append("Piernas rectas (%.0f) - dobla las rodillas" % a_rod)
            res.codigos.append(("piernas_rectas", 2))
        elif a_rod < reglas.rodilla_error_min:
            res.errores.append("Rodilla muy flexionada (%.0f) - sube un poco" % a_rod)
            res.codigos.append(("rodilla_baja", 2))
        elif not (reglas.rodilla_ideal_min <= a_rod <= reglas.rodilla_ideal_max):
            res.advertencias.append("Rodilla %.0f - busca %.0f a %.0f" % (
                a_rod, reglas.rodilla_ideal_min, reglas.rodilla_ideal_max))
            res.codigos.append(("rodilla_fuera", 1))
    else:
        res.pierna_ok = False
        res.avisos.append("No se ve la pierna completa (incluye los pies)")

    # Rodilla de apoyo (opcional)
    ids_opp = (idx["cadera_opp"], idx["rodilla_opp"], idx["tobillo_opp"])
    if all(i in kp for i in ids_opp):
        a_rod_opp = angulo(*(pt_arr(kp[i], W, H) for i in ids_opp))
        res.angulos["rodilla_opp"] = a_rod_opp
        if a_rod_opp > reglas.rodilla_error_max:
            res.advertencias.append("Dobla tambien la rodilla de apoyo")
            res.codigos.append(("rodilla_apoyo", 1))

    return res


def evaluar_estabilidad_rodilla(res: Resultado, rango: Optional[float], reglas: Reglas):
    """Solo informativa: se evalua sobre el golpe completo, no frame a frame."""
    if rango is None:
        return
    res.angulos["rango_rodilla"] = rango
    if rango > reglas.rodilla_rango_warn:
        res.advertencias.append("Rodilla se mueve %.0f en el golpe - estabilizala" % rango)
        res.codigos.append(("rodilla_rango", 1))


# ==============================================================
# CALIBRACION
# ==============================================================

@dataclass
class MuestraCalibracion:
    n:                int
    timestamp:        str
    origen:           str      # "auto" (golpe detectado) o "manual" (SPACE)
    mano:             str
    codo:             float
    caida_muneca_pct: float
    rodilla:          float
    rodilla_apoyo:    float
    rango_rodilla:    float
    vel_pico:         float


class Calibrador:
    # v3: agrega columnas origen y vel_pico. Cada golpe detectado se
    # guarda solo (origen=auto); SPACE sigue guardando manualmente.
    RUTA_CSV = "calibracion_forehand_jetson_v3.csv"

    def __init__(self):
        self.muestras: List[MuestraCalibracion] = []
        nuevo = not os.path.exists(self.RUTA_CSV)
        self.csv_previo = not nuevo
        self._f = open(self.RUTA_CSV, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._f)
        if nuevo:
            self._writer.writerow(
                ["n", "timestamp", "origen", "mano", "codo_deg",
                 "caida_muneca_pct_brazo", "rodilla_deg", "rodilla_apoyo_deg",
                 "rango_rodilla_deg", "vel_pico_brazos_frame"])
            self._f.flush()

    def capturar(self, resultado: "Resultado", mano: str, origen: str,
                 vel_pico: float = float("nan")) -> MuestraCalibracion:
        nan = float("nan")
        m = MuestraCalibracion(
            n=len(self.muestras) + 1,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            origen=origen,
            mano=mano,
            codo=resultado.angulos.get("codo", nan),
            caida_muneca_pct=resultado.angulos.get("caida_muneca_%", nan),
            rodilla=resultado.angulos.get("rodilla", nan),
            rodilla_apoyo=resultado.angulos.get("rodilla_opp", nan),
            rango_rodilla=resultado.angulos.get("rango_rodilla", nan),
            vel_pico=vel_pico,
        )
        self.muestras.append(m)
        self._writer.writerow([
            m.n, m.timestamp, m.origen, m.mano,
            "%.1f" % m.codo, "%.1f" % m.caida_muneca_pct,
            "%.1f" % m.rodilla, "%.1f" % m.rodilla_apoyo,
            "%.1f" % m.rango_rodilla, "%.2f" % m.vel_pico,
        ])
        self._f.flush()
        return m

    def deshacer_ultima(self) -> Optional[MuestraCalibracion]:
        if not self.muestras:
            return None
        eliminada = self.muestras.pop()
        self._f.close()
        with open(self.RUTA_CSV, "r", encoding="utf-8") as f:
            filas = f.readlines()
        if len(filas) > 1:
            filas = filas[:-1]
        with open(self.RUTA_CSV, "w", newline="", encoding="utf-8") as f:
            f.writelines(filas)
        self._f = open(self.RUTA_CSV, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._f)
        return eliminada

    def resumen(self) -> str:
        if not self.muestras:
            return "\nNo se registro ningun golpe durante la sesion."

        def stats(vals: List[float]) -> str:
            vals = [v for v in vals if v == v]
            if not vals:
                return "sin datos"
            txt = "min %.0f | max %.0f | media %.0f" % (min(vals), max(vals), _st.mean(vals))
            if len(vals) > 1:
                txt += " | desv.est %.1f" % _st.stdev(vals)
            return txt

        ms = self.muestras
        n_auto = sum(1 for m in ms if m.origen == "auto")
        out = ["\n" + "=" * 58, "  RESUMEN DE CALIBRACION (Jetson / trt_pose v5)", "=" * 58]
        out.append("  Muestras: %d  (golpes auto: %d | manuales: %d)"
                   % (len(ms), n_auto, len(ms) - n_auto))
        out.append("  Codo (grados):          " + stats([m.codo for m in ms]))
        out.append("  Rodilla (grados):       " + stats([m.rodilla for m in ms]))
        out.append("  Caida muneca (% brazo): " + stats([m.caida_muneca_pct for m in ms]))
        out.append("  Rango rodilla (grados): " + stats([m.rango_rodilla for m in ms]))
        out.append("  Vel. pico (brazos/fr):  " + stats([m.vel_pico for m in ms]))
        out.append("-" * 58)
        out.append("  Sugerencia: usa min/max reales +/- margen de 5-10")
        out.append("  para los rangos ideales en la clase Reglas.")
        out.append("  Datos completos en: " + os.path.abspath(self.RUTA_CSV))
        out.append("=" * 58)
        return "\n".join(out)

    def cerrar(self):
        self._f.close()


# ==============================================================
# COLORES (BGR) Y AVISOS PERSISTENTES
# ==============================================================

VERDE    = (55,  215, 75)
AMARILLO = (35,  205, 255)
ROJO     = (55,  55,  225)
GRIS     = (145, 145, 145)
AZUL     = (255, 190, 80)
OSCURO   = (12,  12,  12)


class AvisoPersistente:
    """
    Un aviso de vision solo se muestra si su condicion se mantiene
    `seg` segundos SEGUIDOS. Asi un brazo tapado un instante en la
    terminacion del golpe (algo normal) no hace parpadear la pantalla.
    """
    def __init__(self, seg: float):
        self.seg = seg
        self._desde: Optional[float] = None

    def actualizar(self, condicion: bool, ahora: float) -> bool:
        if not condicion:
            self._desde = None
            return False
        if self._desde is None:
            self._desde = ahora
        return ahora - self._desde >= self.seg

    def reiniciar(self):
        self._desde = None


# ==============================================================
# INTERFAZ v4 - pantallas de INICIO, SESION y RESUMEN
# Todo se dibuja con OpenCV (sin librerias extra) y texto ASCII.
# ==============================================================

FUENTE   = cv2.FONT_HERSHEY_SIMPLEX
FUENTE_T = cv2.FONT_HERSHEY_DUPLEX
BLANCO   = (240, 240, 240)
TENUE    = (150, 150, 150)
FONDO    = (22, 20, 18)

CLASES = {
    "ok":       ("OK",   VERDE),
    "casi":     ("CASI", AMARILLO),
    "corregir": ("!",    ROJO),
}

CONSEJOS = {
    "codo_recto":     "Dobla mas el codo al impactar",
    "codo_cerrado":   "Abre el brazo: el codo esta muy cerrado",
    "codo_cerrar":    "Cierra un poco el codo al impactar",
    "codo_abrir":     "Abre un poco el codo",
    "muneca_caida":   "Sube la muneca: que no caiga bajo el codo",
    "muneca_baja":    "Muneca un poco baja al impactar",
    "piernas_rectas": "Flexiona las rodillas",
    "rodilla_baja":   "Estas muy agachado: sube un poco",
    "rodilla_fuera":  "Ajusta la flexion de la rodilla",
    "rodilla_rango":  "Manten la rodilla estable al golpear",
    "rodilla_apoyo":  "Dobla tambien la rodilla de apoyo",
}

# Codigos que colorean el brazo o la pierna en el "fantasma" del impacto
_COD_BRAZO = ("codo_recto", "codo_cerrado", "codo_cerrar", "codo_abrir",
              "muneca_caida", "muneca_baja")
_COD_PIERNA = ("piernas_rectas", "rodilla_baja", "rodilla_fuera", "rodilla_rango")


def clase_de(res: Resultado) -> str:
    if res.es_optimo:
        return "ok"
    if res.es_correcto:
        return "casi"
    return "corregir"


def consejo_principal(res: Resultado) -> Optional[str]:
    """El consejo mas importante del golpe: primero errores, luego avisos."""
    if not res.codigos:
        return None
    cod = sorted(res.codigos, key=lambda c: -c[1])[0][0]
    return CONSEJOS.get(cod)


def color_segmento(res: Resultado, grupo) -> Tuple[int, int, int]:
    sev = max([s for c, s in res.codigos if c in grupo] or [0])
    return ROJO if sev == 2 else (AMARILLO if sev == 1 else VERDE)


# ---------- primitivas graficas ----------

def mezclar_rect(img, x1, y1, x2, y2, color, alpha):
    """Rectangulo semitransparente (solo mezcla la zona, es rapido)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    if color == (0, 0, 0):                      # oscurecer: una sola operacion
        cv2.convertScaleAbs(roi, roi, 1.0 - alpha)
        return
    capa = np.empty_like(roi)
    capa[:] = color
    cv2.addWeighted(capa, alpha, roi, 1.0 - alpha, 0, roi)


def texto_centrado(img, txt, cy, escala, color, grosor=1, fuente=FUENTE, cx=None):
    t = texto_ascii(txt)
    (w, h), _ = cv2.getTextSize(t, fuente, escala, grosor)
    if cx is None:
        cx = img.shape[1] // 2
    cv2.putText(img, t, (int(cx - w / 2), int(cy + h / 2)), fuente, escala,
                color, grosor, cv2.LINE_AA)


def pastilla(img, x, y, txt, color_txt, escala=0.5, grosor=1, alpha=0.55,
             derecha=False):
    """Etiqueta con fondo semitransparente. Retorna el ancho usado."""
    t = texto_ascii(txt)
    (w, h), _ = cv2.getTextSize(t, FUENTE, escala, grosor)
    ancho = w + 20
    if derecha:
        x = x - ancho
    mezclar_rect(img, x, y, x + ancho, y + h + 16, FONDO, alpha)
    cv2.putText(img, t, (x + 10, y + h + 8), FUENTE, escala, color_txt, grosor, cv2.LINE_AA)
    return ancho


def anillo_progreso(img, centro, radio, progreso, color, grosor=4):
    cv2.circle(img, centro, radio, (60, 60, 60), grosor, cv2.LINE_AA)
    if progreso > 0:
        cv2.ellipse(img, centro, (radio, radio), -90, 0, int(360 * min(progreso, 1.0)),
                    color, grosor, cv2.LINE_AA)


def cuna_angulo(img, vertice, p1, p2, radio, color, alpha=0.45):
    """Sector relleno entre los segmentos vertice-p1 y vertice-p2."""
    a1 = np.degrees(np.arctan2(p1[1] - vertice[1], p1[0] - vertice[0]))
    a2 = np.degrees(np.arctan2(p2[1] - vertice[1], p2[0] - vertice[0]))
    d = (a2 - a1) % 360.0
    if d > 180.0:
        a1, d = a2, 360.0 - d
    x, y = int(vertice[0]), int(vertice[1])
    h, w = img.shape[:2]
    x1, y1 = max(0, x - radio - 2), max(0, y - radio - 2)
    x2, y2 = min(w, x + radio + 2), min(h, y + radio + 2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    capa = roi.copy()
    cv2.ellipse(capa, (x - x1, y - y1), (radio, radio), 0, a1, a1 + d, color, -1, cv2.LINE_AA)
    cv2.addWeighted(capa, alpha, roi, 1.0 - alpha, 0, roi)
    cv2.ellipse(img, (x, y), (radio, radio), 0, a1, a1 + d, color, 2, cv2.LINE_AA)


def dibujar_esqueleto_suave(frame, kp, W, H, color=(90, 200, 90), grosor=2):
    for a, b in _CONEXIONES:
        if a in kp and b in kp:
            cv2.line(frame, pt_px(kp[a], W, H), pt_px(kp[b], W, H), color, grosor, cv2.LINE_AA)
    for p in kp.values():
        cv2.circle(frame, pt_px(p, W, H), 3, BLANCO, -1, cv2.LINE_AA)


def dibujar_fantasma_impacto(frame, golpe: Golpe, mano: str, W, H, vida: float):
    """
    Dibuja la pose del IMPACTO congelada sobre el video, con el brazo y
    la pierna coloreados segun la evaluacion, y la cuna del angulo del codo.
    vida: 1.0 recien ocurrido -> 0.0 a punto de desaparecer.
    """
    res, kp = golpe.resultado, golpe.resultado.kp
    if not kp:
        return
    idx = _IDX[mano]
    c_brazo = color_segmento(res, _COD_BRAZO)
    c_pierna = color_segmento(res, _COD_PIERNA)
    grosor = 3 + int(3 * vida)

    for a, b in (("hombro", "codo"), ("codo", "muneca")):
        if idx[a] in kp and idx[b] in kp:
            cv2.line(frame, pt_px(kp[idx[a]], W, H), pt_px(kp[idx[b]], W, H),
                     c_brazo, grosor, cv2.LINE_AA)
    for a, b in (("cadera", "rodilla"), ("rodilla", "tobillo")):
        if idx[a] in kp and idx[b] in kp:
            cv2.line(frame, pt_px(kp[idx[a]], W, H), pt_px(kp[idx[b]], W, H),
                     c_pierna, grosor, cv2.LINE_AA)

    if "codo" in res.angulos and all(idx[n] in kp for n in ("hombro", "codo", "muneca")):
        v = pt_px(kp[idx["codo"]], W, H)
        cuna_angulo(frame, v, pt_px(kp[idx["hombro"]], W, H), pt_px(kp[idx["muneca"]], W, H),
                    26, c_brazo)
        pastilla(frame, v[0] + 16, v[1] - 34, "%.0f" % res.angulos["codo"], c_brazo, 0.6, 2)
    if "rodilla" in res.angulos and idx["rodilla"] in kp:
        v = pt_px(kp[idx["rodilla"]], W, H)
        pastilla(frame, v[0] + 14, v[1] - 12, "%.0f" % res.angulos["rodilla"], c_pierna, 0.55, 2)


# ---------- gesto: mano libre sobre la cabeza ----------

class GestoMano:
    """
    Detecta la mano LIBRE (la que no golpea) levantada por encima de la
    cabeza durante un tiempo. Sirve para empezar y terminar la sesion sin
    ir hasta el teclado. Despues de disparar, exige bajar la mano antes
    de poder disparar otra vez.
    """
    def __init__(self):
        self.frames = 0
        self.bloqueado = False

    @staticmethod
    def mano_arriba(kp: Dict[int, Punto], mano: str) -> Optional[Punto]:
        libre = 9 if mano == "derecha" else 10       # muneca contraria
        if libre not in kp:
            return None
        cabeza = [kp[c].y for c in _CABEZA if c in kp]
        if not cabeza and 17 in kp:
            cabeza = [kp[17].y - 0.05]
        if not cabeza:
            return None
        return kp[libre] if kp[libre].y < min(cabeza) else None

    def actualizar(self, arriba: bool, necesarios: int) -> Tuple[float, bool]:
        if not arriba:
            self.frames = 0
            self.bloqueado = False
            return 0.0, False
        if self.bloqueado:
            return 0.0, False
        self.frames += 1
        if self.frames >= necesarios:
            self.frames = 0
            self.bloqueado = True
            return 1.0, True
        return self.frames / float(necesarios), False


# ---------- sesion ----------

class Sesion:
    def __init__(self, mano: str):
        self.mano = mano
        self.golpes: List[Golpe] = []
        self.t_inicio = time.time()
        self.t_fin: Optional[float] = None

    def duracion(self) -> float:
        return (self.t_fin or time.time()) - self.t_inicio

    def clases(self) -> List[str]:
        return [clase_de(g.resultado) for g in self.golpes]

    def mejor_racha(self) -> int:
        mejor = act = 0
        for c in self.clases():
            act = act + 1 if c == "ok" else 0
            mejor = max(mejor, act)
        return mejor

    def enfoque(self) -> Optional[Tuple[str, int]]:
        """Correccion mas repetida en la sesion (ponderando errores x2)."""
        cuenta: Dict[str, int] = {}
        veces: Dict[str, int] = {}
        for g in self.golpes:
            for cod, sev in set(g.resultado.codigos):
                cuenta[cod] = cuenta.get(cod, 0) + sev
                veces[cod] = veces.get(cod, 0) + 1
        if not cuenta:
            return None
        cod = max(cuenta, key=lambda c: cuenta[c])
        return cod, veces[cod]


def mmss(seg: float) -> str:
    seg = int(seg)
    return "%02d:%02d" % (seg // 60, seg % 60)


# ---------- pantalla de INICIO ----------

def pantalla_inicio(frame, kp, mano, checks, progreso_gesto, fps):
    H, W = frame.shape[:2]
    mezclar_rect(frame, 0, 0, W, H, (0, 0, 0), 0.45)
    if kp:
        dibujar_esqueleto_suave(frame, kp, W, H)

    mezclar_rect(frame, 0, 0, W, 78, FONDO, 0.75)
    texto_centrado(frame, "ENTRENADOR DE DERECHA", 28, 0.85, BLANCO, 2, FUENTE_T)
    texto_centrado(frame, "Tenis de mesa  |  modo %s  (D/Z para cambiar)"
                   % ("DIESTRO" if mano == "derecha" else "ZURDO"), 60, 0.45, TENUE)

    # Lista de verificacion
    x, y = 24, 110
    mezclar_rect(frame, x - 12, y - 22, x + 300, y + len(checks) * 30 - 4, FONDO, 0.65)
    for etiqueta, ok in checks:
        cv2.circle(frame, (x, y - 5), 8, VERDE if ok else (70, 70, 70), -1, cv2.LINE_AA)
        poner_texto(frame, etiqueta, (x + 18, y), 0.5, BLANCO if ok else TENUE)
        y += 30

    listo = all(ok for _, ok in checks)
    texto_centrado(frame, "Levanta la mano libre sobre la cabeza para empezar",
                   H - 70, 0.55, BLANCO if listo else TENUE, 1)
    texto_centrado(frame, "(o presiona ENTER)", H - 45, 0.45, TENUE)
    if progreso_gesto > 0:
        anillo_progreso(frame, (W // 2, H // 2), 46, progreso_gesto, AZUL, 6)
    poner_texto(frame, "FPS %.0f" % fps, (W - 70, H - 12), 0.4, (90, 90, 90))


def pantalla_cuenta(frame, restante: float):
    H, W = frame.shape[:2]
    mezclar_rect(frame, 0, 0, W, H, (0, 0, 0), 0.35)
    n = int(restante) + 1
    frac = restante - int(restante)
    anillo_progreso(frame, (W // 2, H // 2), 70, frac, AZUL, 8)
    texto_centrado(frame, str(n), H // 2, 2.6, BLANCO, 5, FUENTE_T)
    texto_centrado(frame, "Preparate", H // 2 + 110, 0.8, TENUE, 2)


# ---------- HUD durante la SESION ----------

def hud_sesion(frame, sesion: Sesion, ultimo: Optional[Golpe], t_ultimo: float,
               golpeando: bool, avisos: List[str], progreso_fin: float,
               fps: float, pausado: bool):
    H, W = frame.shape[:2]
    ahora = time.time()

    # Contador y tiempo (arriba izquierda)
    mezclar_rect(frame, 12, 12, 150, 78, FONDO, 0.6)
    poner_texto(frame, "GOLPES", (24, 32), 0.42, TENUE)
    cv2.putText(frame, str(len(sesion.golpes)), (24, 68), FUENTE_T, 1.2, BLANCO, 2, cv2.LINE_AA)
    poner_texto(frame, mmss(sesion.duracion()), (90, 68), 0.5, TENUE)

    # Estado (arriba derecha)
    if pausado:
        pastilla(frame, W - 12, 12, "PAUSA", (80, 180, 255), 0.55, 2, derecha=True)
    elif golpeando:
        pastilla(frame, W - 12, 12, "GOLPEANDO", (80, 200, 255), 0.55, 2, derecha=True)

    # Insignia del ultimo golpe: aparece con "pop" y se desvanece en 3 s
    edad = ahora - t_ultimo
    if ultimo is not None and edad < 3.0:
        txt, col = CLASES[clase_de(ultimo.resultado)]
        radio = int(34 + 10 * max(0.0, 1.0 - edad / 0.25))
        cx, cy = W - 58, 100
        mezclar_rect(frame, cx - radio - 4, cy - radio - 4, cx + radio + 4, cy + radio + 4,
                     (0, 0, 0), 0.25)
        cv2.circle(frame, (cx, cy), radio, col, -1, cv2.LINE_AA)
        texto_centrado(frame, txt, cy, 0.9 if len(txt) > 1 else 1.3, (20, 20, 20), 2,
                       FUENTE_T, cx=cx)
        texto_centrado(frame, "#%d" % ultimo.n, cy + radio + 16, 0.5, BLANCO, 1, cx=cx)

    # Consejo principal del ultimo golpe (barra inferior, 5 s)
    y_base = H - 16
    if ultimo is not None and edad < 5.0:
        consejo = consejo_principal(ultimo.resultado)
        if consejo is None:
            consejo, col = "Buen golpe, sigue asi", VERDE
        else:
            col = ROJO if max(s for _, s in ultimo.resultado.codigos) == 2 else AMARILLO
        mezclar_rect(frame, 0, H - 44, W, H, FONDO, 0.7)
        cv2.rectangle(frame, (0, H - 44), (6, H), col, -1)
        texto_centrado(frame, consejo, H - 22, 0.62, col, 2)
        y_base = H - 52

    # Racha de los ultimos golpes (puntos de colores)
    ult = sesion.clases()[-14:]
    for i, c in enumerate(ult):
        cv2.circle(frame, (20 + i * 18, y_base - 8), 6, CLASES[c][1], -1, cv2.LINE_AA)

    # Avisos de vision (uno a la vez, discreto, en gris). Solo llegan aqui
    # si la condicion lleva varios segundos seguidos (AvisoPersistente).
    if avisos:
        (w, _), _ = cv2.getTextSize(texto_ascii(avisos[0]), FUENTE, 0.45, 1)
        pastilla(frame, W // 2 - (w + 20) // 2, 92, avisos[0], (200, 200, 200), 0.45,
                 alpha=0.45)

    # Gesto de fin
    if progreso_fin > 0:
        anillo_progreso(frame, (W // 2, H // 2), 46, progreso_fin, AZUL, 6)
        texto_centrado(frame, "Terminando sesion...", H // 2 + 72, 0.6, BLANCO, 2)

    poner_texto(frame, "FPS %.0f" % fps, (W - 70, H - 12 if y_base == H - 16 else H - 52),
                0.4, (90, 90, 90))



# ==============================================================
# RESUMEN v5 - TARJETAS: foto real del golpe vs postura ideal
# ==============================================================
# Se construyen UNA sola vez al terminar la sesion (no cuestan FPS).
# Paleta inspirada en la mesa: azul de mesa, lineas de tiza blanca,
# rojo de la goma para lo que hay que corregir, naranja de la bola para
# lo "casi", verde para la postura ideal.

RW, RH = 960, 720                 # tamano de las tarjetas
SEG_PAGINA = 9.0                  # cada cuanto pasa sola a la siguiente

MESA   = (58, 36, 15)             # fondo  #0F243A
PANEL  = (86, 56, 29)             # paneles #1D3856
TIZA   = (246, 242, 238)          # texto principal
NIEBLA = (190, 170, 150)          # texto secundario
GOMA   = (64, 58, 226)            # rojo goma  #E23A40
BOLA   = (36, 160, 255)           # naranja bola #FFA024
IDEAL  = (130, 225, 60)           # verde ideal #3CE182

_RUTAS_TTF = {
    False: ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf", "DejaVuSans.ttf"],
    True:  ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"],
}
_FUENTES: Dict[Tuple[int, bool], object] = {}


def _fuente(tam: int, negrita: bool = False):
    k = (tam, negrita)
    if k not in _FUENTES:
        f = None
        if _HAY_PIL:
            for ruta in _RUTAS_TTF[negrita]:
                if os.path.exists(ruta):
                    try:
                        f = _PILFont.truetype(ruta, tam)
                        break
                    except Exception:
                        pass
        _FUENTES[k] = f
    return _FUENTES[k]


def _ancho_txt(txt: str, tam: int, negrita: bool = False) -> int:
    f = _fuente(tam, negrita)
    if f is None:
        (w, _), _ = cv2.getTextSize(texto_ascii(txt), FUENTE, tam / 30.0, 2 if negrita else 1)
        return w
    if hasattr(f, "getlength"):
        return int(f.getlength(txt))
    return f.getsize(txt)[0]


class Pizarra:
    """Acumula los textos de una tarjeta y los dibuja juntos con PIL
    (una sola conversion de imagen). Sin PIL/TTF cae a Hershey ASCII."""

    def __init__(self):
        self.items = []

    def texto(self, txt, x, y, tam, color, negrita=False, alinear="l") -> int:
        """(x, y) = esquina superior de la linea. Retorna el ancho."""
        w = _ancho_txt(txt, tam, negrita)
        if alinear == "c":
            x -= w // 2
        elif alinear == "r":
            x -= w
        self.items.append((txt, int(x), int(y), tam, color, negrita))
        return w

    def parrafo(self, txt, x, y, ancho, tam, color, negrita=False, interlinea=1.34) -> int:
        """Texto con salto de linea automatico. Retorna la y siguiente."""
        alto = int(round(tam * interlinea))
        linea = ""
        for p in txt.split():
            prueba = (linea + " " + p).strip()
            if linea and _ancho_txt(prueba, tam, negrita) > ancho:
                self.texto(linea, x, y, tam, color, negrita)
                y += alto
                linea = p
            else:
                linea = prueba
        if linea:
            self.texto(linea, x, y, tam, color, negrita)
            y += alto
        return y

    def dibujar(self, img):
        if not self.items:
            return img
        if _fuente(16) is None:
            for txt, x, y, tam, color, neg in self.items:
                cv2.putText(img, texto_ascii(txt), (x, y + int(tam * 0.85)), FUENTE,
                            tam / 30.0, color, 2 if neg else 1, cv2.LINE_AA)
            return img
        pil = _PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dr = _PILDraw.Draw(pil)
        for txt, x, y, tam, color, neg in self.items:
            dr.text((x, y), txt, font=_fuente(tam, neg), fill=(color[2], color[1], color[0]))
        img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return img


# ---------- textos de cada correccion (lenguaje de entrenador) ----------

TARJETAS = {
    "piernas_rectas": dict(
        parte="Rodillas", titulo="Flexiona las rodillas",
        paso="Llegaste a la bola con las piernas casi rectas.",
        como="Baja la cadera como si fueras a sentarte en una silla alta, "
             "con el peso en la parte delantera de los pies.",
        porque="Con las rodillas dobladas empujas desde el suelo: más potencia "
               "y llegas antes a la bola."),
    "rodilla_baja": dict(
        parte="Rodillas", titulo="Sube un poco la cadera",
        paso="Golpeaste demasiado agachado.",
        como="Sube la cadera unos centímetros, sin llegar a estirar las piernas.",
        porque="Muy abajo pierdes movilidad y te cuesta volver a la posición."),
    "rodilla_fuera": dict(
        parte="Rodillas", titulo="Ajusta la flexión de rodillas",
        paso="La flexión de la rodilla quedó fuera de la zona ideal.",
        como="Busca una flexión media y cómoda, la misma en cada golpe.",
        porque="Una altura constante hace tu golpe más repetible."),
    "rodilla_rango": dict(
        parte="Altura", titulo="Golpea sin levantarte",
        paso="Tu rodilla se estiró o se dobló mucho durante el golpe: "
             "cambiaste de altura.",
        como="Mantén la cadera a la misma altura de principio a fin. "
             "La línea punteada marca la altura que debes conservar.",
        porque="Si subes al golpear cambia la trayectoria de la paleta y pierdes control."),
    "rodilla_apoyo": dict(
        parte="Apoyo", titulo="Dobla también la otra rodilla",
        paso="La pierna de apoyo quedó recta.",
        como="Reparte el peso entre los dos pies, con las dos rodillas flexionadas.",
        porque="Dos piernas activas dan equilibrio y pasan el peso al golpe."),
    "codo_recto": dict(
        parte="Codo", titulo="Dobla el codo",
        paso="Golpeaste con el brazo casi estirado.",
        como="Mantén el codo doblado, como una V abierta, con la paleta "
             "delante del cuerpo.",
        porque="Con el brazo recto pierdes control y cargas el hombro."),
    "codo_cerrado": dict(
        parte="Codo", titulo="Abre el brazo",
        paso="El codo estaba muy cerrado: la paleta quedó pegada al cuerpo.",
        como="Separa el antebrazo del cuerpo, como si sostuvieras un vaso lejos de ti.",
        porque="Con el brazo encogido no hay espacio para acelerar la paleta."),
    "codo_cerrar": dict(
        parte="Codo", titulo="Dobla un poco más el codo",
        paso="El brazo estaba algo estirado al tocar la bola.",
        como="Al llegar a la bola, dobla un poco más el codo.",
        porque="Un codo flexionado da un golpe más compacto y controlado."),
    "codo_abrir": dict(
        parte="Codo", titulo="Abre un poco el codo",
        paso="El codo estaba algo cerrado al tocar la bola.",
        como="Separa un poco el antebrazo del cuerpo.",
        porque="Más espacio para el brazo es más aceleración de la paleta."),
    "muneca_caida": dict(
        parte="Muñeca", titulo="Sube la muñeca",
        paso="La muñeca cayó por debajo del codo al tocar la bola.",
        como="Golpea con la muñeca firme, a la altura del codo o un poco más arriba.",
        porque="Con la muñeca caída cambia el ángulo de la paleta y la bola sale sin control."),
    "muneca_baja": dict(
        parte="Muñeca", titulo="Sube un poco la muñeca",
        paso="La muñeca quedó algo más baja que el codo.",
        como="Sube la mano hasta que la muñeca quede a la altura del codo.",
        porque="Con la muñeca caída cambia el ángulo de la paleta."),
}

_ARMS = ("codo_recto", "codo_cerrado", "codo_cerrar", "codo_abrir")
_WRIST = ("muneca_caida", "muneca_baja")
_KNEE = ("piernas_rectas", "rodilla_baja", "rodilla_fuera")
_SUPERIOR = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17)   # se mueve con la cadera


def _parte(cod: str) -> str:
    return TARJETAS.get(cod, {}).get("parte", cod)


# ---------- geometria de la postura ideal ----------

def _np(p) -> np.ndarray:
    return np.array(p, dtype=np.float64)


def _rot(v: np.ndarray, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _cruz(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _lado_frente(kpf, a: np.ndarray, p: np.ndarray) -> float:
    """Lado de la linea tobillo->cadera hacia donde mira el jugador
    (la rodilla se dobla hacia adelante). Usa nariz vs cuello."""
    if 0 in kpf and 17 in kpf:
        dx = kpf[0][0] - kpf[17][0]
        if abs(dx) > 2:
            return 1.0 if _cruz(p - a, np.array([dx, 0.0])) > 0 else -1.0
    return 1.0


def _rodilla_circulos(a, p, l_tib, l_mus, signo) -> np.ndarray:
    """Rodilla que conserva el largo de tibia y muslo entre tobillo y cadera."""
    dv = p - a
    d = float(np.linalg.norm(dv))
    if d < 1e-6:
        return a.copy()
    if d >= l_tib + l_mus:
        return a + dv * (l_tib / (l_tib + l_mus))
    if d <= abs(l_tib - l_mus):
        return a + dv / d * l_tib
    x = (l_tib ** 2 - l_mus ** 2 + d * d) / (2 * d)
    h = math.sqrt(max(l_tib ** 2 - x * x, 0.0))
    u = dv / d
    perp = np.array([-u[1], u[0]])
    k1, k2 = a + u * x + perp * h, a + u * x - perp * h
    return k1 if (_cruz(dv, k1 - a) > 0) == (signo > 0) else k2


def _signo_rodilla(kpf, P, K, A) -> float:
    c = _cruz(P - A, K - A)
    if abs(c) > 0.05 * np.linalg.norm(P - A) * np.linalg.norm(K - A):
        return 1.0 if c > 0 else -1.0
    return _lado_frente(kpf, A, P)


def pose_ideal(kpf: Dict[int, Tuple[float, float]], cod: str, mano: str, reglas: Reglas):
    """
    Construye la postura CORREGIDA sobre los keypoints reales del golpe.
      - Codo: gira el antebrazo alrededor del codo hasta el angulo ideal.
      - Muneca: sube la muneca a la altura del codo (mismo antebrazo).
      - Rodillas: deja los pies fijos y BAJA (o sube) la cadera hasta el
        angulo ideal conservando el largo de muslo y tibia; todo el torso,
        brazos y cabeza bajan con ella. Es lo que haria un entrenador.
    Retorna (kp_ideal, info) o None si faltan puntos.
    """
    idx = _IDX[mano]
    nuevo = {i: _np(p) for i, p in kpf.items()}
    info = dict(vertice=None, p1=None, p2=None, flecha=None, etiqueta="",
                segmentos=[], linea_altura=None, nivel=None)

    def tiene(*nombres):
        return all(idx[n] in kpf for n in nombres)

    if cod in _ARMS or cod in _WRIST:
        if not tiene("hombro", "codo", "muneca"):
            return None
        S, E, Wr = _np(kpf[idx["hombro"]]), _np(kpf[idx["codo"]]), _np(kpf[idx["muneca"]])
        u, w = S - E, Wr - E
        L = float(np.linalg.norm(w))
        if cod in _ARMS:
            obj = (reglas.codo_ideal_min + reglas.codo_ideal_max) / 2.0
            # Hay dos antebrazos posibles con ese angulo: en el drive la paleta
            # sube al doblar el codo, asi que se elige el de la muneca mas alta
            # (de paso, la postura ideal nunca deja la muneca caida).
            un = u / max(np.linalg.norm(u), 1e-6)
            c1 = E + _rot(un, math.radians(obj)) * L
            c2 = E + _rot(un, -math.radians(obj)) * L
            W2 = c1 if c1[1] < c2[1] else c2
            info["etiqueta"] = u"\u2248 %.0f\u00b0" % obj
        else:
            sx = np.sign(w[0]) or np.sign(E[0] - S[0]) or 1.0
            W2 = E + np.array([sx * L, 0.0])
            info["etiqueta"] = u"nivel del codo"
        nuevo[idx["muneca"]] = W2
        if cod in _ARMS:
            info.update(vertice=idx["codo"], p1=idx["hombro"], p2=idx["muneca"],
                        segmentos=[(idx["hombro"], idx["codo"]), (idx["codo"], idx["muneca"])])
        else:
            info.update(vertice=idx["muneca"], p1=idx["codo"], p2=idx["codo"],
                        nivel=(E, W2), segmentos=[(idx["codo"], idx["muneca"])])
        info["flecha"] = (Wr, W2)
        return nuevo, info

    if cod in _KNEE or cod == "rodilla_apoyo":
        lado, otro = ("", "_opp") if cod != "rodilla_apoyo" else ("_opp", "")
        ci, ri, ti = idx["cadera" + lado], idx["rodilla" + lado], idx["tobillo" + lado]
        if not all(i in kpf for i in (ci, ri, ti)):
            return None
        P, K, A = _np(kpf[ci]), _np(kpf[ri]), _np(kpf[ti])
        l_mus, l_tib = float(np.linalg.norm(P - K)), float(np.linalg.norm(K - A))
        obj = (reglas.rodilla_ideal_min + reglas.rodilla_ideal_max) / 2.0
        d_nueva = math.sqrt(max(l_mus ** 2 + l_tib ** 2
                                - 2 * l_mus * l_tib * math.cos(math.radians(obj)), 1.0))
        P2 = A + (P - A) / max(np.linalg.norm(P - A), 1e-6) * d_nueva
        K2 = _rodilla_circulos(A, P2, l_tib, l_mus, _signo_rodilla(kpf, P, K, A))
        delta = P2 - P
        for i in _SUPERIOR:
            if i in nuevo:
                nuevo[i] = nuevo[i] + delta
        nuevo[ci], nuevo[ri] = P2, K2
        # la otra pierna: pies fijos, cadera baja igual
        co, ro, to = idx["cadera" + otro], idx["rodilla" + otro], idx["tobillo" + otro]
        if co in kpf:
            Po2 = _np(kpf[co]) + delta
            nuevo[co] = Po2
            if ro in kpf and to in kpf:
                Po, Ko, Ao = _np(kpf[co]), _np(kpf[ro]), _np(kpf[to])
                nuevo[ro] = _rodilla_circulos(Ao, Po2, float(np.linalg.norm(Ko - Ao)),
                                              float(np.linalg.norm(Po - Ko)),
                                              _signo_rodilla(kpf, Po, Ko, Ao))
        info.update(vertice=ri, p1=ci, p2=ti, flecha=(K, K2),
                    etiqueta=u"\u2248 %.0f\u00b0" % obj,
                    segmentos=[(ci, ri), (ri, ti)])
        return nuevo, info

    if cod == "rodilla_rango":
        caderas = [kpf[i][1] for i in (11, 12) if i in kpf]
        if not caderas or not tiene("cadera", "rodilla", "tobillo"):
            return None
        info.update(vertice=idx["rodilla"], p1=idx["cadera"], p2=idx["tobillo"],
                    linea_altura=sum(caderas) / len(caderas),
                    etiqueta=u"misma altura",
                    segmentos=[(idx["cadera"], idx["rodilla"]),
                               (idx["rodilla"], idx["tobillo"])])
        return nuevo, info
    return None


# ---------- primitivas de dibujo de las tarjetas ----------

def rect_redondo(img, x1, y1, x2, y2, r, color):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    r = int(max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2)))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def pildora(img, piz: Pizarra, x, y, txt, tam, fondo, color_txt, negrita=True,
            limites=None) -> int:
    """Etiqueta redondeada con texto TTF. limites=(x1,y1,x2,y2) la mantiene dentro."""
    w = _ancho_txt(txt, tam, negrita) + 2 * int(tam * 0.6)
    h = int(tam * 1.65)
    if limites is not None:
        x = min(max(x, limites[0] + 4), limites[2] - w - 4)
        y = min(max(y, limites[1] + 4), limites[3] - h - 4)
    rect_redondo(img, x, y, x + w, y + h, h // 2, fondo)
    piz.texto(txt, x + int(tam * 0.6), y + int((h - tam * 1.18) / 2), tam, color_txt, negrita)
    return w


def linea_punteada(img, p1, p2, color, grosor=2, guion=9, hueco=7):
    p1, p2 = _np(p1), _np(p2)
    L = float(np.linalg.norm(p2 - p1))
    if L < 1:
        return
    d = (p2 - p1) / L
    t = 0.0
    while t < L:
        a, b = p1 + d * t, p1 + d * min(t + guion, L)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, grosor, cv2.LINE_AA)
        t += guion + hueco


def _ipt(p) -> Tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _linea_brillo(img, p1, p2, color, grosor=5):
    """Segmento con halo: se lee bien sobre cualquier fondo de video."""
    cv2.line(img, _ipt(p1), _ipt(p2), (20, 20, 20), grosor + 4, cv2.LINE_AA)
    cv2.line(img, _ipt(p1), _ipt(p2), color, grosor, cv2.LINE_AA)


def _esqueleto_foto(img, kpf, color, grosor=1, excluir=()):
    for a, b in _CONEXIONES:
        if a in kpf and b in kpf and (a, b) not in excluir and (b, a) not in excluir:
            cv2.line(img, _ipt(kpf[a]), _ipt(kpf[b]), color, grosor, cv2.LINE_AA)


def _flecha(img, p1, p2, color=TIZA):
    if np.linalg.norm(_np(p2) - _np(p1)) < 7:
        return
    cv2.arrowedLine(img, _ipt(p1), _ipt(p2), (20, 20, 20), 7, cv2.LINE_AA, 0, 0.32)
    cv2.arrowedLine(img, _ipt(p1), _ipt(p2), color, 3, cv2.LINE_AA, 0, 0.32)


def _panel_real(foto, kpf, info, piz, ox, oy, etiqueta):
    """Foto real: esqueleto tenue + la parte mal en rojo goma con su angulo."""
    img = foto.copy()
    _esqueleto_foto(img, kpf, (205, 205, 205), 1, excluir=info["segmentos"])
    for a, b in info["segmentos"]:
        if a in kpf and b in kpf:
            _linea_brillo(img, kpf[a], kpf[b], GOMA, 5)
    v = info["vertice"]
    if v is not None and info["p1"] != info["p2"] and all(i in kpf for i in (v, info["p1"], info["p2"])):
        cuna_angulo(img, _ipt(kpf[v]), _ipt(kpf[info["p1"]]), _ipt(kpf[info["p2"]]), 24, GOMA, 0.5)
    for p in kpf.values():
        cv2.circle(img, _ipt(p), 3, TIZA, -1, cv2.LINE_AA)
    return img, (lambda: _etiqueta_foto(piz, kpf, info, ox, oy, etiqueta, GOMA, TIZA))


def _panel_ideal(foto, kpf, kp_ideal, info, piz, ox, oy):
    """Misma foto oscurecida + la postura corregida en verde + flecha."""
    img = cv2.convertScaleAbs(foto, alpha=0.42)
    _esqueleto_foto(img, kpf, (95, 95, 95), 1)                   # tu golpe, tenue
    for a, b in info["segmentos"]:                               # lo que estaba mal
        if a in kpf and b in kpf:
            linea_punteada(img, kpf[a], kpf[b], GOMA, 2)
    capa = img.copy()                                            # halo verde
    for a, b in _CONEXIONES:
        if a in kp_ideal and b in kp_ideal:
            cv2.line(capa, _ipt(kp_ideal[a]), _ipt(kp_ideal[b]), IDEAL, 12, cv2.LINE_AA)
    cv2.addWeighted(capa, 0.22, img, 0.78, 0, img)
    _esqueleto_foto(img, kp_ideal, IDEAL, 2, excluir=info["segmentos"])
    for a, b in info["segmentos"]:
        if a in kp_ideal and b in kp_ideal:
            _linea_brillo(img, kp_ideal[a], kp_ideal[b], IDEAL, 5)
    v = info["vertice"]
    if v is not None and info["linea_altura"] is None and info["p1"] != info["p2"] and all(
            i in kp_ideal for i in (v, info["p1"], info["p2"])):
        cuna_angulo(img, _ipt(kp_ideal[v]), _ipt(kp_ideal[info["p1"]]),
                    _ipt(kp_ideal[info["p2"]]), 24, IDEAL, 0.5)
    for p in kp_ideal.values():
        cv2.circle(img, _ipt(p), 3, TIZA, -1, cv2.LINE_AA)
    if info["linea_altura"] is not None:
        y = info["linea_altura"]
        linea_punteada(img, (6, y), (FOTO_W - 6, y), BOLA, 2, 12, 8)
    if info["nivel"] is not None:
        e, w2 = info["nivel"]
        x1, x2 = min(e[0], w2[0]) - 26, max(e[0], w2[0]) + 26
        linea_punteada(img, (x1, e[1]), (x2, e[1]), BOLA, 2, 8, 6)
    if info["flecha"] is not None:
        _flecha(img, info["flecha"][0], info["flecha"][1])
    return img, (lambda: _etiqueta_foto(piz, kp_ideal, info, ox, oy, info["etiqueta"], IDEAL,
                                        (25, 40, 20)))


def _etiqueta_foto(piz, kpf, info, ox, oy, txt, fondo, color_txt):
    """Devuelve una funcion que dibuja la etiqueta del angulo. Se ubica del
    lado CONTRARIO a los dos segmentos (no tapa el brazo ni la flecha)."""
    def _f(img):
        v = info["vertice"]
        if not txt or v is None or v not in kpf:
            return
        V = _np(kpf[v])
        d = np.zeros(2)
        for i in (info["p1"], info["p2"]):
            if i in kpf:
                s = _np(kpf[i]) - V
                d += s / max(np.linalg.norm(s), 1e-6)
        if info["flecha"] is not None:
            s = _np(info["flecha"][1]) - V
            d += s / max(np.linalg.norm(s), 1e-6)
        d = -d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-3 else np.array([1.0, 0.0])
        w = _ancho_txt(txt, 16, True) + 20
        h = 26
        # Primero del lado contrario a los segmentos; si ahi se sale de la
        # foto, prueba abajo / arriba / izquierda / derecha.
        cands = [d, np.array([0.0, 1.0]), np.array([0.0, -1.0]),
                 np.array([-1.0, 0.0]), np.array([1.0, 0.0])]
        cx, cy = V + d * 44
        for i, c in enumerate(cands):
            dist = 44 + (w / 2.0 - 20) * abs(c[0])
            px, py = V + c * dist
            if i in (1, 2):              # abajo / arriba: se puede correr de lado
                px = min(max(px, w / 2.0 + 4), FOTO_W - w / 2.0 - 4)
            if 4 <= px - w / 2 and px + w / 2 <= FOTO_W - 4 and 4 <= py - h / 2 \
                    and py + h / 2 <= FOTO_H - 4:
                cx, cy = px, py
                break
        pildora(img, piz, int(ox + cx - w / 2), int(oy + cy - h / 2), txt, 16, fondo, color_txt,
                limites=(ox, oy, ox + FOTO_W, oy + FOTO_H))
    return _f


def _pegar(dst, src, x, y):
    h, w = src.shape[:2]
    dst[y:y + h, x:x + w] = src


def _marco(img, x, y, w, h, color, grosor=2):
    cv2.rectangle(img, (x - grosor, y - grosor), (x + w + grosor - 1, y + h + grosor - 1),
                  color, grosor)


def _medidor(img, piz, x, y, ancho, valores, destacado, lo, hi, unidad, etiqueta_ideal,
             color_dest):
    """Barra horizontal: zona ideal en verde, cada golpe como una marca y
    el golpe mostrado como un triangulo. Mucho mas claro que una grafica."""
    datos = [v for v in valores if v == v]
    if not datos:
        return
    pad = max((hi - lo) * 0.4, 6.0)
    vmin = min(min(datos), lo) - pad
    vmax = max(max(datos), hi) + pad

    def xp(v):
        return int(x + (v - vmin) / (vmax - vmin) * ancho)

    rect_redondo(img, x, y, x + ancho, y + 10, 5, PANEL)
    rect_redondo(img, xp(lo), y, xp(hi), y + 10, 5, IDEAL)
    for v in datos:
        cv2.line(img, (xp(v), y - 5), (xp(v), y + 15), NIEBLA, 1, cv2.LINE_AA)
    if destacado == destacado:
        xd = xp(destacado)
        tri = np.array([[xd - 8, y - 16], [xd + 8, y - 16], [xd, y - 3]], np.int32)
        cv2.fillConvexPoly(img, tri, color_dest, cv2.LINE_AA)
        cv2.line(img, (xd, y - 3), (xd, y + 13), color_dest, 2, cv2.LINE_AA)
        piz.texto(u"este golpe %.0f%s" % (destacado, unidad), min(max(xd, x + 60), x + ancho - 60),
                  y - 38, 14, color_dest, True, "c")
    piz.texto(etiqueta_ideal, (xp(lo) + xp(hi)) // 2, y + 20, 14, IDEAL, False, "c")


def _encabezado(img, piz, sesion: "Sesion", linea_sup, color_sup, titulo):
    n = len(sesion.golpes)
    piz.texto(linea_sup, 40, 22, 17, color_sup, True)
    piz.texto(titulo, 40, 46, 33, TIZA, True)
    mano = "diestro" if sesion.mano == "derecha" else "zurdo"
    piz.texto(u"%d golpes en %s, %s" % (n, mmss(sesion.duracion()), mano),
              RW - 40, 24, 16, NIEBLA, False, "r")
    if n:
        clases = sesion.clases()
        x0, x1, y = RW - 250, RW - 40, 54
        rect_redondo(img, x0, y, x1, y + 10, 5, PANEL)
        x = x0
        for clase, col in (("ok", IDEAL), ("casi", BOLA), ("corregir", GOMA)):
            w = int(round((x1 - x0) * clases.count(clase) / float(n)))
            if w > 0:
                cv2.rectangle(img, (x, y), (min(x + w, x1), y + 10), col, -1)
            x += w
        piz.texto(u"%d%% con buena técnica" % round(100.0 * clases.count("ok") / n),
                  RW - 40, 72, 15, NIEBLA, False, "r")
    cv2.line(img, (40, 106), (RW - 40, 106), (120, 96, 78), 1, cv2.LINE_AA)


def _pie(img, piz, pag, n_pags):
    if n_pags > 1:
        x0 = RW // 2 - (n_pags - 1) * 11
        for i in range(n_pags):
            c = (x0 + i * 22, RH - 58)
            if i == pag:
                cv2.circle(img, c, 6, TIZA, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, c, 5, NIEBLA, 1, cv2.LINE_AA)
    txt = u"Levanta la mano libre o pulsa ENTER para otra sesión"
    if n_pags > 1:
        txt += u".   N: siguiente tarjeta"
    piz.texto(txt + u".   Q: salir", RW // 2, RH - 40, 15, NIEBLA, False, "c")


def _tira_golpes(img, piz, x, y, ancho, sesion, codigos_parte, n_dest):
    """Fila de golpes numerados: en color los que tuvieron este error."""
    piz.texto(u"En qué golpes pasó", x, y, 15, NIEBLA, True)
    paso = 28
    max_n = max(1, ancho // paso)
    golpes = sesion.golpes[-max_n:]
    cy = y + 42
    for i, g in enumerate(golpes):
        cx = x + 12 + i * paso
        sev = max([s for c, s in g.resultado.codigos if c in codigos_parte] or [0])
        col = GOMA if sev == 2 else (BOLA if sev == 1 else PANEL)
        if g.n == n_dest:
            cv2.circle(img, (cx, cy), 15, TIZA, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 11, col, -1, cv2.LINE_AA)
        piz.texto(str(g.n), cx, cy - 8, 12, TIZA if sev else NIEBLA, True, "c")
    if len(sesion.golpes) > max_n:
        piz.texto(u"(últimos %d)" % max_n, x + ancho, y, 13, NIEBLA, False, "r")


# ---------- seleccion de que mostrar ----------

def _metrica(cod: str, reglas: Reglas):
    """(clave en angulos, unidad, zona ideal lo, hi, texto zona)."""
    if cod in _ARMS:
        return ("codo", u"\u00b0", reglas.codo_ideal_min, reglas.codo_ideal_max,
                u"ideal %.0f a %.0f\u00b0" % (reglas.codo_ideal_min, reglas.codo_ideal_max))
    if cod in _WRIST:
        return ("caida_muneca_%", "%", -40.0, reglas.muneca_caida_warn * 100.0,
                u"ideal: caída menor a %.0f%%" % (reglas.muneca_caida_warn * 100.0))
    if cod == "rodilla_rango":
        return ("rango_rodilla", u"\u00b0", 0.0, reglas.rodilla_rango_warn,
                u"ideal: menos de %.0f\u00b0" % reglas.rodilla_rango_warn)
    clave = "rodilla_opp" if cod == "rodilla_apoyo" else "rodilla"
    return (clave, u"\u00b0", reglas.rodilla_ideal_min, reglas.rodilla_ideal_max,
            u"ideal %.0f a %.0f\u00b0" % (reglas.rodilla_ideal_min, reglas.rodilla_ideal_max))


def _valor_real_txt(cod, res: Resultado) -> str:
    if cod in _WRIST:
        return u"muñeca baja"
    clave = _metrica(cod, Reglas())[0]
    v = res.angulos.get(clave, float("nan"))
    if v != v:
        return ""
    if cod == "rodilla_rango":
        return u"varió %.0f\u00b0" % v
    return u"%.0f\u00b0" % v


def correcciones(sesion: "Sesion", max_n: int = 3):
    """
    Agrupa los errores por PARTE DEL CUERPO (no dos tarjetas del codo) y
    las ordena por importancia: errores pesan 2, advertencias 1.
    Retorna [(cod_principal, codigos_de_la_parte, n_golpes_con_error)].
    """
    grupos: Dict[str, dict] = {}
    for g in sesion.golpes:
        vistos = set()
        for cod, sev in set(g.resultado.codigos):
            parte = _parte(cod)
            gr = grupos.setdefault(parte, dict(peso=0, cods={}, golpes=set()))
            gr["peso"] += sev
            gr["cods"][cod] = gr["cods"].get(cod, 0) + sev
            if parte not in vistos:
                gr["golpes"].add(g.n)
                vistos.add(parte)
    orden = sorted(grupos.values(), key=lambda gr: -gr["peso"])
    # La primera siempre; las demas solo si se repiten (>= 2 golpes y >= 20%):
    # un aviso aislado no merece una tarjeta entera.
    n = max(len(sesion.golpes), 1)
    orden = orden[:1] + [gr for gr in orden[1:]
                         if len(gr["golpes"]) >= max(2, 0.2 * n)]
    out = []
    for gr in orden[:max_n]:
        cod = max(gr["cods"], key=lambda c: gr["cods"][c])
        out.append((cod, tuple(gr["cods"].keys()), len(gr["golpes"])))
    return out


def golpe_tipico(sesion: "Sesion", cod: str, reglas: Reglas, mano: str):
    """Golpe REPRESENTATIVO del error (la mediana, no el peor: el peor
    suele ser un frame mal detectado). Debe tener foto y postura ideal."""
    clave = _metrica(cod, reglas)[0]
    cand = []
    for g in sesion.golpes:
        if g.foto is None or not any(c == cod for c, _ in g.resultado.codigos):
            continue
        if pose_ideal(g.kp_foto, cod, mano, reglas) is None:
            continue
        cand.append((g.resultado.angulos.get(clave, 0.0), g))
    if not cand:
        return None
    cand.sort(key=lambda t: t[0])
    return cand[len(cand) // 2][1]


# ---------- tarjetas ----------

X_REAL, X_IDEAL, Y_FOTO = 28, 372, 152
X_TXT, ANCHO_TXT = 706, 226


def tarjeta_correccion(sesion, reglas, k, total, cod, cods_parte, n_con_error):
    img = np.full((RH, RW, 3), MESA, dtype=np.uint8)
    piz = Pizarra()
    t = TARJETAS.get(cod, dict(titulo=CONSEJOS.get(cod, cod), paso="", como="", porque=""))
    n = len(sesion.golpes)
    grave = any(s == 2 for g in sesion.golpes for c, s in g.resultado.codigos if c in cods_parte)
    col_sup = GOMA if grave else BOLA
    _encabezado(img, piz, sesion,
                u"Corrección %d de %d.  Pasó en %d de tus %d golpes" % (k, total, n_con_error, n),
                col_sup, t["titulo"])

    g = golpe_tipico(sesion, cod, reglas, sesion.mano)
    etiquetas = []
    if g is not None:
        foto = foto_de(g)
        kp_ideal, info = pose_ideal(g.kp_foto, cod, sesion.mano, reglas)
        real, et_r = _panel_real(foto, g.kp_foto, info, piz, X_REAL, Y_FOTO,
                                 _valor_real_txt(cod, g.resultado))
        ideal, et_i = _panel_ideal(foto, g.kp_foto, kp_ideal, info, piz, X_IDEAL, Y_FOTO)
        _pegar(img, real, X_REAL, Y_FOTO)
        _pegar(img, ideal, X_IDEAL, Y_FOTO)
        _marco(img, X_REAL, Y_FOTO, FOTO_W, FOTO_H, GOMA)
        _marco(img, X_IDEAL, Y_FOTO, FOTO_W, FOTO_H, IDEAL)
        etiquetas += [et_r(), et_i()]
        pildora(img, piz, X_REAL, 116, u"\u2717  Tu golpe %d" % g.n, 16, GOMA, TIZA)
        pildora(img, piz, X_IDEAL, 116, u"\u2713  Así debería verse", 16, IDEAL, (25, 40, 20))
        # la "red" entre las dos fotos, con la direccion de lectura
        xm = (X_REAL + FOTO_W + X_IDEAL) // 2
        linea_punteada(img, (xm, Y_FOTO + 8), (xm, Y_FOTO + FOTO_H - 8), (120, 96, 78), 1, 5, 5)
        ym = Y_FOTO + FOTO_H // 2
        cv2.circle(img, (xm, ym), 15, MESA, -1, cv2.LINE_AA)
        cv2.circle(img, (xm, ym), 15, TIZA, 1, cv2.LINE_AA)
        cv2.polylines(img, [np.array([[xm - 4, ym - 7], [xm + 4, ym], [xm - 4, ym + 7]], np.int32)],
                      False, TIZA, 2, cv2.LINE_AA)
        n_dest = g.n
    else:
        rect_redondo(img, X_REAL, Y_FOTO, X_IDEAL + FOTO_W, Y_FOTO + FOTO_H, 10, PANEL)
        piz.texto(u"No hay foto clara de este golpe", (X_REAL + X_IDEAL + FOTO_W) // 2,
                  Y_FOTO + FOTO_H // 2 - 10, 18, NIEBLA, False, "c")
        n_dest = -1

    # Columna de texto: que paso, como corregirlo, por que
    y = 120
    piz.texto(u"Qué pasó", X_TXT, y, 17, col_sup, True)
    y = piz.parrafo(t["paso"], X_TXT, y + 26, ANCHO_TXT, 17, TIZA) + 16
    piz.texto(u"Cómo corregirlo", X_TXT, y, 17, IDEAL, True)
    y = piz.parrafo(t["como"], X_TXT, y + 26, ANCHO_TXT, 17, TIZA) + 16
    piz.texto(u"Por qué importa", X_TXT, y, 17, NIEBLA, True)
    piz.parrafo(t["porque"], X_TXT, y + 26, ANCHO_TXT, 15, NIEBLA)

    # En que golpes paso + medidor con la zona ideal
    _tira_golpes(img, piz, X_REAL, 572, X_IDEAL + FOTO_W - X_REAL, sesion, cods_parte, n_dest)
    clave, unidad, lo, hi, txt_ideal = _metrica(cod, reglas)
    valores = [gg.resultado.angulos.get(clave, float("nan")) for gg in sesion.golpes]
    dest = g.resultado.angulos.get(clave, float("nan")) if g is not None else float("nan")
    piz.texto(u"Todos tus golpes", X_TXT, 572, 15, NIEBLA, True)
    _medidor(img, piz, X_TXT + 8, 626, ANCHO_TXT - 16, valores, dest, lo, hi, unidad,
             txt_ideal, col_sup)
    for f in etiquetas:
        f(img)
    return img, piz


def tarjeta_mejor(sesion, reglas):
    """Refuerzo positivo: el golpe mas cercano al ideal, para repetirlo."""
    oks = [g for g in sesion.golpes if clase_de(g.resultado) == "ok" and g.foto is not None]
    if not oks:
        return None
    mc = (reglas.codo_ideal_min + reglas.codo_ideal_max) / 2.0
    mr = (reglas.rodilla_ideal_min + reglas.rodilla_ideal_max) / 2.0
    g = min(oks, key=lambda gg: abs(gg.resultado.angulos.get("codo", mc) - mc)
            + abs(gg.resultado.angulos.get("rodilla", mr) - mr))
    img = np.full((RH, RW, 3), MESA, dtype=np.uint8)
    piz = Pizarra()
    _encabezado(img, piz, sesion, u"Tu mejor golpe de la sesión", IDEAL, u"Repite este golpe")
    foto = foto_de(g)
    kpf = g.kp_foto
    idx = _IDX[sesion.mano]
    capa = foto.copy()
    _esqueleto_foto(capa, kpf, IDEAL, 2)
    for a, b in (("hombro", "codo"), ("codo", "muneca"), ("cadera", "rodilla"), ("rodilla", "tobillo")):
        if idx[a] in kpf and idx[b] in kpf:
            _linea_brillo(capa, kpf[idx[a]], kpf[idx[b]], IDEAL, 5)
    for p in kpf.values():
        cv2.circle(capa, _ipt(p), 3, TIZA, -1, cv2.LINE_AA)
    x0 = 120
    _pegar(img, capa, x0, Y_FOTO)
    _marco(img, x0, Y_FOTO, FOTO_W, FOTO_H, IDEAL)
    pildora(img, piz, x0, 116, u"\u2713  Golpe %d" % g.n, 16, IDEAL, (25, 40, 20))
    lim = (x0, Y_FOTO, x0 + FOTO_W, Y_FOTO + FOTO_H)
    for clave, nombre in (("codo", "codo"), ("rodilla", "rodilla")):
        if clave in g.resultado.angulos and idx[nombre] in kpf:
            x, y = kpf[idx[nombre]]
            pildora(img, piz, int(x0 + x + 22), int(Y_FOTO + y - 30),
                    u"%.0f\u00b0" % g.resultado.angulos[clave], 16, IDEAL, (25, 40, 20), limites=lim)

    xt, y = 500, 160
    piz.texto(u"Así se hace", xt, y, 26, TIZA, True)
    y = piz.parrafo(u"Este golpe estuvo dentro de la zona ideal en todo lo que mide el "
                    u"entrenador. Antes de la próxima sesión, haz 5 golpes lentos copiando "
                    u"esta postura.", xt, y + 44, 400, 17, TIZA) + 22
    for nombre, clave, lo, hi in (("Codo", "codo", reglas.codo_ideal_min, reglas.codo_ideal_max),
                                  ("Rodilla", "rodilla", reglas.rodilla_ideal_min,
                                   reglas.rodilla_ideal_max)):
        v = g.resultado.angulos.get(clave)
        if v is None:
            continue
        cv2.circle(img, (xt + 7, y + 11), 7, IDEAL, -1, cv2.LINE_AA)
        piz.texto(u"%s %.0f\u00b0" % (nombre, v), xt + 24, y, 19, TIZA, True)
        piz.texto(u"zona ideal %.0f a %.0f\u00b0" % (lo, hi), xt + 24, y + 26, 15, NIEBLA)
        y += 62
    if "caida_muneca_%" in g.resultado.angulos:
        cv2.circle(img, (xt + 7, y + 11), 7, IDEAL, -1, cv2.LINE_AA)
        piz.texto(u"Muñeca firme", xt + 24, y, 19, TIZA, True)
        piz.texto(u"a la altura del codo al tocar la bola", xt + 24, y + 26, 15, NIEBLA)
        y += 62
    n_ok = len([gg for gg in sesion.golpes if clase_de(gg.resultado) == "ok"])
    rect_redondo(img, xt, y + 20, xt + 400, y + 74, 8, PANEL)
    piz.texto(u"Golpes así en esta sesión: %d de %d" % (n_ok, len(sesion.golpes)),
              xt + 200, y + 37, 17, IDEAL, True, "c")
    return img, piz


def _miniatura(g: "Golpe", mano: str, w: int, h: int):
    foto = foto_de(g)
    if foto is None:
        m = np.full((h, w, 3), PANEL, dtype=np.uint8)
        return m
    m = cv2.resize(foto, (w, h), interpolation=cv2.INTER_AREA)
    e = w / float(FOTO_W)
    kpf = {i: (p[0] * e, p[1] * e) for i, p in g.kp_foto.items()}
    idx = _IDX[mano]
    res = g.resultado
    _esqueleto_foto(m, kpf, (200, 200, 200), 1)
    sev_col = {2: GOMA, 1: BOLA, 0: IDEAL}
    for grupo, segs in ((_COD_BRAZO, (("hombro", "codo"), ("codo", "muneca"))),
                        (_COD_PIERNA, (("cadera", "rodilla"), ("rodilla", "tobillo")))):
        sev = max([s for c, s in res.codigos if c in grupo] or [0])
        for a, b in segs:
            if idx[a] in kpf and idx[b] in kpf:
                _linea_brillo(m, kpf[idx[a]], kpf[idx[b]], sev_col[sev], 3)
    return m


def tarjeta_galeria(sesion, reglas, prioridad):
    img = np.full((RH, RW, 3), MESA, dtype=np.uint8)
    piz = Pizarra()
    n = len(sesion.golpes)
    clases = sesion.clases()
    n_ok, n_casi = clases.count("ok"), clases.count("casi")
    n_corr = n - n_ok - n_casi
    titulo = u"Todos tus golpes en la zona ideal" if n_ok == n else u"Tus %d golpes" % n
    _encabezado(img, piz, sesion, u"Resumen de la sesión", NIEBLA, titulo)

    # Dona de calidad
    c = (168, 290)
    _dona(img, c, 88, 22, [n_ok / float(n), n_casi / float(n), n_corr / float(n)],
          [IDEAL, BOLA, GOMA])
    piz.texto(u"%d%%" % round(100.0 * n_ok / n), c[0], c[1] - 30, 40, TIZA, True, "c")
    piz.texto(u"buena técnica", c[0], c[1] + 18, 15, NIEBLA, False, "c")
    y = 420
    for etiqueta, cnt, col in ((u"Bien", n_ok, IDEAL), (u"Casi", n_casi, BOLA),
                               (u"A corregir", n_corr, GOMA)):
        cv2.circle(img, (78, y + 10), 7, col, -1, cv2.LINE_AA)
        piz.texto(etiqueta, 96, y, 17, TIZA)
        piz.texto(str(cnt), 258, y, 17, TIZA, True, "r")
        y += 32
    r = sesion.mejor_racha()
    piz.texto(u"Mejor racha: %d %s" % (r, u"golpe bien" if r == 1 else u"golpes bien seguidos"),
              70, y + 10, 15, NIEBLA)

    # Galeria de miniaturas (las ultimas 8)
    tw, th, gap = 135, 180, 16
    gx, gy = 330, 128
    mostrar = sesion.golpes[-8:]
    for i, g in enumerate(mostrar):
        x = gx + (i % 4) * (tw + gap)
        yy = gy + (i // 4) * (th + 54)
        clase = clase_de(g.resultado)
        col = {"ok": IDEAL, "casi": BOLA, "corregir": GOMA}[clase]
        _pegar(img, _miniatura(g, sesion.mano, tw, th), x, yy)
        _marco(img, x, yy, tw, th, col, 3)
        cv2.circle(img, (x + 18, yy + 18), 14, col, -1, cv2.LINE_AA)
        piz.texto(str(g.n), x + 18, yy + 9, 14, (25, 25, 25), True, "c")
        partes = []
        for c, _ in sorted(g.resultado.codigos, key=lambda cs: -cs[1]):
            p = _parte(c)
            if p not in partes:
                partes.append(p)
        etiqueta = u"Bien" if not partes else u", ".join(
            [partes[0]] + [p.lower() for p in partes[1:3]])
        piz.parrafo(etiqueta, x, yy + th + 8, tw, 14, col, True, 1.2)
    if n > 8:
        piz.texto(u"Se muestran los últimos 8 de %d" % n, RW - 40, gy + 2 * (th + 54) - 6,
                  13, NIEBLA, False, "r")

    # Lo primero a trabajar
    y0 = 600
    rect_redondo(img, 40, y0, RW - 40, y0 + 44, 8, PANEL)
    if prioridad:
        cod, _, veces = prioridad[0]
        cv2.rectangle(img, (40, y0), (47, y0 + 44), BOLA, -1)
        piz.texto(u"Lo primero a trabajar:  %s  (%d de %d golpes)"
                  % (TARJETAS.get(cod, {}).get("titulo", cod), veces, n),
                  RW // 2, y0 + 12, 17, BOLA, True, "c")
    else:
        cv2.rectangle(img, (40, y0), (47, y0 + 44), IDEAL, -1)
        piz.texto(u"Excelente: sigue así y sube la velocidad de a poco",
                  RW // 2, y0 + 12, 17, IDEAL, True, "c")
    return img, piz


def _dona(img, centro, radio, grosor, fracciones, colores):
    inicio = -90.0
    cv2.circle(img, centro, radio, PANEL, grosor, cv2.LINE_AA)
    for f, col in zip(fracciones, colores):
        if f <= 0:
            continue
        fin = inicio + 360.0 * f
        cv2.ellipse(img, centro, (radio, radio), 0, inicio, fin, col, grosor, cv2.LINE_AA)
        inicio = fin


def construir_resumen(sesion: "Sesion", reglas: Reglas) -> List[np.ndarray]:
    """Lista de tarjetas (imagenes RW x RH) listas para mostrar."""
    if not sesion.golpes:
        img = np.full((RH, RW, 3), MESA, dtype=np.uint8)
        piz = Pizarra()
        _encabezado(img, piz, sesion, u"Resumen de la sesión", NIEBLA, u"No se registraron golpes")
        piz.parrafo(u"Revisa que la cámara vea tu cuerpo completo y el brazo con el que golpeas. "
                    u"Cada golpe se cuenta cuando la muñeca hace un movimiento rápido y amplio.",
                    120, 300, RW - 240, 19, NIEBLA)
        _pie(img, piz, 0, 1)
        return [piz.dibujar(img)]

    prio = correcciones(sesion)
    crudas = [tarjeta_correccion(sesion, reglas, k + 1, len(prio), cod, cods, veces)
              for k, (cod, cods, veces) in enumerate(prio)]
    mejor = tarjeta_mejor(sesion, reglas)
    if mejor is not None:
        crudas.append(mejor)
    crudas.append(tarjeta_galeria(sesion, reglas, prio))
    paginas = []
    for i, (img, piz) in enumerate(crudas):
        _pie(img, piz, i, len(crudas))
        paginas.append(piz.dibujar(img))
    return paginas


def dibujar_avance(img, fraccion: float):
    """Barra fina inferior: cuanto falta para pasar a la siguiente tarjeta."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 4), (int(w * min(max(fraccion, 0.0), 1.0)), h), NIEBLA, -1)

# ==============================================================
# FPS
# ==============================================================

class FPS:
    def __init__(self, n: int = 30):
        self._t: deque = deque(maxlen=n)

    def tick(self) -> float:
        self._t.append(time.perf_counter())
        if len(self._t) < 2:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])



# ==============================================================
# CAMARA EN HILO - siempre el frame mas nuevo
# ==============================================================

class CamaraHilo:
    """
    Lee la camara USB en un hilo aparte. El bucle principal toma siempre
    el frame MAS NUEVO (sin cola de frames viejos = menos retraso) y la
    lectura/decodificacion MJPG ocurre en paralelo con la inferencia.
    Cada frame entregado es un arreglo nuevo, asi que dibujar sobre el no
    afecta al hilo.
    """
    def __init__(self, indice: int, ancho: int, alto: int, fps: int, mjpg: bool):
        self.cap = self._abrir(indice, ancho, alto, fps, mjpg)
        self.vivo = self.cap is not None and self.cap.isOpened()
        self._frame = None
        self._id = 0
        self._cond = threading.Condition()
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        if self.vivo:
            self._hilo.start()

    @staticmethod
    def _configurar(cap, ancho, alto, fps, mjpg):
        if mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def _abrir(self, indice, ancho, alto, fps, mjpg):
        """
        En la Jetson, cv2.VideoCapture(0) usa GStreamer por defecto y ese
        backend no acepta MJPG/resolucion/buffer (falla el pipeline).
        Se prueba primero V4L2 (driver nativo USB) y luego respaldos.
        Solo se acepta la camara si entrega un frame real.
        """
        intentos = [
            (indice, cv2.CAP_V4L2, mjpg, "V4L2 + MJPG" if mjpg else "V4L2"),
            (indice, cv2.CAP_V4L2, False, "V4L2 sin MJPG"),
            (indice, None, False, "backend por defecto"),
        ]
        otro = 1 if indice == 0 else 0      # la camara tambien sale en /dev/video1
        intentos.append((otro, cv2.CAP_V4L2, False, "V4L2 sin MJPG (video%d)" % otro))
        for idx, backend, usar_mjpg, nombre in intentos:
            try:
                cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
            except Exception:
                continue
            if not cap.isOpened():
                cap.release()
                continue
            if backend is not None:
                self._configurar(cap, ancho, alto, fps, usar_mjpg)
            ok = False
            for _ in range(10):                 # algunas camaras tardan en dar el 1er frame
                ok, f = cap.read()
                if ok and f is not None:
                    break
                time.sleep(0.05)
            if ok:
                log("[OK] Camara /dev/video%d abierta con %s (%dx%d)" % (
                    idx, nombre,
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                return cap
            log("[INFO] Camara /dev/video%d con %s no entrego imagen, probando otra forma..." % (idx, nombre))
            cap.release()
        return None

    def _bucle(self):
        fallos = 0
        while self.vivo:
            ok, f = self.cap.read()
            if not ok:
                fallos += 1
                if fallos > 60:                 # ~2 s sin imagen: camara caida
                    break
                time.sleep(0.03)
                continue
            fallos = 0
            with self._cond:
                self._frame = f
                self._id += 1
                self._cond.notify_all()
        with self._cond:
            self.vivo = False
            self._cond.notify_all()

    def leer(self, ultimo_id: int, timeout: float = 1.0):
        """Espera un frame NUEVO (id distinto al ultimo procesado)."""
        with self._cond:
            if self._id == ultimo_id and self.vivo:
                self._cond.wait(timeout)
            if self._frame is None or self._id == ultimo_id:
                return None, ultimo_id
            return self._frame, self._id

    def liberar(self):
        self.vivo = False
        if self._hilo.is_alive():
            self._hilo.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


# ==============================================================
# CARGA DEL MODELO trt_pose (fp32 -> TensorRT, con cache en disco)
# ==============================================================

def cargar_modelo_trt():
    # coco_category_to_topology espera el esqueleto en base 1 (como el
    # human_pose.json original); HUMAN_POSE esta en base 0 para dibujar.
    esqueleto_1b = [[a + 1, b + 1] for a, b in HUMAN_POSE["skeleton"]]
    topology = trt_pose.coco.coco_category_to_topology(
        {"keypoints": HUMAN_POSE["keypoints"], "skeleton": esqueleto_1b})
    num_parts = len(HUMAN_POSE["keypoints"])
    num_links = len(HUMAN_POSE["skeleton"])

    if os.path.exists(OPTIMIZED_MODEL):
        log("[OK] Cargando engine TensorRT ya optimizado: " + OPTIMIZED_MODEL)
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
    else:
        # Solo aqui hacen falta estos modulos (pesados): import diferido.
        # Alias: 'import trt_pose.models' aqui volveria local a 'trt_pose'
        # en toda la funcion y romperia la linea de la topologia.
        from trt_pose import models as _trt_models
        from torch2trt import torch2trt
        import onnx_graphsurgeon as _gs
        # El backend ONNX de torch2trt intenta "constant folding", que puede
        # requerir onnxruntime (dificil en la Nano). Es solo optimizacion.
        _gs.Graph.fold_constants = lambda self, *a, **k: self

        log("[INFO] No existe %s todavia." % OPTIMIZED_MODEL)
        log("[INFO] Cargando pesos base: " + MODEL_WEIGHTS)
        model = _trt_models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
        model.load_state_dict(torch.load(MODEL_WEIGHTS))
        log("[INFO] Convirtiendo a TensorRT (fp16, backend ONNX)... puede tardar varios minutos.")
        data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
        # use_onnx=True evita el bug de torch2trt 0.5.0 con el modelo "_att"
        # (ver trt_pose issue #178 / PR #188).
        model_trt = torch2trt(model, [data], fp16_mode=True,
                              max_workspace_size=1 << 25, use_onnx=True)
        torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
        log("[OK] Engine guardado en: " + OPTIMIZED_MODEL)

    parse_objects = ParseObjects(topology)
    return model_trt, parse_objects


class Preproceso:
    """
    Resize en CPU (640x480 -> 224x224, ~0.3 ms) y el resto en la GPU:
    BGR->RGB, a float, normalizacion ImageNet en una sola operacion
    (x * escala + sesgo). Reemplaza cvtColor + PIL + to_tensor en CPU.
    """
    def __init__(self):
        dev = torch.device("cuda")
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1)
        self.escala = 1.0 / (255.0 * std)
        self.sesgo = -mean / std
        self.dev = dev

    def __call__(self, frame_bgr: np.ndarray):
        peq = cv2.resize(frame_bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(peq).to(self.dev)              # HxWx3 uint8 (BGR)
        t = t.permute(2, 0, 1).flip(0).float()              # 3xHxW RGB float
        return (t * self.escala + self.sesgo).unsqueeze(0).contiguous()


def inferir(model_trt, parse_objects, prep: Preproceso, frame):
    """Retorna (n_personas, objects[0] numpy, peaks[0] numpy)."""
    with torch.no_grad():
        cmap, paf = model_trt(prep(frame))
        cmap, paf = cmap.cpu(), paf.cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    return int(counts[0]), objects[0].numpy(), peaks[0].numpy()


# ==============================================================
# MAIN - flujo: INICIO -> CUENTA (3 s) -> SESION -> RESUMEN
# ==============================================================

VENTANA = "Entrenador de Derecha - Tenis de Mesa"
SEG_CUENTA = 3.0
FRAMES_GESTO_INICIO = 12    # ~1 s a 12 FPS
FRAMES_GESTO_FIN    = 18    # ~1.5 s (mas largo para no terminar sin querer)
FANTASMA_SEG        = 1.5   # cuanto dura la pose del impacto en pantalla
AVISO_BRAZO_SEG     = 5.0   # "no veo tu brazo" solo tras 5 s seguidos
AVISO_CUERPO_SEG    = 2.0   # "cuerpo incompleto" solo tras 2 s seguidos

# Codigos de waitKeyEx para las flechas (GTK en la Jetson / Windows)
_FLECHA_DER = (65363, 2555904)
_FLECHA_IZQ = (65361, 2424832)


def main():
    MANO   = "derecha"
    PAUSA  = False
    REGLAS = Reglas()
    fps_c  = FPS()
    det_c  = DetectorGolpe()
    sel_c  = SelectorJugador()
    gesto  = GestoMano()
    calib  = Calibrador()
    av_brazo = AvisoPersistente(AVISO_BRAZO_SEG)
    av_cuerpo = AvisoPersistente(AVISO_CUERPO_SEG)

    estado = "INICIO"
    sesion: Optional[Sesion] = None
    ultimo: Optional[Golpe] = None
    t_ultimo = 0.0
    t_cuenta = 0.0
    paginas: List[np.ndarray] = []
    pag, t_pag = 0, 0.0
    congelado = None
    pantalla_completa = False
    kp: Dict[int, Punto] = {}
    vivo: Optional[Resultado] = None
    interf = False

    log("\n" + "=" * 58)
    log("  GOLPE DE DERECHA - TENIS DE MESA (Jetson / trt_pose v5)")
    log("=" * 58)
    model_trt, parse_objects = cargar_modelo_trt()
    prep = Preproceso()

    # Calentamiento: los primeros llamados al engine son lentos
    dummy = np.zeros((CAP_HEIGHT, CAP_WIDTH, 3), np.uint8)
    for _ in range(3):
        inferir(model_trt, parse_objects, prep, dummy)

    cam = CamaraHilo(CAMARA_INDICE, CAP_WIDTH, CAP_HEIGHT, CAP_FPS, CAMARA_MJPG)
    if not cam.vivo:
        log("[ERROR] No se pudo abrir la camara %d." % CAMARA_INDICE)
        return
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA, 960, 720)

    log("  Mano libre sobre la cabeza (~1 s) o ENTER -> empezar sesion")
    log("  Mano libre sobre la cabeza (~1.5 s) o F   -> terminar sesion")
    log("  SPACE  -> Pausar (y guardar muestra manual)")
    log("  C      -> Deshacer el ultimo golpe/muestra")
    log("  D / Z  -> Diestro / Zurdo (fuera de la sesion)")
    log("  N / B  -> Siguiente / anterior tarjeta del resumen")
    log("  P      -> Pantalla completa")
    log("  S      -> Screenshot")
    log("  Q      -> Salir")
    log("=" * 58 + "\n")

    def terminar_sesion():
        sesion.t_fin = time.time()
        t0 = time.perf_counter()
        pags = construir_resumen(sesion, REGLAS)
        sello = time.strftime("%Y%m%d_%H%M%S")
        for i, p in enumerate(pags):
            cv2.imwrite("resumen_sesion_%s_%d.png" % (sello, i + 1), p)
        log("[FIN] Sesion: %d golpes en %s. %d tarjetas (%.0f ms) -> resumen_sesion_%s_*.png"
            % (len(sesion.golpes), mmss(sesion.duracion()), len(pags),
               1000 * (time.perf_counter() - t0), sello))
        return pags

    fid = 0
    while True:
        frame, fid_nuevo = cam.leer(fid)
        if frame is None:
            if not cam.vivo:
                log("[ERROR] No se puede leer la camara.")
                break
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            continue
        fid = fid_nuevo
        H, W = frame.shape[:2]
        fps = fps_c.tick()
        ahora = time.time()

        # ---------- inferencia (se pausa solo con SPACE en sesion) ----------
        if not PAUSA:
            count, objs, pks = inferir(model_trt, parse_objects, prep, frame)
            obj_idx = sel_c.elegir(count, objs, pks)
            kp = obtener_keypoints(objs, pks, obj_idx)
            vivo = analizar(kp, MANO, REGLAS, W, H) if kp else None
            interf = sel_c.interferencia
        arriba = bool(kp) and not interf and GestoMano.mano_arriba(kp, MANO) is not None

        # ---------- maquina de estados ----------
        if estado == "INICIO":
            checks = [
                ("Jugador en cuadro", vivo is not None),
                ("Cuerpo completo (cabeza a pies)", vivo is not None and cuerpo_completo(kp)),
                ("Brazo que golpea visible", vivo is not None and vivo.brazo_ok),
                ("Nadie mas encima del jugador", vivo is not None and not interf),
            ]
            prog, disparo = gesto.actualizar(arriba, FRAMES_GESTO_INICIO)
            vista = frame
            pantalla_inicio(vista, kp, MANO, checks, prog, fps)
            if disparo:
                estado, t_cuenta = "CUENTA", ahora

        elif estado == "CUENTA":
            restante = SEG_CUENTA - (ahora - t_cuenta)
            vista = frame
            if kp:
                dibujar_esqueleto_suave(vista, kp, W, H, (80, 150, 80))
            if restante <= 0:
                sesion = Sesion(MANO)
                det_c.reiniciar()
                av_brazo.reiniciar()
                av_cuerpo.reiniciar()
                ultimo, t_ultimo = None, 0.0
                estado = "SESION"
                log("[INICIO] Sesion en marcha")
            else:
                pantalla_cuenta(vista, restante)

        elif estado == "SESION":
            avisos: List[str] = []
            prog_fin = 0.0
            if PAUSA:
                vista = congelado.copy()
            else:
                vista = frame
                nuevo = None
                idx = _IDX[MANO]

                # 1) ANALISIS primero, con el frame todavia LIMPIO (la foto
                #    del impacto no debe llevar el HUD dibujado encima)
                if vivo is None or interf:
                    det_c.reiniciar()
                elif vivo.brazo_ok:
                    mun = pt_arr(kp[idx["muneca"]], W, H)
                    largo_sup = float(np.linalg.norm(pt_arr(kp[idx["hombro"]], W, H)
                                                     - pt_arr(kp[idx["codo"]], W, H)))
                    nuevo = det_c.actualizar(mun, largo_sup, vivo, frame)
                else:
                    det_c.frame_sin_brazo()

                if nuevo is not None:
                    evaluar_estabilidad_rodilla(nuevo.resultado, nuevo.rango_rodilla, REGLAS)
                    nuevo.n = len(sesion.golpes) + 1
                    sesion.golpes.append(nuevo)
                    ultimo, t_ultimo = nuevo, ahora
                    calib.capturar(nuevo.resultado, MANO, "auto", nuevo.vel_pico)
                    r = nuevo.resultado
                    log("[GOLPE #%d] %-8s codo %.0f | rodilla %.0f | caida %.0f%% | %s"
                        % (nuevo.n, clase_de(r).upper(),
                           r.angulos.get("codo", float("nan")),
                           r.angulos.get("rodilla", float("nan")),
                           r.angulos.get("caida_muneca_%", float("nan")),
                           consejo_principal(r) or "buen golpe"))

                # 2) DIBUJO
                if vivo is not None:
                    if interf:
                        dibujar_esqueleto_suave(vista, kp, W, H, (120, 120, 120))
                        avisos.append("Otra persona encima del jugador - analisis en pausa")
                    else:
                        dibujar_esqueleto_suave(vista, kp, W, H)
                        if det_c.activo and idx["muneca"] in kp:
                            cv2.circle(vista, pt_px(kp[idx["muneca"]], W, H), 12,
                                       (80, 200, 255), 2, cv2.LINE_AA)
                brazo_perdido = vivo is not None and not interf and not vivo.brazo_ok
                cuerpo_mal = vivo is not None and not interf and not cuerpo_completo(kp)
                if av_brazo.actualizar(brazo_perdido, ahora):
                    avisos.append("No veo tu brazo de golpe: revisa la camara")
                elif av_cuerpo.actualizar(cuerpo_mal, ahora):
                    avisos.append("Alejate un poco: no se ve tu cuerpo completo")

                if ultimo is not None and ahora - t_ultimo < FANTASMA_SEG:
                    dibujar_fantasma_impacto(vista, ultimo, MANO, W, H,
                                             1.0 - (ahora - t_ultimo) / FANTASMA_SEG)

                prog_fin, disparo = gesto.actualizar(arriba and not det_c.activo,
                                                     FRAMES_GESTO_FIN)
                if disparo:
                    paginas = terminar_sesion()
                    pag, t_pag = 0, ahora
                    estado = "RESUMEN"

            if estado == "SESION":
                hud_sesion(vista, sesion, ultimo, t_ultimo, det_c.activo, avisos,
                           prog_fin, fps, PAUSA)

        else:  # RESUMEN - tarjetas que pasan solas
            if len(paginas) > 1 and ahora - t_pag > SEG_PAGINA:
                pag, t_pag = (pag + 1) % len(paginas), ahora
            vista = paginas[pag].copy()
            if len(paginas) > 1:
                dibujar_avance(vista, (ahora - t_pag) / SEG_PAGINA)
            prog, disparo = gesto.actualizar(arriba, FRAMES_GESTO_INICIO)
            if prog > 0:
                anillo_progreso(vista, (RW - 48, RH - 52), 22, prog, IDEAL, 5)
            if disparo:
                estado, t_cuenta = "CUENTA", ahora

        cv2.imshow(VENTANA, vista)
        k = cv2.waitKeyEx(1)
        tecla = 255 if k == -1 else (k & 0xFF)
        if k in _FLECHA_DER:
            tecla = ord('n')
        elif k in _FLECHA_IZQ:
            tecla = ord('b')

        # ---------- teclado ----------
        if tecla == ord('q'):
            break
        elif tecla in (13, 10):
            if estado in ("INICIO", "RESUMEN"):
                estado, t_cuenta = "CUENTA", ahora
        elif tecla in (ord('n'), ord('b')):
            if estado == "RESUMEN" and paginas:
                pag = (pag + (1 if tecla == ord('n') else -1)) % len(paginas)
                t_pag = ahora
        elif tecla == ord('f'):
            if estado == "SESION":
                PAUSA = False
                paginas = terminar_sesion()
                pag, t_pag = 0, ahora
                estado = "RESUMEN"
        elif tecla in (ord('d'), ord('z')):
            if estado == "SESION":
                log("[INFO] Termina la sesion (F) para cambiar de mano.")
            else:
                MANO = "derecha" if tecla == ord('d') else "zurdo"
                det_c.reiniciar()
                log("[OK] Modo: " + ("DIESTRO" if MANO == "derecha" else "ZURDO"))
        elif tecla == ord(' '):
            if estado == "SESION":
                PAUSA = not PAUSA
                if PAUSA:
                    congelado = vista.copy()
                    det_c.reiniciar()
                    if vivo is not None and vivo.brazo_ok and not interf:
                        m = calib.capturar(vivo, MANO, "manual")
                        log("[MANUAL] Muestra #%d -> codo %.0f | rodilla %.0f"
                            % (m.n, m.codo, m.rodilla))
                    else:
                        log("[PAUSA] Sin jugador valido, no se guardo muestra.")
                else:
                    log("[OK] REANUDADO")
        elif tecla == ord('c'):
            eliminada = calib.deshacer_ultima()
            if eliminada is None:
                log("[INFO] No hay muestras para deshacer.")
            else:
                if eliminada.origen == "auto" and sesion is not None and sesion.golpes \
                        and estado == "SESION":
                    sesion.golpes.pop()
                    ultimo = sesion.golpes[-1] if sesion.golpes else None
                    t_ultimo = 0.0
                log("[OK] Muestra #%d (%s) eliminada." % (eliminada.n, eliminada.origen))
        elif tecla == ord('p'):
            pantalla_completa = not pantalla_completa
            cv2.setWindowProperty(VENTANA, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if pantalla_completa
                                  else cv2.WINDOW_NORMAL)
        elif tecla == ord('s'):
            fn = "forehand_jetson_%d.jpg" % int(time.time())
            cv2.imwrite(fn, vista)
            log("[OK] Screenshot: " + fn)

    if estado == "SESION" and sesion is not None:
        terminar_sesion()
    cam.liberar()
    cv2.destroyAllWindows()
    log(calib.resumen())
    calib.cerrar()
    log("\n[OK] Programa terminado.")


if __name__ == "__main__":
    main()