"""
BeeCounter — Compteur d'abeilles par vision IA
================================================
Pipeline :
  1. YOLO v8n  → détecte les objets "bee" / "insect" frame par frame
  2. SORT       → traque chaque abeille entre les frames (ID stable)
  3. Crossing   → ligne virtuelle horizontale au centre → IN/OUT selon direction

Le modèle utilisé est yolov8n.pt (insectes = classe "insect" du dataset Open Images
ou un modèle fine-tuné). À défaut, toutes les détections de petits objets
sont considérées comme des abeilles (fallback générique).
"""

import os
import cv2
import json
import time
import tempfile
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

from tracker import Sort

logger = logging.getLogger("bee_counter")

# ── Paramètres ────────────────────────────────────────────────────────────────
MODEL_PATH   = os.getenv("YOLO_MODEL", "models/best.pt")  # fine-tuné si dispo
FALLBACK_MODEL = "yolov8n.pt"                              # COCO générique sinon
CONF_THRESH  = float(os.getenv("CONF_THRESH",  "0.30"))   # seuil de confiance YOLO
IOU_THRESH   = float(os.getenv("IOU_THRESH",   "0.45"))   # NMS IoU
LINE_POS     = float(os.getenv("LINE_POS",     "0.50"))   # position relative de la ligne (0-1)
MAX_AGE      = int(os.getenv("SORT_MAX_AGE",   "8"))      # frames avant perte d'un tracker
MIN_HITS     = int(os.getenv("SORT_MIN_HITS",  "2"))      # hits min avant activation

# Classes COCO qui peuvent être des abeilles (approx si pas de modèle fin-tuné)
BEE_CLASSES_COCO = {
    # YOLOv8 COCO n'a pas "bee" — on filtre par petite surface si modèle générique
}

@dataclass
class CountResult:
    bee_in:      int   = 0
    bee_out:     int   = 0
    total_tracks: int  = 0
    frames:      int   = 0
    duration_s:  float = 0.0
    fps:         float = 0.0
    line_y_px:   int   = 0
    annotated_video_path: Optional[str] = None
    per_frame: list = field(default_factory=list)   # liste {frame, in_cumul, out_cumul}

    def to_dict(self):
        d = asdict(self)
        d.pop("per_frame")  # trop lourd pour l'API
        d["solde"] = self.bee_in - self.bee_out
        return d


class BeeCounter:
    def __init__(self):
        self._model   = None
        self._use_fine_tuned = False
        self._load_model()

    def _load_model(self):
        from ultralytics import YOLO
        if Path(MODEL_PATH).exists():
            logger.info(f"Chargement modèle fine-tuné : {MODEL_PATH}")
            self._model = YOLO(MODEL_PATH)
            self._use_fine_tuned = True
        else:
            logger.info(f"Modèle fine-tuné introuvable → fallback {FALLBACK_MODEL}")
            self._model = YOLO(FALLBACK_MODEL)
            self._use_fine_tuned = False
        logger.info("Modèle YOLO prêt.")

    def _detect_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Retourne np.array(N,5) = [x1,y1,x2,y2,conf]
        """
        results = self._model.predict(
            frame,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            verbose=False,
            device="cpu",
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 5))

        xyxy  = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls   = boxes.cls.cpu().numpy().astype(int)

        if self._use_fine_tuned:
            # modèle fine-tuné → classe 0 = bee, toutes gardées
            mask = np.ones(len(cls), dtype=bool)
        else:
            # COCO générique → filtre sur petite surface (< 5% de l'image)
            h, w = frame.shape[:2]
            img_area = h * w
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            # Garder petits objets (insectes typiquement < 3% de l'image)
            mask = areas < img_area * 0.03

        dets = np.column_stack([xyxy[mask], confs[mask]])
        return dets

    def process_video(
        self,
        video_path: str,
        annotate: bool = True,
        line_position: Optional[float] = None,
    ) -> CountResult:
        """
        Traite une vidéo complète et retourne le CountResult.
        line_position : 0.0 = haut, 1.0 = bas (défaut = LINE_POS env var)
        """
        line_pos = line_position if line_position is not None else LINE_POS

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")

        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
        line_y = int(H * line_pos)

        # Writer annoté
        out_path = None
        writer   = None
        if annotate:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            out_path = tmp.name
            tmp.close()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

        sort_tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=0.3)

        # Suivi de position précédente par ID
        prev_y: dict[int, float] = {}   # {track_id: centre_y_frame_précédente}
        counted_ids = set()             # IDs déjà comptés (évite double-comptage)

        bee_in   = 0
        bee_out  = 0
        frame_n  = 0
        t_start  = time.perf_counter()
        per_frame = []

        # Couleurs
        COL_LINE   = (0, 200, 255)   # jaune-orange
        COL_IN     = (50, 220, 50)   # vert
        COL_OUT    = (50, 50, 230)   # rouge
        COL_TRACK  = (200, 170, 50)  # ambre

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_n += 1

            dets = self._detect_frame(frame)

            if len(dets) > 0:
                tracks = sort_tracker.update(dets)
            else:
                tracks = sort_tracker.update()

            for trk in tracks:
                x1, y1, x2, y2, tid = trk
                tid = int(tid)
                cx  = (x1 + x2) / 2
                cy  = (y1 + y2) / 2

                # Détection de traversée de la ligne
                if tid in prev_y and tid not in counted_ids:
                    py = prev_y[tid]
                    # Entrée : vient du bas (cy > line_y) → monte (cy < line_y → IN)
                    # Sortie : vient du haut → descend (py < line_y → OUT)
                    if py > line_y and cy <= line_y:
                        bee_in += 1
                        counted_ids.add(tid)
                        color_flash = COL_IN
                    elif py < line_y and cy >= line_y:
                        bee_out += 1
                        counted_ids.add(tid)
                        color_flash = COL_OUT
                    else:
                        color_flash = COL_TRACK
                else:
                    color_flash = COL_TRACK

                prev_y[tid] = cy

                if annotate and writer:
                    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color_flash, 2)
                    label = f"#{tid}"
                    cv2.putText(frame, label, (x1i, y1i - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_flash, 1)

            # Annotation de la frame
            if annotate and writer:
                # Ligne de comptage
                cv2.line(frame, (0, line_y), (W, line_y), COL_LINE, 2)
                cv2.putText(frame, "LIGNE DE COMPTAGE",
                            (10, line_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_LINE, 1)

                # Overlay compteurs
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (260, 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame, f"IN  : {bee_in:4d}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, COL_IN, 2)
                cv2.putText(frame, f"OUT : {bee_out:4d}", (10, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, COL_OUT, 2)
                cv2.putText(frame, f"Frame {frame_n}", (W - 120, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                writer.write(frame)

            if frame_n % 30 == 0:
                per_frame.append({
                    "frame": frame_n,
                    "in_cumul":  bee_in,
                    "out_cumul": bee_out,
                })
                logger.debug(f"Frame {frame_n}: IN={bee_in} OUT={bee_out} tracks={len(tracks)}")

        cap.release()
        if writer:
            writer.release()

        elapsed = time.perf_counter() - t_start
        fps_proc = frame_n / elapsed if elapsed > 0 else 0

        logger.info(
            f"Traitement terminé : {frame_n} frames | {elapsed:.1f}s | "
            f"{fps_proc:.1f} fps | IN={bee_in} OUT={bee_out}"
        )

        return CountResult(
            bee_in=bee_in,
            bee_out=bee_out,
            total_tracks=int(sort_tracker.frame_count),
            frames=frame_n,
            duration_s=round(elapsed, 2),
            fps=round(fps_proc, 1),
            line_y_px=line_y,
            annotated_video_path=out_path,
            per_frame=per_frame,
        )
