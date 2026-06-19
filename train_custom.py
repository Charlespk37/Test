"""
Fine-tuning YOLOv8 sur un dataset d'abeilles
=============================================
Utiliser ce script UNE FOIS pour entraîner un modèle spécialisé "bee".
Résultat → models/best.pt (utilisé automatiquement par bee_counter.py)

Dataset recommandé :
  • Roboflow "Bee Detection" (gratuit, ~3000 images annotées YOLO)
    https://universe.roboflow.com/mel-bees/bee-detection-ijmyp
  • Ou constituer le tien : filmer l'entrée de la ruche, annoter avec
    LabelImg ou Roboflow Annotate, exporter en format YOLOv8.

Usage :
  python train_custom.py --data /chemin/vers/data.yaml --epochs 50

Structure data.yaml (générée par Roboflow) :
  train: ../train/images
  val:   ../valid/images
  nc: 1
  names: ['bee']
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def train(data_yaml: str, epochs: int = 50, imgsz: int = 640, batch: int = 16):
    model = YOLO("yolov8n.pt")   # partir du nano pre-trained COCO

    results = model.train(
        data     = data_yaml,
        epochs   = epochs,
        imgsz    = imgsz,
        batch    = batch,
        name     = "ruche_bees",
        patience = 15,           # early stopping
        device   = "cpu",        # remplacer par "0" si GPU disponible
        augment  = True,
        mosaic   = 0.5,
        flipud   = 0.3,
        fliplr   = 0.5,
        degrees  = 10,
        scale    = 0.5,
        project  = "runs/train",
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    dest = Path("models/best.pt")
    dest.parent.mkdir(exist_ok=True)
    import shutil
    shutil.copy(best, dest)
    print(f"\n✅ Modèle fine-tuné copié vers : {dest}")
    print("   Relance le backend — bee_counter.py l'utilisera automatiquement.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True, help="Chemin vers data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--batch",  type=int, default=16)
    args = parser.parse_args()
    train(args.data, args.epochs, args.imgsz, args.batch)
