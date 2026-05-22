from ultralytics import YOLO
import torch

if __name__ == "__main__":
    print("Demarrage de l'entrainement...")

    # verifier ida GPU yamchi
    if torch.cuda.is_available():
        print(f"GPU detecte: {torch.cuda.get_device_name(0)}")
    else:
        print("Attention: Rak temchi CPU, rah ytol bzaf.")

    # 1. Chargement du model de base (small)
    mdl = YOLO("yolo11s.pt")

    # 2. Lancement du training
    res = mdl.train(
        data="dataset/data.yaml",  # fichier config
        epochs=100,
        patience=15,  # yhabes ida makach amelioration
        # --- Config special 1660 Ti ---
        imgsz=1024,  # resolution tal3a bach ychouf mlih les plaques
        batch=4,  # batch sghir bach ma yasserch OOM (Out Of Memory)
        workers=2,  # bach windows ma ybeuguich
        device=0,  # GPU ta3na
        project="yolo_plaques",  # dossier principal
        name="test1",  # sous-dossier
        exist_ok=True,  # ecrasi ancien dossier normal
    )

    print("\nC'est bon")
    print("Le fichier best.pt rah f: yolo_plaques/test1/weights/best.pt")
