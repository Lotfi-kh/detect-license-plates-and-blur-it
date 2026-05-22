import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time

# n7adrou les arguments bach ntemchou l script
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Chemin ta3 l model (ex: best.pt)")
parser.add_argument(
    "--source", required=True, help="Fichier video wla numero ta3 cam (0, 1...)"
)
args = parser.parse_args()

ch_mdl = args.model
src = args.source

# ncharjiw l model, pas la peine ndirou try/except hna, ida maamchach yhabes direct
mdl = YOLO(ch_mdl)

# nchoufou ida la source hiya numero (webcam) wla fichier
# kima hakda code yji khfif
if src.isdigit():
    src = int(src)


cap = cv2.VideoCapture(src)


coul = [(164, 120, 87)]


while True:

    t_deb = time.perf_counter()

    ret, img = cap.read()

    # if problem
    if not ret:
        break

    # detection YOLO (sans blabla verbose)
    res_yolo = mdl(img, verbose=False)
    boxs = res_yolo[0].boxes
    cpt = 0

    # ndourou 3la ga3 wach lga l model
    for i in range(len(boxs)):

        # coordonnees w conf
        xyxy = boxs[i].xyxy.cpu().numpy().squeeze()
        x1, y1, x2, y2 = xyxy.astype(int)
        conf = boxs[i].conf.item()
        cls_id = int(boxs[i].cls.item())

        # filtrage b 0.5 (seuil)
        if conf > 0.5:
            c = coul[cls_id % len(coul)]

            # rsm rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)

            # ktiba ta3 l class w confidence
            txt = f"{mdl.names[cls_id]}: {int(conf*100)}%"
            t_txt, base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_txt = max(y1, t_txt[1] + 10)

            # fond sghir bach lktiba taban
            cv2.rectangle(
                img,
                (x1, y_txt - t_txt[1] - 10),
                (x1 + t_txt[0], y_txt + base - 10),
                c,
                cv2.FILLED,
            )
            cv2.putText(
                img, txt, (x1, y_txt - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

            cpt += 1

    # calcul rapide ta3 fps
    t_fin = time.perf_counter()
    fps = 1 / (t_fin - t_deb) if (t_fin - t_deb) > 0 else 0

    # affichage f l ecran
    cv2.putText(
        img,
        f"FPS: {fps:.2f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        img, f"Total: {cpt}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )

    cv2.imshow("Test PFE - PC", img)

    # 'q' bach nkhourjou
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

# netoyage
cap.release()
cv2.destroyAllWindows()
