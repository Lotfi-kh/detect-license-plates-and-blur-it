import cv2
import numpy as np
from ultralytics import YOLO
import os


def flou_auto(ch_img, ch_mdl):

    # verifier ida l fichier ta3 l model kayen
    if not os.path.exists(ch_mdl):
        print(f"Erreur: l model makachou f: {ch_mdl}")
        return None

    print(f"Ncharjiw l model mn {ch_mdl}")
    try:
        mdl = YOLO(ch_mdl)  # chargement
    except Exception as e:
        print(f"Probleme f chargement ta3 YOLO ")
        return None

    img = cv2.imread(ch_img)
    if img is None:
        print(f"Erreur: L'image mate9rach {ch_img}")
        return None

    seuil = 0.4
    cls_id = 0  # id ta3 la plaque

    # detection (pas besoin de verbose)
    res = mdl(img, verbose=False, classes=[cls_id])
    boxes = res[0].boxes
    cpt = 0  # compteur

    h, w, _ = img.shape

    for i in range(len(boxes)):
        conf = boxes[i].conf.item()

        if conf > seuil:
            # njibou les coordonees xyxy
            coords = boxes[i].xyxy.cpu().numpy().squeeze()
            x1, y1, x2, y2 = coords.astype(int)

            # verification bach manakhrojch 3la lcadre (boundary check)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # na7akmou ghir la zone li nas7a9oha
            zone = img[y1:y2, x1:x2]

            if zone.shape[0] > 0 and zone.shape[1] > 0:
                # calcul dynamique ta3 k (kernel) bach l flou yji mlih
                k = (x2 - x1) // 5

                # lazem ykon nombre impair sinon erreur cv2
                if k % 2 == 0:
                    k += 1

                # minimum size bach ma ykonch sghir bzaf
                k = max(21, k)

                img_flou = cv2.GaussianBlur(zone, (k, k), 0)
                img[y1:y2, x1:x2] = img_flou

                cpt += 1

                # rectangle vert
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    txt = f"Detecte: {cpt} plaques"
    cv2.putText(img, txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    print(f"C bon {cpt} plaques trouvées.")
    return img
