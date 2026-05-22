import cv2
import numpy as np
from ultralytics import YOLO

# --- Config ---
ch_mdl = "train/weights/best.pt"
ch_vid = "video2.mp4"
seuil = 0.5
id_cible = 0  # id ta3 l objet li rana n7awssou 3lih

coul = (0, 255, 0)

print(f"Chargement du modele: {ch_mdl}...")
mdl = YOLO(ch_mdl)

cap = cv2.VideoCapture(ch_vid)

if not cap.isOpened():
    print(f"Erreur: Impossible d'ouvrir la video {ch_vid}")
    exit()

print("Traitement en cours... Appuyez sur 'q' pour quitter.")

while True:
    ret, img = cap.read()
    if not ret:
        break  # video khlaset

    # optimisation: classes=[id] bach nreb7o chwiya wa9t
    res = mdl(img, verbose=False, classes=[id_cible])

    boxs = res[0].boxes
    cpt = 0  # compteur

    for i in range(len(boxs)):
        conf = boxs[i].conf.item()

        if conf > seuil:
            # njibou les coordonnees
            coords = boxs[i].xyxy.cpu().numpy().squeeze()
            x1, y1, x2, y2 = coords.astype(int)

            # --- Partie Floutage ---

            # na7akmou ghir la zone (ROI)
            zone = img[y1:y2, x1:x2]

            # verifier ida la zone rahi valide bach ma yasrach crash
            if zone.shape[0] > 0 and zone.shape[1] > 0:

                # calcul dynamique ta3 l kernel (k)
                # koul ma l objet ykon kbir, l flou yzid
                k = (x2 - x1) // 5

                # lazem ykon nombre impair sinon OpenCV may9bloch
                if k % 2 == 0:
                    k += 1

                # minimum size bach l flou yban mli7
                k = max(21, k)

                # application ta3 gaussian blur
                zone_flou = cv2.GaussianBlur(zone, (k, k), 0)

                # nraj3ou la zone floué f blast'ha
                img[y1:y2, x1:x2] = zone_flou

            cpt += 1

    # affichage du nombre
    cv2.putText(
        img,
        f"Total: {cpt}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        coul,
        2,
    )

    cv2.imshow("Detection & Flou", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
