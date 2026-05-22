import cv2

from ultralytics import YOLO
import os


class TraitementVid:
    def __init__(self, ch_mdl):
        self.ch_mdl = ch_mdl
        self.mdl = None
        self.init_mdl()

    def init_mdl(self):
        # verifier ida l fichier ta3 l model kayen
        if not os.path.exists(self.ch_mdl):
            print(f"Erreur: Fichier introuvable: {self.ch_mdl}")
            return

        print(f"Chargement du modele: {self.ch_mdl}...")
        try:
            self.mdl = YOLO(self.ch_mdl)
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")

    def gen_video(self, src_vid):
        """
        Generateur li ymed les frames traitees mn video wla camera.
        src_vid: chemin fichier (str) wla ID camera (int).
        """
        if self.mdl is None:
            print("Le modele n'est pas charge.")
            return

        cam = cv2.VideoCapture(src_vid)

        if not cam.isOpened():
            print(f"Erreur: Impossible d'ouvrir la source {src_vid}")
            return

        seuil = 0.5
        cls_id = 0  # id ta3 l objet cible

        while True:
            ret, img = cam.read()
            if not ret:
                break  # khlaset l video

            # Lancement detection
            res_yolo = self.mdl(img, verbose=False, classes=[cls_id])
            boxs = res_yolo[0].boxes
            cpt = 0

            h, w = img.shape[:2]

            for i in range(len(boxs)):
                conf = boxs[i].conf.item()

                if conf > seuil:
                    coords = boxs[i].xyxy.cpu().numpy().squeeze()
                    x1, y1, x2, y2 = coords.astype(int)

                    # Correction des bords (clamp) bach ma nkhrojch 3la lcadre
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    zone = img[y1:y2, x1:x2]

                    if zone.shape[0] > 0 and zone.shape[1] > 0:
                        # calcul dynamique ta3 flou
                        k = (x2 - x1) // 5

                        # lazem ykon impair
                        if k % 2 == 0:
                            k += 1

                        # minimum size
                        k = max(21, k)

                        img_flou = cv2.GaussianBlur(zone, (k, k), 0)
                        img[y1:y2, x1:x2] = img_flou

                        cpt += 1

            # Affichage du compteur foug l video
            cv2.putText(
                img,
                f"Total: {cpt}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Yield l image lel GUI (RGB lazem)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img_rgb

        cam.release()
