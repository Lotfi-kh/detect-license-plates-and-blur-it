import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

# nchoufou ida rana nkhadmo b raspberry wala pc
try:
    from picamera2 import Picamera2

    RPI_DISPO = True
except ImportError:
    RPI_DISPO = False


class TraitementVid:
    def __init__(self, ch_mdl):
        self.ch_mdl = ch_mdl
        self.mdl = None
        self.noms = []
        self.init_mdl()

        # couleur ta3 l carree (boite)
        self.coul = (164, 120, 87)

    def init_mdl(self):
        # verifier ida l fichier kayen
        if not os.path.exists(self.ch_mdl):
            print(f"Erreur: Fichier introuvable: {self.ch_mdl}")
            return

        print(f"Chargement du modele: {self.ch_mdl}...")
        try:
            # pas la peine task="detect", yolo ya3raf wa7dou
            self.mdl = YOLO(self.ch_mdl)
            self.noms = self.mdl.names
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")

    def gen_video(self, typ_src, ch_src=None, res=None):

        if self.mdl is None:
            print("Le modele n'est pas charge.")
            return

        cam = None

        # Cas 1: Camera USB / Fichier video
        if typ_src == "usb":
            cam = cv2.VideoCapture(ch_src)
            if not cam.isOpened():
                print(f"Erreur: Impossible d'ouvrir la source {ch_src}")
                return

            if res:
                cam.set(3, res[0])
                cam.set(4, res[1])

        # Cas 2: Picamera (Raspberry Pi)
        elif typ_src == "picamera":
            if not RPI_DISPO:
                print("Erreur: Bibliotheque Picamera2 introuvable.")
                return

            cam = Picamera2()
            w, h = res if res else (640, 480)

            # configurer l camera bach tmd format mli7
            cfg = cam.create_video_configuration(
                main={"format": "XRGB8888", "size": (w, h)}
            )
            cam.configure(cfg)
            cam.start()

        # Variables pour hsab l FPS (vitesse)
        tab_fps = []
        fps_moy = 0

        # Boucle principale (inference)
        while True:
            t_debut = time.perf_counter()

            img = None

            # 9raya ta3 l frame
            if typ_src == "usb":
                ret, img = cam.read()
                if not ret:
                    break
            elif typ_src == "picamera":
                # njibou l image mn l buffer direct
                tmp_arr = cam.capture_array()
                img = cv2.cvtColor(np.copy(tmp_arr), cv2.COLOR_BGRA2BGR)

            if img is None:
                break

            # Resize ida lazm (sauf pour picamera deja reglee)
            if res and typ_src != "picamera":
                img = cv2.resize(img, res)

            # Lancement YOLO
            res_yolo = self.mdl(img, verbose=False)
            boxs = res_yolo[0].boxes
            cpt = 0  # compteur objets

            for i in range(len(boxs)):
                conf = boxs[i].conf.item()

                # filtre b seuil 0.5
                if conf > 0.5:
                    coords = boxs[i].xyxy.cpu().numpy().squeeze()
                    x1, y1, x2, y2 = coords.astype(int)

                    # ==================================================
                    # 1. PARTIE FLOU (BLUR)
                    # ==================================================
                    roi = img[y1:y2, x1:x2]

                    # Verifier beli l ROI machi vide
                    if roi.shape[0] > 0 and roi.shape[1] > 0:
                        # 7sab dynamique ta3 kernel bach ykon flou mlih
                        # koul ma l matricule kbira, koul ma l flou yzid
                        k_size = (x2 - x1) // 5

                        # lazem ykon nombre impair (ferdi)
                        if k_size % 2 == 0:
                            k_size += 1

                        # Minimum 21 bach maykonch flou khfif bzaf
                        k_size = max(21, k_size)

                        # Applique GaussianBlur
                        roi_flou = cv2.GaussianBlur(roi, (k_size, k_size), 0)

                        # Rja3 l partie flou f l image d'origine
                        img[y1:y2, x1:x2] = roi_flou

                    # ==================================================
                    # 2. DESSIN DU CADRE ET TEXTE (Foug l flou)
                    # ==================================================

                    # indice ta3 l classe
                    id_cls = int(boxs[i].cls.item())

                    # njibou l'esm (label)
                    if id_cls < len(self.noms):
                        txt_cls = self.noms[id_cls]
                    else:
                        txt_cls = str(id_cls)

                    # rsm rectangle
                    cv2.rectangle(img, (x1, y1), (x2, y2), self.coul, 2)

                    # etiquette foug rectangle
                    lbl = f"{txt_cls}: {int(conf*100)}%"
                    taille_lbl, base = cv2.getTextSize(
                        lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    y_lbl = max(y1, taille_lbl[1] + 10)

                    # background sghir wara l ktiba bach taban mli7
                    cv2.rectangle(
                        img,
                        (x1, y_lbl - taille_lbl[1] - 10),
                        (x1 + taille_lbl[0], y_lbl + base - 10),
                        self.coul,
                        cv2.FILLED,
                    )
                    cv2.putText(
                        img,
                        lbl,
                        (x1, y_lbl - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                    )

                    cpt += 1

            # Calcul FPS
            t_fin = time.perf_counter()
            duree = t_fin - t_debut

            # bach ma nti7ouch f division par zero
            fps = 1 / duree if duree > 0 else 0

            # moyenne lissée sur 200 frames
            if len(tab_fps) >= 200:
                tab_fps.pop(0)
            tab_fps.append(fps)
            fps_moy = np.mean(tab_fps)

            # Affichage info sur ecran
            cv2.putText(
                img,
                f"FPS: {fps_moy:.2f}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                img,
                f"Objets: {cpt}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Yield pour l'interface graphique (RGB oblig)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img_rgb

        # Nettoyage ki nkamlou
        if typ_src == "usb":
            cam.release()
        elif typ_src == "picamera":
            cam.stop()
