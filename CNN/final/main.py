import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2
import pic
import load_video
import life_video
import os
import sys
import time


def chemin_src(rel_path):
    # hada bach yjib chemin swaswa f pyinstaller wala dev
    try:
        base = sys._MEIPASS
    except Exception:
        base = os.path.abspath(".")
    return os.path.join(base, rel_path)


ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


class AppPlaque(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- CONFIG ---
        self.ch_mdl = chemin_src("train/weights/best.pt")

        self.proc_vid = load_video.TraitementVid(self.ch_mdl)
        self.proc_live = life_video.TraitementVid(self.ch_mdl)

        self.gen_cours = None  # le generateur actif (video ou live)
        self.en_marche = False

        # Reglage Fenetre
        self.title("Detection Plaques & Flou")
        self.geometry("1100x650")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- BARRE A GAUCHE (SIDEBAR) ---
        self.frame_gauche = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.frame_gauche.grid(row=0, column=0, sticky="nsew")
        self.frame_gauche.grid_rowconfigure(5, weight=1)
        self.frame_gauche.grid_columnconfigure(2, weight=1)

        self.lbl_logo = ctk.CTkLabel(
            self.frame_gauche,
            text="App\nAutoFlou",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.lbl_logo.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Les Boutons
        self.btn_img = ctk.CTkButton(
            self.frame_gauche, text="Charger Image", command=self.charger_img
        )
        self.btn_img.grid(row=1, column=0, padx=20, pady=10)

        self.btn_vid = ctk.CTkButton(
            self.frame_gauche, text="Charger Video", command=self.lancer_video_fichier
        )
        self.btn_vid.grid(row=2, column=0, padx=20, pady=10)

        self.btn_live = ctk.CTkButton(
            self.frame_gauche, text="Live Camera", command=self.mode_live
        )
        self.btn_live.grid(row=3, column=0, padx=20, pady=10)

        # --- CONTENU PRINCIPAL ---
        self.conteneur = ctk.CTkFrame(self, fg_color="transparent")
        self.conteneur.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.conteneur.grid_rowconfigure(0, weight=1)
        self.conteneur.grid_columnconfigure(0, weight=1)

        # On prepare les vues
        self.creer_vue_img()
        self.creer_vue_vid()
        self.afficher_vue_img()  # Par defaut

    def creer_vue_img(self):
        self.vue_img = ctk.CTkFrame(self.conteneur, fg_color="transparent")
        self.vue_img.grid_columnconfigure((0, 1), weight=1)
        self.vue_img.grid_rowconfigure(0, weight=1)

        # Original
        self.cadre_orig = ctk.CTkFrame(self.vue_img)
        self.cadre_orig.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.lbl_titre1 = ctk.CTkLabel(
            self.cadre_orig, text="Image Originale", font=("Arial", 14, "bold")
        )
        self.lbl_titre1.pack(pady=10)
        self.disp_orig = ctk.CTkLabel(self.cadre_orig, text="[Aucune Image]")
        self.disp_orig.pack(expand=True, fill="both", padx=10, pady=10)

        # Resultat
        self.cadre_res = ctk.CTkFrame(self.vue_img)
        self.cadre_res.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.lbl_titre2 = ctk.CTkLabel(
            self.cadre_res, text="Resultat", font=("Arial", 14, "bold")
        )
        self.lbl_titre2.pack(pady=10)
        self.disp_res = ctk.CTkLabel(self.cadre_res, text="[En attente]")
        self.disp_res.pack(expand=True, fill="both", padx=10, pady=10)

    def creer_vue_vid(self):
        self.vue_vid = ctk.CTkFrame(self.conteneur, fg_color="transparent")
        self.vue_vid.grid_columnconfigure(0, weight=1)
        self.vue_vid.grid_rowconfigure((1, 2), weight=1)

        # Controles (caché au debut)
        self.barre_ctrl = ctk.CTkFrame(self.vue_vid, height=50)
        self.barre_ctrl.grid(row=0, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.lbl_src = ctk.CTkLabel(self.barre_ctrl, text="Source:")
        self.lbl_src.pack(side="left", padx=10)

        # Choix PC vs Raspberry
        self.choix_src = ctk.CTkSegmentedButton(
            self.barre_ctrl,
            values=["PC (USB)", "Raspberry Pi"],
            command=self.changement_src,
        )
        self.choix_src.set("PC (USB)")
        self.choix_src.pack(side="left", padx=10)

        self.btn_stop = ctk.CTkButton(
            self.barre_ctrl,
            text="Arreter",
            fg_color="red",
            command=self.arret_video,
        )
        self.btn_stop.pack(side="right", padx=10)

        # Zone Video
        self.cadre_ecran = ctk.CTkFrame(self.vue_vid)
        self.cadre_ecran.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.lbl_ecran_titre = ctk.CTkLabel(
            self.cadre_ecran, text="Lecteur Video", font=("Arial", 14, "bold")
        )
        self.lbl_ecran_titre.pack(pady=10)

        self.disp_vid = ctk.CTkLabel(self.cadre_ecran, text="[Video Vide]")
        self.disp_vid.pack(expand=True, fill="both", padx=10, pady=10)

    # --- LOGIQUE VIDEO ---

    def mode_live(self):
        self.arret_video()
        self.afficher_vue_vid("Mode Camera Live")
        self.barre_ctrl.grid(row=0, column=0, sticky="ew")  # Afficher les boutons

        # Demarrer par defaut sur PC
        self.changement_src("PC")

    def lancer_video_fichier(self):
        self.arret_video()
        self.barre_ctrl.grid_forget()  # Pas besoin de choix PC/Pi pour un fichier

        ch_fich = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")])
        if ch_fich:
            self.afficher_vue_vid("Lecture Fichier")
            # Appel a gen_video de load_video.py (fichier 3)
            self.gen_cours = self.proc_vid.gen_video(ch_fich)
            self.en_marche = True
            self.boucle_video()
        else:
            self.afficher_vue_img()  # Annulé

    def changement_src(self, val):
        self.arret_video()

        if val == "PC (USB)":
            print("Lancement Camera USB...")
            # Appel a gen_video de life_video.py (fichier 2)
            self.gen_cours = self.proc_live.gen_video(typ_src="usb", ch_src=0)
        else:
            print("Lancement Camera Raspberry...")
            self.gen_cours = self.proc_live.gen_video(typ_src="picamera")

        self.en_marche = True
        self.boucle_video()

    def boucle_video(self):
        if not self.en_marche or self.gen_cours is None:
            return

        # temps debut du traitement
        t_debut = time.time()

        try:
            # njibou la frame next
            frame = next(self.gen_cours)

            # Mise a jour GUI
            img_pil = Image.fromarray(frame)
            img_ctk = ctk.CTkImage(
                light_image=img_pil, dark_image=img_pil, size=(640, 480)
            )
            self.disp_vid.configure(image=img_ctk, text="")
            self.disp_vid.image = img_ctk

            # calcul temps traitement
            t_traitement = time.time() - t_debut

            # fps cible 30, donc delai ideal = 33ms
            delai_ideal = 33
            delai_reste = max(1, int(delai_ideal - (t_traitement * 1000)))

            # Rappel apres delai ajusté
            self.after(delai_reste, self.boucle_video)

        except StopIteration:
            self.arret_video()
            self.disp_vid.configure(text="Fin du flux")
        except Exception as e:
            print(f"Erreur Flux: {e}")
            self.arret_video()

    def arret_video(self):
        self.en_marche = False
        self.gen_cours = None
        self.disp_vid.configure(image="", text="[Arrete]")

    # --- LOGIQUE IMAGE ---
    def charger_img(self):
        self.arret_video()
        self.afficher_vue_img()

        ch_fich = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if ch_fich:
            img_cv = cv2.imread(ch_fich)
            self.afficher_ds_gui(img_cv, self.disp_orig)
            self.disp_res.configure(text="Traitement...", image="")
            self.update()

            # Appel a flou_auto de pic.py (fichier 1)
            res_cv = pic.flou_auto(ch_fich, self.ch_mdl)
            if res_cv is not None:
                self.afficher_ds_gui(res_cv, self.disp_res)

    def afficher_ds_gui(self, img, lbl_wid):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(480, 360))
        lbl_wid.configure(image=img_ctk, text="")
        lbl_wid.image = img_ctk

    # --- CHANGEMENT DE VUE ---
    def afficher_vue_img(self):
        self.vue_vid.grid_forget()
        self.vue_img.grid(row=0, column=0, sticky="nsew")

    def afficher_vue_vid(self, titre):
        self.vue_img.grid_forget()
        self.vue_vid.grid(row=0, column=0, sticky="nsew")
        self.lbl_ecran_titre.configure(text=titre)


if __name__ == "__main__":
    app = AppPlaque()
    # zoomer l fenetre direct
    app.state("zoomed")
    app.mainloop()
