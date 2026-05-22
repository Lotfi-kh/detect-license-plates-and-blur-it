import yaml
import os

# ================= CONFIG =================
# Dossier win rah dataset ta3na (classes.txt, images, labels)
dossier = "dataset/data"
# ==========================================


def gen_yaml():
    # 1. 9raya ta3 classes.txt
    ch_cls = os.path.join(dossier, "classes.txt")
    noms = []

    if os.path.exists(ch_cls):
        with open(ch_cls, "r") as f:
            # n7iw les lignes vides bach ma yasserach decalage
            noms = [l.strip() for l in f.readlines() if l.strip()]
        print(f"Classes trouvées")
    else:
        print("Erreur: classes.txt makachou !")
        return

    # 2. La structure li yas7a9ha YOLO
    cfg = {
        "path": dossier,  # Chemin complet
        "train": "images/train",  # Chemin relatif vers train
        "val": "images/val",  # Chemin relatif vers val
        # Hna ndirou mapping bin l ID w l'esm (0: license_plate...)
        "names": {i: nom for i, nom in enumerate(noms)},
    }

    # 3. Sauvegarde (nkhabiwah f dossier li 9bel data)
    sortie = os.path.join(dossier, "..", "data.yaml")

    with open(sortie, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print("Ouiiiiii")


if __name__ == "__main__":
    gen_yaml()
