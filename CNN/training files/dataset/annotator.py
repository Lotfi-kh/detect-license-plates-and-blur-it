import matplotlib.pyplot as plt
from PIL import Image
import os


def annotate_image(img_path, save_path):
    img = Image.open(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)

    coords = []

    def onclick(event):
        if event.xdata and event.ydata:
            coords.append((int(event.xdata), int(event.ydata)))
            if len(coords) == 2:
                x0, y0 = coords[0]
                x1, y1 = coords[1]
                ax.add_patch(
                    plt.Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        edgecolor="red",
                        facecolor="none",
                        linewidth=2,
                    )
                )
                plt.draw()
                with open(save_path, "w") as f:
                    f.write(f"{int(0)} {x0} {y0} {x1} {y1}\n")
                print("Annotation saved:", save_path)
                fig.canvas.mpl_disconnect(cid)

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


COUNTER_FILE = "counter.txt"


def read_counter():

    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "r") as file:
            try:
                return int(file.read().strip())
            except ValueError:
                return 0
    return 0


def increment_counter():
    count = read_counter()
    count += 1
    if count > 101:
        count = 1
    with open(COUNTER_FILE, "w") as file:
        file.write(str(count))
    return count


count = increment_counter()

# Example usage
if __name__ == "__main__":

    annotate_image(
        f"dataset/data/images/train/a{count}.jpeg", f"dataset/data/labels/a{count}.txt"
    )
