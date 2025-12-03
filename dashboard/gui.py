import io
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from dashboard import server as server_mod


# --- GUI CONFIG ---
TILE_ROWS = 4
TILE_COLS = 4
TILE_SIZE = (128, 128)  # width, height in pixels
REFRESH_MS = 100        # how often it check for new batches


class DashboardGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier Dashboard")

        # Keep references to PhotoImage objects so they don't get GC'd
        self.tile_images = [None] * (TILE_ROWS * TILE_COLS)

        # ---- Layout ----
        main_frame = ttk.Frame(root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        # Frames
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.grid(row=0, column=0, padx=5, pady=5)

        self.text_frame = ttk.Frame(main_frame)
        self.text_frame.grid(row=1, column=0, padx=5, pady=5)

        self.info_frame = ttk.Frame(main_frame)
        self.info_frame.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        # 16 image labels
        self.image_labels = []
        for r in range(TILE_ROWS):
            for c in range(TILE_COLS):
                lbl = ttk.Label(self.image_frame)
                lbl.grid(row=r, column=c, padx=2, pady=2)
                self.image_labels.append(lbl)

        # 16 text labels (pred / true)
        self.text_labels = []
        for r in range(TILE_ROWS):
            for c in range(TILE_COLS):
                lbl = ttk.Label(self.text_frame, text="pred: ? / true: ?")
                lbl.grid(row=r, column=c, padx=2, pady=2)
                self.text_labels.append(lbl)

        # Info label for iteration / loss
        self.info_label = ttk.Label(self.info_frame, text="iter: -  loss: -")
        self.info_label.grid(row=0, column=0, sticky="w")

        # Start periodic refresh
        self.schedule_refresh()

    def schedule_refresh(self):
        self.refresh_from_state()
        self.root.after(REFRESH_MS, self.schedule_refresh)

    def refresh_from_state(self):
        batch = server_mod.state.get()
        if batch is None:
            return

        # Update info text
        self.info_label.config(
            text=f"iter: {batch.iteration}   loss: {batch.loss:.4f}"
        )

        # Update 16 tiles
        for i, img_msg in enumerate(batch.images):
            if i >= TILE_ROWS * TILE_COLS:
                break

            img = Image.open(io.BytesIO(img_msg.image_data))
            img = img.resize(TILE_SIZE)
            photo = ImageTk.PhotoImage(img)

            self.image_labels[i].config(image=photo)
            self.image_labels[i].image = photo     # keep reference

            self.text_labels[i].config(
                text=f"pred: {img_msg.predicted_label} / true: {img_msg.true_label}"
            )

        # If fewer than 16, clear the rest
        for j in range(len(batch.images), TILE_ROWS * TILE_COLS):
            self.image_labels[j].config(image="")
            self.image_labels[j].image = None
            self.text_labels[j].config(text="(empty)")


def main():
    # Start gRPC server in background
    server_mod.start_server_in_thread()

    # Start Tkinter GUI
    root = tk.Tk()
    gui = DashboardGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
