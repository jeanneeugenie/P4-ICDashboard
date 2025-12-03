import io
import time
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from dashboard import server as server_mod


# --- GUI CONFIG ---
TILE_ROWS = 4
TILE_COLS = 4
TILE_SIZE = (128, 128)  # width, height in pixels
REFRESH_MS = 50         # GUI refresh period (ms)


class DashboardGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier Dashboard")

        # FPS tracking
        self.last_frame_time = None
        self.current_fps = 0.0

        # Keep references to PhotoImage objects so they don't get GC'd
        self.tile_images = [None] * (TILE_ROWS * TILE_COLS)

        # ---- Layout ----
        main_frame = ttk.Frame(root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        # Frames
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.grid(row=0, column=0, padx=5, pady=5, sticky="n")

        self.text_frame = ttk.Frame(main_frame)
        self.text_frame.grid(row=1, column=0, padx=5, pady=5, sticky="n")

        self.info_frame = ttk.Frame(main_frame)
        self.info_frame.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.grid(row=0, column=1, rowspan=3, padx=5, pady=5, sticky="nsew")

        main_frame.columnconfigure(0, weight=0)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

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

        # Info label for iteration / loss / FPS / latency
        self.info_label = ttk.Label(self.info_frame, text="iter: -  loss: -  fps: -  latency: - ms")
        self.info_label.grid(row=0, column=0, sticky="w")

        # ---- Matplotlib loss plot ----
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Loss")
        self.loss_line, = self.ax.plot([], [], lw=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Start periodic refresh
        self.schedule_refresh()

    def schedule_refresh(self):
        self.refresh_from_state()
        self.root.after(REFRESH_MS, self.schedule_refresh)

    def refresh_from_state(self):
        # FPS
        now = time.time()
        if self.last_frame_time is not None:
            dt = now - self.last_frame_time
            if dt > 0:
                # simple moving FPS: clamp to avoid insane spikes
                self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt)
        self.last_frame_time = now

        batch, history, last_update_time = server_mod.state.get_snapshot()
        if batch is None:
            return

        # latency = time since last batch arrived
        latency_ms = 0.0
        if last_update_time is not None:
            latency_ms = (now - last_update_time) * 1000.0

        # Update info text
        self.info_label.config(
            text=(
                f"iter: {batch.iteration}   "
                f"loss: {batch.loss:.4f}   "
                f"fps: {self.current_fps:5.1f}   "
                f"latency: {latency_ms:5.1f} ms"
            )
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

        # Update loss plot
        if history:
            iters = [p[0] for p in history]
            losses = [p[1] for p in history]
            self.loss_line.set_data(iters, losses)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw_idle()


def main():
    # Start gRPC server in background
    server_mod.start_server_in_thread()

    # Start Tkinter GUI
    root = tk.Tk()
    gui = DashboardGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
