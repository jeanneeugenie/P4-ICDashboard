#Simulates training script.
#Currently sends fake batches (random images + labels) for testing
import io
import random
import time

import grpc
import torch
from torchvision.utils import save_image

from proto import dashboard_pb2, dashboard_pb2_grpc


LABELS = ["cat", "dog", "car", "plane"]  # example label names


def tensor_to_png_bytes(tensor):
    """
    Convert a (C, H, W) tensor in [0, 1] to PNG bytes.
    """
    buf = io.BytesIO()
    save_image(tensor, buf, format="PNG")
    return buf.getvalue()


def generate_fake_batches(num_batches=10, batch_size=32):
    """
    Generator that yields TrainingBatch messages with random data.
    """
    for iteration in range(num_batches):
        # fake images: random noise
        images = torch.rand(batch_size, 3, 64, 64)
        labels = torch.randint(0, len(LABELS), (batch_size,))
        preds = torch.randint(0, len(LABELS), (batch_size,))

        batch_msg = dashboard_pb2.TrainingBatch()
        batch_msg.iteration = iteration
        batch_msg.loss = random.random()
        batch_msg.fps = 0.0  # we'll fill real FPS later

        # pick up to 16 indices to send to the dashboard
        num_tiles = min(16, batch_size)
        indices = random.sample(range(batch_size), k=num_tiles)

        for idx in indices:
            img_msg = batch_msg.images.add()
            img_msg.id = idx
            img_msg.true_label = LABELS[labels[idx]]
            img_msg.predicted_label = LABELS[preds[idx]]
            img_msg.image_data = tensor_to_png_bytes(images[idx])

        print(f"[client] sending batch iter={iteration}, tiles={num_tiles}")
        yield batch_msg
        time.sleep(0.3)  # simulate training time per iteration


def main():
    channel = grpc.insecure_channel("localhost:50051")
    stub = dashboard_pb2_grpc.DashboardServiceStub(channel)

    # Optional: test Ping first
    hb = dashboard_pb2.Heartbeat(timestamp_ms=int(time.time() * 1000))
    ack = stub.Ping(hb)
    print("[client] Ping ack:", ack.ok, ack.message)

    # Send the stream of batches
    ack2 = stub.StreamTraining(generate_fake_batches())
    print("[client] StreamTraining finished:", ack2.ok, ack2.message)


if __name__ == "__main__":
    main()

