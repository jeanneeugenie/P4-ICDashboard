import io
import random
import time

import grpc
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from proto import dashboard_pb2, dashboard_pb2_grpc
from training.model import SimpleCNN


# --- CONFIG ---
BATCH_SIZE = 64
NUM_EPOCHS = 1          # keep small at first
NUM_TILES = 16          # tiles per batch for dashboard
DATA_ROOT = "./data"    # where CIFAR-10 will be downloaded


def tensor_to_png_bytes(tensor):
    """
    Convert a (C, H, W) tensor in [0, 1] to PNG bytes.
    """
    buf = io.BytesIO()
    save_image(tensor, buf, format="PNG")
    return buf.getvalue()


def make_dataloader():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # -> [0,1] float, shape (C,H,W)
    ])

    train_dataset = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    return loader, train_dataset.classes  # label names


def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train] Using device:", device)

    # data
    loader, label_names = make_dataloader()

    # model
    model = SimpleCNN(num_classes=len(label_names)).to(device)

    # loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # gRPC channel + stub
    channel = grpc.insecure_channel("localhost:50051")
    stub = dashboard_pb2_grpc.DashboardServiceStub(channel)

    dashboard_online = True  # will flip to False if we lose connection

    # helper to send a single batch (as a 1-element stream)
    def send_batch_to_dashboard(batch_msg):
        nonlocal dashboard_online
        if not dashboard_online:
            return

        try:
            # iter([batch_msg]) creates a one-element "stream"
            ack = stub.StreamTraining(iter([batch_msg]))
            # you can optionally check ack.ok here
        except grpc.RpcError as e:
            print("[train] Dashboard connection lost, disabling streaming.")
            print("        Error:", e)
            dashboard_online = False

    # Optional thing but here's quick Ping (also fault-tolerant)
    try:
        hb = dashboard_pb2.Heartbeat(timestamp_ms=int(time.time() * 1000))
        ack = stub.Ping(hb)
        print("[train] Ping ack:", ack.ok, ack.message)
    except grpc.RpcError as e:
        print("[train] Could not reach dashboard on Ping, will train offline.")
        print("        Error:", e)
        dashboard_online = False

    iteration = 0
    print("[train] Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        print(f"[train] Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, labels in loader:
            start_time = time.time()

            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # preds
            preds = outputs.argmax(dim=1)

            # --- Build TrainingBatch message ---
            batch_msg = dashboard_pb2.TrainingBatch()
            batch_msg.iteration = iteration
            batch_msg.loss = float(loss.item())
            batch_msg.fps = 0.0  # dashboard computes its own FPS

            # choose up to NUM_TILES images
            batch_size = images.size(0)
            num_tiles = min(NUM_TILES, batch_size)
            indices = random.sample(range(batch_size), k=num_tiles)

            images_cpu = images.detach().cpu()
            labels_cpu = labels.detach().cpu()
            preds_cpu = preds.detach().cpu()

            for idx in indices:
                img_msg = batch_msg.images.add()
                img_msg.id = idx
                img_msg.true_label = label_names[int(labels_cpu[idx])]
                img_msg.predicted_label = label_names[int(preds_cpu[idx])]
                img_msg.image_data = tensor_to_png_bytes(images_cpu[idx])

            print(
                f"[train] iter={iteration}, loss={batch_msg.loss:.4f}, "
                f"tiles={num_tiles}, dashboard_online={dashboard_online}"
            )

            # send to dashboard (if still online)
            send_batch_to_dashboard(batch_msg)

            # simulate a bit of delay (optional, to control pace)
            elapsed = time.time() - start_time
            target_step_time = 0.1  # seconds
            if elapsed < target_step_time:
                time.sleep(target_step_time - elapsed)

            iteration += 1

    print("[train] Training finished.")


if __name__ == "__main__":
    main()
