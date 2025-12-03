# P4 Image Classifier Dashboard

This project implements a real-time dashboard for CNN training using gRPC, as required by the
**Image Classifier Dashboard** problem set. The dashboard visualizes:

- Input image batch (16 tiles)
- Prediction vs ground-truth labels
- Training loss per iteration
- Frame rate (FPS) and display latency

The dashboard communicates with an external ML training application (PyTorch) via gRPC and
demonstrates limited fault tolerance.   

---

## 0. Deliverables Overview

This submission contains **two ways** to run the dashboard:

1. **Standalone Windows application** – `ICDashboard.exe`  
   
2. **From source (Python)** – this git repo  

---

## 1. Running the Dashboard (Recommended: .EXE)

### 1.1. Option A – Standalone Windows EXE

1. Locate `ICDashboard.exe`.

2. Double-click `ICDashboard.exe`.

3. You should see the dashboard window open with:
   - 4×4 image tiles
   - Labels panel
   - Loss plot on the right
   - FPS and latency info at the bottom

4. In a separate terminal (with Python + dependencies installed), run one of:
   - `python -m training.train_client` – fake random data
   - `python -m training.train_real` – real CIFAR-10 training

The exe internally runs the same code as `dashboard.gui` (gRPC server + Tkinter GUI).

> **Note:** The EXE does not include the training code; that still runs via Python.
> This matches the spec: dashboard is a standalone Windows application, training app is external.   

---

## 2. Running From Source (Python)

If you’re cloning the git repo and **don’t** have the exe (or want to hack on the code), follow these steps.

### 2.1. Setup virtual environment

From the project root:

```bash
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install grpcio grpcio-tools protobuf
pip install pillow matplotlib
