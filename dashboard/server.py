from concurrent import futures
import time
import threading

import grpc
from proto import dashboard_pb2, dashboard_pb2_grpc


class DashboardState:
    """
    Shared state where we keep the most recent batch and loss history.
    The GUI will read from here.
    """
    def __init__(self, max_points=500):
        self.lock = threading.Lock()
        self.last_batch = None
        self.loss_history = []        # list of (iteration, loss)
        self.last_update_time = None  # when we last received a batch (time.time())
        self.max_points = max_points

    def update(self, batch):
        now = time.time()
        with self.lock:
            self.last_batch = batch
            self.last_update_time = now
            self.loss_history.append((batch.iteration, batch.loss))
            # keep only the last N points
            if len(self.loss_history) > self.max_points:
                self.loss_history = self.loss_history[-self.max_points:]

    def get_snapshot(self):
        """
        Return a copy of the current state for the GUI: (batch, history, last_time).
        """
        with self.lock:
            batch = self.last_batch
            history = list(self.loss_history)
            last_time = self.last_update_time
        return batch, history, last_time


state = DashboardState()


class DashboardServiceImpl(dashboard_pb2_grpc.DashboardServiceServicer):
    def StreamTraining(self, request_iterator, context):
        """
        Receives a stream of TrainingBatch messages from the training client.
        """
        for batch in request_iterator:
            state.update(batch)
            print(
                f"[server] batch iter={batch.iteration}, "
                f"images={len(batch.images)}, loss={batch.loss:.4f}"
            )

        return dashboard_pb2.Ack(ok=True, message="stream ended on server")

    def Ping(self, request, context):
        """
        Simple ping RPC to test connectivity.
        """
        print(f"[server] Ping received, timestamp={request.timestamp_ms}")
        return dashboard_pb2.Ack(ok=True, message="pong from dashboard server")


def _make_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    dashboard_pb2_grpc.add_DashboardServiceServicer_to_server(
        DashboardServiceImpl(), server
    )
    server.add_insecure_port("[::]:50051")
    return server


def serve():
    """
    Blocking version: run only the gRPC server (no GUI).
    """
    server = _make_server()
    server.start()
    print("[server] Dashboard server listening on port 50051")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("[server] Shutting down...")
        server.stop(0)


def start_server_in_thread():
    """
    Non-blocking version: start gRPC server in a background thread.
    Used by the GUI.
    """
    server = _make_server()
    server.start()
    print("[server] Dashboard server listening on port 50051 (background)")

    def keep_alive():
        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            print("[server-thread] Shutting down...")
            server.stop(0)

    t = threading.Thread(target=keep_alive, daemon=True)
    t.start()
    return server


if __name__ == "__main__":
    serve()
