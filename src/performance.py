from __future__ import annotations
import threading
import time
import psutil

class Monitor:
    def __init__(self, gpu=False):
        self.gpu = gpu
        self.result = {}
    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.start_rss = self.peak
        self.started = time.perf_counter()
        self.stop = threading.Event()
        self.torch = None
        if self.gpu:
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                torch.cuda.reset_peak_memory_stats()
        def sample():
            while not self.stop.wait(.05):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, *args):
        if self.torch:
            self.torch.cuda.synchronize()
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.result = {"seconds": time.perf_counter()-self.started, "peak_rss_mb": self.peak/1024**2,
                       "start_rss_mb": self.start_rss/1024**2,
                       "gpu_peak_mb": self.torch.cuda.max_memory_allocated()/1024**2 if self.torch else 0}
