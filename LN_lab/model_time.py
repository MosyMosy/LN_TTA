# Benchmark B=1 vs B=32 for a model taking (B, 1024, 3) inputs,
# and write results to a text file.
#
# Assumes you already have: source_model

import time, torch, contextlib
from pathlib import Path
from datetime import datetime

def make_input(B, N=1024, C=3, device=None, dtype=torch.float32):
    return torch.randn(B, N, C, dtype=dtype, device=device)

def time_per_batch(model, x, warmup=20, iters=100, use_amp=False):
    device = x.device
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) \
              if (use_amp and device.type == "cuda") else contextlib.nullcontext()
    with torch.inference_mode(), amp_ctx:
        for _ in range(warmup):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / iters  # seconds per batch

def _format_report(device, t1, t32, fps1, fps32, speed, iters, warmup, use_amp, use_compile, tf32, in_shape):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dev_name = (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
    return (
        "==== Inference Benchmark (B=1 vs B=32) ====\n"
        f"Timestamp: {ts}\n"
        f"Device: {device} ({dev_name})\n"
        f"Input shape: {in_shape}\n"
        f"iters={iters}, warmup={warmup}, AMP={use_amp}, compile={use_compile}, TF32={tf32}\n"
        f"T(1)    = {t1*1e3:.3f} ms\n"
        f"T(32)   = {t32*1e3:.3f} ms\n"
        f"FPS@B=1 = {fps1:.2f}\n"
        f"FPS@B=32= {fps32:.2f}\n"
        f"Throughput speedup (B=32 vs B=1): x{speed:.2f}\n"
        "-------------------------------------------\n"
    )

def _save_report(text, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(text)

def benchmark(model,
              device=None,
              iters=100,
              warmup=30,
              use_amp=False,
              use_compile=False,
              tf32=True,
              out_path="bench_results.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
    model = model.eval().to(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        if use_compile and hasattr(torch, "compile"):
            model = torch.compile(model, mode="max-autotune")

    x1  = make_input(1,  device=device)
    x32 = make_input(32, device=device)

    t1  = time_per_batch(model, x1,  warmup=warmup, iters=iters, use_amp=use_amp)
    t32 = time_per_batch(model, x32, warmup=warmup, iters=iters, use_amp=use_amp)

    fps1  = 1.0 / t1
    fps32 = 32.0 / t32
    speed = fps32 / fps1

    # Print to console
    print(f"T(1)={t1*1e3:.3f} ms  | FPS@B=1={fps1:.2f}")
    print(f"T(32)={t32*1e3:.3f} ms | FPS@B=32={fps32:.2f}")
    print(f"Throughput speedup (B=32 vs B=1): x{speed:.2f}")

    # Save to text file
    report = _format_report(device, t1, t32, fps1, fps32, speed,
                            iters, warmup, use_amp, use_compile, tf32,
                            in_shape="(B, 1024, 3)")
    _save_report(report, out_path)

    return {"T1_ms": t1*1e3, "T32_ms": t32*1e3, "FPS_B1": fps1, "FPS_B32": fps32, "speedup": speed, "out_path": str(out_path)}

# --- Run it ---
results = benchmark(
    source_model,
    iters=100,
    warmup=30,
    use_amp=False,       # set True to test FP16 on CUDA
    use_compile=False,   # set True (PyTorch 2+) to try extra tuning
    tf32=True,
    out_path="bench_results.txt"  # <- your text file path
)
print(f"Report saved to: {results['out_path']}")
