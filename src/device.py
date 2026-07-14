"""
device.py — Unified device selection for the entire pipeline.

Priority: CUDA > DirectML (AMD GPU on Windows) > CPU.
Import `get_device()` everywhere — never hardcode a device string.
"""

import torch


def get_device() -> torch.device:
    """Return the best available compute device."""
    # 1. Check for NVIDIA CUDA
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] Using CUDA — {torch.cuda.get_device_name(0)}")
        return dev

    # 2. Check for AMD DirectML (Windows)
    try:
        import torch_directml  # noqa: F401
        dev = torch_directml.device()
        print("[device] Using DirectML (AMD GPU)")
        return dev
    except ImportError:
        pass

    # 3. Fallback to CPU
    dev = torch.device("cpu")
    print(f"[device] Using CPU — {torch.get_num_threads()} threads available")
    return dev


if __name__ == "__main__":
    device = get_device()
    print(f"Selected device: {device}")
