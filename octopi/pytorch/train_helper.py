import torch

def auto_amp(device: torch.device):
    """
    Pick the best mixed-precision config for the current device.

    Returns (enabled, dtype, scaler):
      - Ampere+ (H100/H200, A100, RTX 30xx/40xx, RTX Axxxx): bf16, no scaler.
      - Volta/Turing (V100, T4, RTX 20xx):                   fp16 + GradScaler.
      - Pascal or older, CPU, MPS:                           fp32, disabled.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return False, torch.float32, None

    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, None

    major, _ = torch.cuda.get_device_capability(device)
    if major >= 7:
        return True, torch.float16, torch.amp.GradScaler("cuda")

    return False, torch.float32, None