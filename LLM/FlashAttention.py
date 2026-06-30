import torch
import torch.nn.functional as F
import time


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def benchmark_sdpa_mps(
    B=4,
    H=8,
    L=1024,
    D=64,
    dtype=torch.float16,
    device="mps",
    warmup=10,
    iters=50,
):
    assert torch.backends.mps.is_available(), "当前 PyTorch 没有检测到 MPS"

    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype)

    def run(is_causal: bool):
        for _ in range(warmup):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
            )

        sync(device)
        start = time.perf_counter()

        for _ in range(iters):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
            )

        sync(device)
        end = time.perf_counter()

        return (end - start) * 1000 / iters

    non_causal_ms = run(False)
    causal_ms = run(True)

    print(f"device={device}, dtype={dtype}")
    print(f"B={B}, H={H}, L={L}, D={D}")
    print(f"non-causal avg time: {non_causal_ms:.4f} ms")
    print(f"causal     avg time: {causal_ms:.4f} ms")
    print(f"causal / non-causal: {causal_ms / non_causal_ms:.4f}")


if __name__ == "__main__":
    benchmark_sdpa_mps(
        B=400,
        H=8,
        L=1024,
        D=64,
        dtype=torch.float16,
        device="mps",
        warmup=10,
        iters=50,
    )