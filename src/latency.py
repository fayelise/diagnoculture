"""Partie 2 — Coût des modèles : paramètres, taille, latence CPU/GPU, export ONNX + quantification."""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import torch


def count_params(model, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def model_size_mb(model) -> float:
    """Taille réelle du state_dict sérialisé (float32 : ~4 octets par paramètre + buffers BN)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


@torch.inference_mode()
def measure_latency(model, img_size: int, device: str = "cpu", runs: int = 100, warmup: int = 20,
                    cpu_threads: int | None = 1) -> dict:
    """Latence d'inférence pour UNE image (batch = 1, cas d'usage smartphone).

    - warm-up : les premières itérations incluent des coûts ponctuels
      (allocation mémoire, initialisation cuDNN, caches) qu'on exclut ;
    - GPU : les noyaux CUDA sont asynchrones, il faut torch.cuda.synchronize()
      avant de lire l'horloge, sinon on ne mesure que le temps de lancement ;
    - CPU : on fixe le nombre de threads (1 par défaut ~ un cœur de smartphone
      d'entrée de gamme) pour une mesure comparable entre modèles.
    """
    model = model.to(device).eval()
    prev_threads = torch.get_num_threads()
    if device == "cpu" and cpu_threads:
        torch.set_num_threads(cpu_threads)
    x = torch.randn(1, 3, img_size, img_size, device=device)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    for _ in range(warmup):
        model(x)
    sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(prev_threads)
    return {"mean_ms": float(np.mean(times)), "std_ms": float(np.std(times)),
            "p95_ms": float(np.percentile(times, 95)), "runs": runs, "device": device,
            "threads": cpu_threads if device == "cpu" else None}


def export_onnx(model, img_size: int, path: Path) -> Path:
    """Export ONNX (format d'échange lisible par ONNX Runtime, y compris sur Android)."""
    model = model.cpu().eval()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.randn(1, 3, img_size, img_size), str(path), input_names=["image"],
                      output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    return path


def quantize_onnx_dynamic(src: Path, dst: Path, op_types=None) -> Path:
    """Quantification dynamique INT8 : poids stockés sur 8 bits (taille / ~4).

    Les activations sont quantifiées « à la volée » à chaque inférence.
    op_types=None quantifie toutes les couches supportées (Conv -> ConvInteger,
    MatMul/Gemm -> MatMulInteger) ; op_types=["MatMul", "Gemm"] ne quantifie que
    les couches denses. À discuter : sur CPU x86, les noyaux ConvInteger d'ONNX
    Runtime sont souvent PLUS LENTS que les convolutions float optimisées ; le
    gain est surtout en taille. La quantification statique (format QDQ, calibration
    sur quelques images d'apprentissage) est la voie normale pour accélérer un CNN.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = Path(dst).with_suffix(".pre.onnx")
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=op_types)
    pre.unlink(missing_ok=True)
    return Path(dst)


def onnx_session(path: Path, threads: int = 1):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def onnx_latency(path: Path, img_size: int, runs: int = 100, warmup: int = 20, threads: int = 1) -> dict:
    sess = onnx_session(path, threads)
    x = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"image": x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"image": x})
        times.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": float(np.mean(times)), "std_ms": float(np.std(times)),
            "size_mb": os.path.getsize(path) / 1e6}


def onnx_predict(path: Path, loader, threads: int = 4) -> np.ndarray:
    """Prédictions ONNX sur un loader (pour mesurer la perte de précision due à la quantification)."""
    sess = onnx_session(path, threads)
    preds = [sess.run(None, {"image": x.numpy()})[0].argmax(1) for x, _ in loader]
    return np.concatenate(preds)


def pareto_front(df, x: str, y: str):
    """Indices des points Pareto-optimaux : x à minimiser (latence/taille), y à maximiser (F1)."""
    d = df.sort_values([x, y], ascending=[True, False])
    front, best_y = [], -np.inf
    for i, r in d.iterrows():
        if r[y] > best_y:
            front.append(i)
            best_y = r[y]
    return front
