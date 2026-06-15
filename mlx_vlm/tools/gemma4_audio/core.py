import sys


def load_model(model_path: str):
    """Load model and processor, exit with code 1 if audio weights absent."""
    import time
    print(f"[init] Loading {model_path}...")
    t0 = time.time()
    from mlx_vlm import load
    model, processor = load(model_path)
    print(f"[init] Loaded in {time.time()-t0:.1f}s")
    print(f"[init] model_type: {model.model_type}")
    print(f"[init] embed_audio: {getattr(model, 'embed_audio', None) is not None}")
    if getattr(model, "embed_audio", None) is None:
        print("[ERROR] No embed_audio — checkpoint missing audio weights.")
        sys.exit(1)
    return model, processor
