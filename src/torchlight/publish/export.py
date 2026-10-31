import json
import shutil
import subprocess
import sys
from pathlib import Path

from torchlight.train.config import LoRARun

MLX_ADAPTER_FILES = ("adapters.safetensors", "adapter_config.json")
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_TARGETS = ("llama-quantize", "llama-completion")
BASE_TOKENIZER_FILES = (
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
# Phi-4-mini's remote code, which transformers now ships natively; the chat template is
# also in the base tokenizer_config.json.
STALE_FILES = ("configuration_phi3.py", "modeling_phi3.py", "sample_finetune.py")


def match_base_files(model_dir: Path, base_model: str) -> None:
    """Give a model fused by mlx-lm the base model's tokenizer files and a config without
    remote code. transformers 5 rewrites the tokenizer config in a form 4.x can't load."""
    from huggingface_hub import snapshot_download

    base = Path(snapshot_download(base_model, allow_patterns=list(BASE_TOKENIZER_FILES)))
    for name in BASE_TOKENIZER_FILES:
        shutil.copy(base / name, model_dir / name)
    for name in (*STALE_FILES, "chat_template.jinja"):
        (model_dir / name).unlink(missing_ok=True)
    config = json.loads((model_dir / "config.json").read_text())
    config.pop("auto_map", None)
    (model_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def peft_adapter(adapter_dir: Path, out_dir: Path, params: LoRARun) -> int:
    """Rewrite an mlx-lm LoRA adapter in PEFT's format, keeping the MLX files under mlx/.
    Returns the number of adapted modules."""
    import mlx.core as mx
    import numpy as np
    from safetensors.numpy import save_file

    tensors, modules = {}, set()
    for key, value in mx.load(str(adapter_dir / "adapters.safetensors")).items():
        module, name = key.rsplit(".", 1)
        modules.add(module)
        # mlx-lm computes x @ lora_a @ lora_b; PEFT keeps A and B as nn.Linear weights,
        # which are stored transposed.
        peft_name = {"lora_a": "lora_A", "lora_b": "lora_B"}[name]
        tensors[f"base_model.model.{module}.{peft_name}.weight"] = np.ascontiguousarray(
            np.array(value.astype(mx.float32)).T
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": params.model,
        "r": params.rank,
        "lora_alpha": params.scale * params.rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "target_modules": sorted(modules),
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "mlx").mkdir(exist_ok=True)
    for name in MLX_ADAPTER_FILES:
        shutil.copy(adapter_dir / name, out_dir / "mlx" / name)
    return len(modules)


def build_llama_cpp(llama_cpp: Path) -> Path:
    """Clone and build the llama.cpp tools that quantize and test GGUF files; returns their
    folder."""
    build = llama_cpp / "build"
    bin_dir = build / "bin"
    if all((bin_dir / target).exists() for target in LLAMA_CPP_TARGETS):
        return bin_dir
    if not llama_cpp.exists():
        subprocess.run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(llama_cpp)], check=True)
    options = ["-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF"]
    subprocess.run(["cmake", "-S", str(llama_cpp), "-B", str(build), *options], check=True)
    subprocess.run(
        ["cmake", "--build", str(build), "--config", "Release", "-j", "8", "--target"]
        + list(LLAMA_CPP_TARGETS),
        check=True,
    )
    return bin_dir


def gguf_files(
    model_dir: Path, out_dir: Path, name: str, llama_cpp: Path, quants: list[str]
) -> list[Path]:
    """Convert a Hugging Face model folder to GGUF and quantize it."""
    bin_dir = build_llama_cpp(llama_cpp)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = out_dir / f"{name}-BF16.gguf"
    subprocess.run(
        [
            sys.executable,
            str(llama_cpp / "convert_hf_to_gguf.py"),
            str(model_dir),
            "--outfile",
            str(full),
            "--outtype",
            "bf16",
        ],
        check=True,
    )
    paths = []
    for quant in quants:
        path = out_dir / f"{name}-{quant}.gguf"
        subprocess.run([str(bin_dir / "llama-quantize"), str(full), str(path), quant], check=True)
        paths.append(path)
    # The 16-bit model is already published as safetensors; GGUF users want the quants.
    full.unlink()
    return paths
