#!/usr/bin/env python3
"""
Standalone Hindi CSM-1B LoRA inference script.

Designed for Google Colab, Kaggle, and local Linux GPU environments.

Example:
    python inference.py \
        --adapter-path checkpoints/checkpoint-2750 \
        --processor-path fixed_csm_processor \
        --reference-audio outputs/reference/reference_female.wav \
        --reference-metadata outputs/reference/reference_female.json \
        --output-dir outputs/inference
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


MODEL_NAME = "unsloth/csm-1b"
TARGET_SAMPLE_RATE = 24_000
CSM_PAD_TOKEN = "<|finetune_right_pad_id|>"
CSM_PAD_TOKEN_ID = 128004

DEFAULT_SENTENCES = [
    {
        "id": "01",
        "name": "namaste_order_sent",
        "text": "नमस्ते, आपका ऑर्डर सफलतापूर्वक भेज दिया गया है।",
        "min_tokens": 40,
        "max_tokens": 125,
    },
    {
        "id": "02",
        "name": "refund_timeline",
        "text": "आपका रिफंड अगले पाँच कार्य दिवसों में आपके खाते में जमा कर दिया जाएगा।",
        "min_tokens": 65,
        "max_tokens": 180,
    },
    {
        "id": "03",
        "name": "further_assistance",
        "text": "क्या मैं आपकी किसी और सहायता कर सकता हूँ?",
        "min_tokens": 38,
        "max_tokens": 125,
    },
    {
        "id": "04",
        "name": "thank_you",
        "text": "धन्यवाद! आपका दिन शुभ हो।",
        "min_tokens": 30,
        "max_tokens": 100,
    },
    {
        "id": "05",
        "name": "order_number",
        "text": "कृपया अपना ऑर्डर नंबर बताइए।",
        "min_tokens": 32,
        "max_tokens": 110,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Hindi speech with the fine-tuned CSM-1B LoRA adapter."
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Local LoRA checkpoint directory or Hugging Face adapter repository ID.",
    )
    parser.add_argument(
        "--processor-path",
        required=True,
        help="Local fixed CSM processor directory or Hugging Face repository ID.",
    )
    parser.add_argument(
        "--reference-audio",
        required=True,
        help="Path to the female reference WAV file.",
    )
    parser.add_argument(
        "--reference-metadata",
        required=True,
        help="Path to JSON containing at least the reference transcript.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/inference",
        help="Directory in which generated WAV files and the manifest are saved.",
    )
    parser.add_argument(
        "--base-model",
        default=MODEL_NAME,
        help=f"Base model ID. Default: {MODEL_NAME}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5107,
        help="Recorded seed. For greedy decoding this is execution metadata.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help=(
            "Custom Hindi text. Repeat --text to generate more than one sentence. "
            "When omitted, the five assignment sentences are generated."
        ),
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling instead of the default greedy decoding.",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the per-sentence maximum token limit.",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=None,
        help="Override the per-sentence minimum token limit.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. Otherwise HF_TOKEN is read from the environment.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Prevent downloads and require all model files to already exist locally.",
    )
    return parser.parse_args()


def normalize_hindi_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\u200b\u2060\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([।!?.,])\1+", r"\1", text)
    return text


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_hf_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token

    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    # Supports Colab Secrets without making the script Colab-only.
    try:
        from google.colab import userdata  # type: ignore

        token = userdata.get("HF_TOKEN")
        if token:
            return token
    except Exception:
        pass

    return None


def apply_csm_pad_patch() -> None:
    """
    Work around the CSM dedicated-pad-token compatibility issue in Unsloth.
    This must run before loading the base model.
    """
    import unsloth_zoo.pad_token as pad_token_module
    import unsloth_zoo.tokenizer_utils as tokenizer_utils_module

    def csm_pad_token_bypass(
        tokenizer: Any,
        model: Any = None,
        model_config: Any = None,
        *,
        is_vision_model: Any = None,
        allow_add: bool = True,
    ) -> dict[str, Any]:
        del is_vision_model, allow_add

        if tokenizer is None:
            raise ValueError("Tokenizer is required.")

        inner = getattr(tokenizer, "tokenizer", tokenizer)
        token_id = inner.convert_tokens_to_ids(CSM_PAD_TOKEN)

        if token_id != CSM_PAD_TOKEN_ID:
            raise ValueError(
                f"Unexpected CSM pad token ID: {token_id}; "
                f"expected {CSM_PAD_TOKEN_ID}."
            )

        inner.pad_token = CSM_PAD_TOKEN
        inner.padding_side = "right"

        configs = [
            model_config,
            getattr(model, "config", None) if model is not None else None,
        ]

        for config in configs:
            if config is None:
                continue

            try:
                config.pad_token_id = CSM_PAD_TOKEN_ID
            except Exception:
                pass

            for nested_name in (
                "text_config",
                "backbone_config",
                "decoder_config",
            ):
                nested = getattr(config, nested_name, None)
                if nested is not None:
                    try:
                        nested.pad_token_id = CSM_PAD_TOKEN_ID
                    except Exception:
                        pass

        if model is not None and hasattr(model, "generation_config"):
            model.generation_config.pad_token_id = CSM_PAD_TOKEN_ID

        return {
            "changed": False,
            "reason": "CSM dedicated padding token selected",
            "old_pad": None,
            "new_pad": CSM_PAD_TOKEN,
            "added": False,
        }

    pad_token_module.fix_pad_token = csm_pad_token_bypass
    tokenizer_utils_module.fix_pad_token = csm_pad_token_bypass


def load_processor(
    processor_path: str,
    token: str | None,
    local_files_only: bool,
):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        processor_path,
        token=token,
        local_files_only=local_files_only,
    )
    processor.tokenizer.pad_token = CSM_PAD_TOKEN
    processor.tokenizer.padding_side = "right"

    if processor.tokenizer.pad_token_id != CSM_PAD_TOKEN_ID:
        raise RuntimeError(
            "The supplied processor does not contain the expected CSM padding token. "
            f"Received ID {processor.tokenizer.pad_token_id}; "
            f"expected {CSM_PAD_TOKEN_ID}."
        )

    return processor


def load_reference(
    audio_path: Path,
    metadata_path: Path,
) -> tuple[np.ndarray, int, str, int]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Reference metadata not found: {metadata_path}")

    waveform, sample_rate = sf.read(audio_path, dtype="float32")
    waveform = np.asarray(waveform, dtype=np.float32).squeeze()

    if waveform.ndim != 1 or waveform.size == 0:
        raise RuntimeError(f"Invalid reference waveform shape: {waveform.shape}")
    if not np.all(np.isfinite(waveform)):
        raise RuntimeError("Reference waveform contains invalid values.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reference_text = normalize_hindi_text(metadata["text"])
    speaker_id = int(metadata.get("speaker_id", 0))

    return waveform, int(sample_rate), reference_text, speaker_id


def load_model(
    base_model_name: str,
    adapter_path: str,
    processor_path: str,
    token: str | None,
    local_files_only: bool,
):
    # Import after applying the pad-token patch.
    from peft import PeftModel
    from transformers import CsmForConditionalGeneration
    from unsloth import FastModel

    base_model, _ = FastModel.from_pretrained(
        model_name=base_model_name,
        tokenizer_name=processor_path,
        max_seq_length=1024,
        dtype=torch.float16,
        auto_model=CsmForConditionalGeneration,
        load_in_4bit=False,
        device_map={"": 0},
        token=token,
        local_files_only=local_files_only,
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
        token=token,
        local_files_only=local_files_only,
    )

    model.config.pad_token_id = CSM_PAD_TOKEN_ID
    model.config.use_cache = True

    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = CSM_PAD_TOKEN_ID

    model.eval()
    return model


def build_sentences(custom_texts: list[str] | None) -> list[dict[str, Any]]:
    if not custom_texts:
        return DEFAULT_SENTENCES

    items = []
    for index, text in enumerate(custom_texts, start=1):
        normalized = normalize_hindi_text(text)
        items.append(
            {
                "id": f"{index:02d}",
                "name": f"custom_{index:02d}",
                "text": normalized,
                "min_tokens": 30,
                "max_tokens": max(100, min(220, len(normalized) * 3)),
            }
        )
    return items


def generate_one(
    model,
    processor,
    reference_audio: np.ndarray,
    reference_text: str,
    reference_speaker_id: int,
    target_text: str,
    output_path: Path,
    min_new_tokens: int,
    max_new_tokens: int,
    seed: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> dict[str, Any]:
    set_seed(seed)

    conversation = [
        {
            "role": str(reference_speaker_id),
            "content": [
                {"type": "text", "text": reference_text},
                {"type": "audio", "path": reference_audio},
            ],
        },
        {
            "role": str(reference_speaker_id),
            "content": [{"type": "text", "text": target_text}],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    generation_args: dict[str, Any] = {
        **inputs,
        "output_audio": True,
        "min_new_tokens": min_new_tokens,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "depth_decoder_do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": CSM_PAD_TOKEN_ID,
    }

    if do_sample:
        generation_args.update(
            {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "depth_decoder_temperature": temperature,
                "depth_decoder_top_k": top_k,
                "depth_decoder_top_p": top_p,
            }
        )

    started_at = time.time()

    with torch.inference_mode():
        generated = model.generate(**generation_args)

    waveform = (
        generated[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .squeeze()
    )

    if waveform.ndim != 1 or waveform.size == 0:
        raise RuntimeError(f"Invalid generated waveform shape: {waveform.shape}")
    if not np.all(np.isfinite(waveform)):
        raise RuntimeError("Generated waveform contains invalid values.")

    waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        output_path,
        waveform,
        TARGET_SAMPLE_RATE,
        subtype="PCM_16",
    )

    result = {
        "duration_seconds": len(waveform) / TARGET_SAMPLE_RATE,
        "generation_seconds": time.time() - started_at,
        "peak_amplitude": float(np.max(np.abs(waveform))),
        "number_of_samples": int(len(waveform)),
    }

    del inputs, generated, waveform
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. In Colab select Runtime → Change runtime type → GPU."
        )

    token = get_hf_token(args.hf_token)
    set_seed(args.seed)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Base model:", args.base_model)
    print("Adapter:", args.adapter_path)
    print("Processor:", args.processor_path)
    print("Decoding:", "sampling" if args.do_sample else "greedy")

    apply_csm_pad_patch()

    processor = load_processor(
        args.processor_path,
        token=token,
        local_files_only=args.local_files_only,
    )

    (
        reference_audio,
        reference_sample_rate,
        reference_text,
        reference_speaker_id,
    ) = load_reference(
        Path(args.reference_audio),
        Path(args.reference_metadata),
    )

    print("Reference sample rate:", reference_sample_rate)
    print("Reference text:", reference_text)

    model = load_model(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        processor_path=args.processor_path,
        token=token,
        local_files_only=args.local_files_only,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = build_sentences(args.text)
    records: list[dict[str, Any]] = []

    for item in sentences:
        min_tokens = (
            args.min_new_tokens
            if args.min_new_tokens is not None
            else int(item["min_tokens"])
        )
        max_tokens = (
            args.max_new_tokens
            if args.max_new_tokens is not None
            else int(item["max_tokens"])
        )

        output_path = output_dir / (
            f"checkpoint2750_greedy_seed{args.seed}_"
            f"{item['id']}_{item['name']}.wav"
        )

        print(f"\nGenerating {item['id']}: {item['text']}")

        try:
            stats = generate_one(
                model=model,
                processor=processor,
                reference_audio=reference_audio,
                reference_text=reference_text,
                reference_speaker_id=reference_speaker_id,
                target_text=item["text"],
                output_path=output_path,
                min_new_tokens=min_tokens,
                max_new_tokens=max_tokens,
                seed=args.seed,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

            record = {
                "sample_id": item["id"],
                "text": item["text"],
                "filename": output_path.name,
                "status": "success",
                "seed": args.seed,
                "decoding": "sampling" if args.do_sample else "greedy",
                **stats,
            }
            print("Saved:", output_path)

        except Exception as exc:
            record = {
                "sample_id": item["id"],
                "text": item["text"],
                "filename": output_path.name,
                "status": "failed",
                "seed": args.seed,
                "error": str(exc),
            }
            print("FAILED:", exc, file=sys.stderr)

        records.append(record)

    manifest_path = output_dir / "inference_manifest.json"
    manifest_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    success_count = sum(record["status"] == "success" for record in records)
    print(f"\nCompleted: {success_count}/{len(records)} successful")
    print("Manifest:", manifest_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return 0 if success_count == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
