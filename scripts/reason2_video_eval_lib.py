# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared offline video inference for Reason-1-style evaluation scripts."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_COSMOS_REASON2_UTILS_ROOT = _REPO_ROOT / "cosmos_reason2_utils"
if str(_COSMOS_REASON2_UTILS_ROOT) not in sys.path:
    sys.path.insert(0, str(_COSMOS_REASON2_UTILS_ROOT))

if TYPE_CHECKING:
    from cosmos_reason2_utils.script.inference import Offline

MODEL_PRESETS: dict[str, str] = {
    "2b": "nvidia/Cosmos-Reason2-2B",
    "8b": "nvidia/Cosmos-Reason2-8B",
}


def resolve_model(model_size: str, model: str | None) -> str:
    """Return Hugging Face model id from --model-size or explicit --model."""
    if model:
        return model
    key = model_size.lower().strip()
    if key not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown --model-size {model_size!r}; choose from {sorted(MODEL_PRESETS)}."
        )
    return MODEL_PRESETS[key]


def write_prompt_yaml(*, system_prompt: str) -> Path:
    """Write a temporary YAML input; user text is passed via Offline.prompt."""
    import yaml

    data = {"system_prompt": system_prompt, "user_prompt": ""}
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="reason2_eval_",
        delete=False,
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(data, f, default_flow_style=False)
        return Path(f.name)


def run_offline_assistant_text(args: Offline) -> str:
    """Run vLLM offline path once; return assistant text (no printing)."""
    import qwen_vl_utils
    import transformers
    import vllm

    from cosmos_reason2_utils.text import create_conversation
    from cosmos_reason2_utils.vision import PIXELS_PER_TOKEN, VisionConfig, save_tensor

    vision_kwargs = args.vision.model_dump(exclude_none=True)
    assert args.sampling_params.max_tokens
    if args.max_model_len < args.sampling_params.max_tokens:
        raise ValueError("Max model length must be greater than max tokens.")
    max_seq_len = args.max_model_len - args.sampling_params.max_tokens
    total_pixels = int(max_seq_len * PIXELS_PER_TOKEN * 0.9)
    if "total_pixels" in vision_kwargs:
        if vision_kwargs["total_pixels"] > total_pixels:
            raise ValueError(
                f"Total pixels {vision_kwargs['total_pixels']} exceeds limit {total_pixels}."
            )
    else:
        vision_kwargs["total_pixels"] = total_pixels
    VisionConfig.model_validate(vision_kwargs)

    conversation = create_conversation(
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        images=args.images,
        videos=args.videos,
        vision_kwargs=vision_kwargs,
    )

    llm = vllm.LLM(
        model=args.model,
        revision=args.revision,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": len(args.images), "video": len(args.videos)},
    )

    processor: transformers.Qwen3VLProcessor = transformers.AutoProcessor.from_pretrained(
        args.model
    )
    add_vision_ids = (len(args.images) + len(args.videos)) > 1
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_ids=add_vision_ids,
    )
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        conversation,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        if image_inputs is not None:
            for i, image in enumerate(image_inputs):
                save_tensor(image, str(out_dir / f"image_{i}"))
        if video_inputs is not None:
            for i, (video, _) in enumerate(video_inputs):
                save_tensor(video, str(out_dir / f"video_{i}"))

    mm_data: dict = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    outputs = llm.generate([llm_inputs], sampling_params=args.sampling_params)
    return outputs[0].outputs[0].text.strip()


def strip_after_reasoning_tag(raw: str) -> str:
    """If the model used </think>, keep only text after the last tag."""
    tag = "</think>"
    if tag in raw:
        return raw.rsplit(tag, 1)[-1].strip()
    return raw.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output (fenced block or raw)."""
    text = strip_after_reasoning_tag(text)
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])
