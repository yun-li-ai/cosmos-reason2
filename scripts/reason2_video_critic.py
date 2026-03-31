#!/usr/bin/env python3
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

"""Cosmos Reason 2 offline structured video critique (Cosmos Cookbook critic-style JSON).

Example:
  python scripts/reason2_video_critic.py --video assets/sample.mp4 --model-size 8b --reasoning

The model returns one JSON object shaped like the cookbook "Video Critic" example, e.g.:
  {"video_analysis": {"physical_accuracy": {...}, "reasoning_chain": {...}, ...}}

This is prompt-based analysis (not a separate Cosmos Reason 1 critic checkpoint).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from reason2_video_eval_lib import (
    extract_json_object,
    resolve_model,
    run_offline_assistant_text,
    write_prompt_yaml,
)


CRITIC_SYSTEM = """You are an expert video critic for physical AI and robotics footage. \
You give structured, evidence-based analysis: physical plausibility, logical and temporal \
consistency, and visual coherence. You ignore audio unless the user asks otherwise."""


CRITIC_USER = """Watch the attached video and produce one JSON object only (no markdown fences, \
no commentary). Match this structure and key names (use nested objects; fill with your analysis):

{
  "video_analysis": {
    "physical_accuracy": {
      "score": <float 0.0-1.0>,
      "violations": [<string, short labels or empty list>],
      "explanation": <string>
    },
    "reasoning_chain": {
      "logical_consistency": <float 0.0-1.0>,
      "causal_relationships": <string, e.g. "strong"|"weak"|"mixed">,
      "temporal_coherence": <string, e.g. "maintained"|"broken"|"mostly_maintained">
    },
    "content_quality": {
      "visual_coherence": <float 0.0-1.0>,
      "object_permanence": <float 0.0-1.0>,
      "scene_understanding": <string, e.g. "excellent"|"good"|"poor">
    }
  }
}

Be concise in string fields; scores must be floats between 0 and 1."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to one video file (e.g. mp4).",
    )
    p.add_argument(
        "--model-size",
        choices=("2b", "8b"),
        default="2b",
        help="Shorthand for nvidia/Cosmos-Reason2-2B or 8B (default: 2b).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Full Hugging Face model id or local path (overrides --model-size).",
    )
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional video sampling fps for the vision preprocessor.",
    )
    p.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable chain-of-thought; the model must still emit valid JSON after reasoning.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "-o",
        "--output-debug",
        type=Path,
        default=None,
        help="If set, dump preprocessed vision tensors under this directory.",
    )
    p.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Write the parsed JSON to this file in addition to stdout.",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Print JSON as one line (default: indented).",
    )
    return p.parse_args()


def validate_critic_shape(data: dict) -> None:
    """Raise ValueError if top-level shape is wrong."""
    va = data.get("video_analysis")
    if not isinstance(va, dict):
        raise ValueError('Expected top-level key "video_analysis" mapping to an object.')
    for key in ("physical_accuracy", "reasoning_chain", "content_quality"):
        if key not in va:
            raise ValueError(f'Missing "video_analysis"."{key}" in critic JSON.')


def main() -> None:
    args = parse_args()
    from cosmos_reason2_utils.script.inference import Offline, SamplingOverrides
    from cosmos_reason2_utils.vision import VisionConfig
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"--video is not a file: {video}", file=sys.stderr)
        sys.exit(1)

    model_id = resolve_model(args.model_size, args.model)
    vision = VisionConfig(fps=args.fps) if args.fps is not None else VisionConfig()
    tmp = write_prompt_yaml(system_prompt=CRITIC_SYSTEM)
    try:
        offline = Offline(
            model=model_id,
            max_model_len=args.max_model_len,
            input_file=tmp,
            prompt=CRITIC_USER,
            videos=[str(video)],
            reasoning=args.reasoning,
            vision=vision,
            verbose=args.verbose,
            output=str(args.output_debug) if args.output_debug else None,
            sampling=SamplingOverrides(max_tokens=4096, temperature=0.3, top_p=0.9),
        )
        raw = run_offline_assistant_text(offline)
    finally:
        tmp.unlink(missing_ok=True)

    try:
        data = extract_json_object(raw)
        validate_critic_shape(data)
    except (ValueError, json.JSONDecodeError, TypeError) as e:
        print("Failed to parse or validate critic JSON. Raw response:\n", file=sys.stderr)
        print(raw, file=sys.stderr)
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    indent = None if args.compact else 2
    text = json.dumps(data, indent=indent, ensure_ascii=False) + "\n"
    sys.stdout.write(text)
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
