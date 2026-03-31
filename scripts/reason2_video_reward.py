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

"""Cosmos Reason 2 offline video scoring in the style of Cosmos Reason 1 reward output.

Example:
  python scripts/reason2_video_reward.py --video assets/sample.mp4 --model-size 8b --reasoning

Output format (matches the Cosmos Cookbook reward example shape):
  Video: <path>
  Physical accuracy: Yes|No
  Score (high is good): <0.0-1.0>

Yes means the clip is physically plausible overall; No means clear violations were found.
This is prompt-based scoring (not the Cosmos-Reason1-7B-Reward checkpoint).
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


REWARD_SYSTEM = """You are a strict physics and physical-plausibility judge for video. \
You evaluate gravity, collisions, object permanence, fluids, and plausible articulated motion. \
You ignore artistic style, cartoon rendering, lighting-only effects, and audio."""


REWARD_USER = """Watch the attached video and judge physical plausibility.

Respond with a single JSON object only (no markdown fences, no commentary). Use this schema:
{"score": <float from 0.0 to 1.0 inclusive, higher means more physically plausible>,
 "physically_accurate": <true if no major physics violations, false otherwise>}

Guidelines:
- score near 1.0 only when motion and interactions look physically consistent.
- set physically_accurate to false if you see clear violations (impossible trajectories, \
interpenetration that breaks scene logic, broken object permanence, etc.)."""


def format_reason1_style(*, video_path: Path, score: float, accurate: bool) -> str:
    yn = "Yes" if accurate else "No"
    lines = [
        f"Video: {video_path.name}",
        f"Physical accuracy: {yn}",
        f"Score (high is good): {score:.4f}",
    ]
    return "\n".join(lines) + "\n"


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
        "--save-text",
        type=Path,
        default=None,
        help="Write the formatted summary lines to this file in addition to stdout.",
    )
    return p.parse_args()


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
    tmp = write_prompt_yaml(system_prompt=REWARD_SYSTEM)
    try:
        offline = Offline(
            model=model_id,
            max_model_len=args.max_model_len,
            input_file=tmp,
            prompt=REWARD_USER,
            videos=[str(video)],
            reasoning=args.reasoning,
            vision=vision,
            verbose=args.verbose,
            output=str(args.output_debug) if args.output_debug else None,
            sampling=SamplingOverrides(max_tokens=512, temperature=0.2, top_p=0.9),
        )
        raw = run_offline_assistant_text(offline)
    finally:
        tmp.unlink(missing_ok=True)

    try:
        data = extract_json_object(raw)
    except (ValueError, json.JSONDecodeError) as e:
        print("Failed to parse model JSON. Raw response:\n", file=sys.stderr)
        print(raw, file=sys.stderr)
        print(f"\nParse error: {e}", file=sys.stderr)
        sys.exit(1)

    score = float(data["score"])
    if not 0.0 <= score <= 1.0:
        print(f"score out of range [0,1]: {score}", file=sys.stderr)
        sys.exit(1)
    acc = data["physically_accurate"]
    if not isinstance(acc, bool):
        print("physically_accurate must be a JSON boolean.", file=sys.stderr)
        sys.exit(1)

    text = format_reason1_style(video_path=video, score=score, accurate=acc)
    sys.stdout.write(text)
    if args.save_text:
        args.save_text.parent.mkdir(parents=True, exist_ok=True)
        args.save_text.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
