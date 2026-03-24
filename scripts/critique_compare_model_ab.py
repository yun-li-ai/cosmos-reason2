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

"""Compare two Cosmos Transfer multiview model outputs (Model A vs Model B) with Cosmos-Reason2."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from string import Template
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = ROOT / "prompts" / "cosmos_transfer_model_ab_compare.yaml"


def _load_critique_generated_video():
    path = SCRIPTS_DIR / "critique_generated_video.py"
    spec = importlib.util.spec_from_file_location("_critique_generated_video", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_user_prompt(
    template_path: Path,
    *,
    generation_prompt: str,
    media_layout: str,
    optional_notes: str,
) -> str:
    data = yaml.safe_load(template_path.read_bytes())
    raw = data.get("user_prompt", "")
    if not raw or not isinstance(raw, str):
        raise ValueError(f"Missing user_prompt string in {template_path}")
    return Template(raw).substitute(
        generation_prompt=generation_prompt.strip(),
        media_layout=media_layout.strip(),
        optional_notes=optional_notes.strip(),
    )


def write_prompt_file(user_prompt: str) -> Path:
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="critique_model_ab_", text=True)
    os.close(fd)
    p = Path(path)
    p.write_text(yaml.safe_dump({"user_prompt": user_prompt}), encoding="utf-8")
    return p


def build_media_layout_triplets(
    rows: list[tuple[str, Path, Path, Path]],
) -> str:
    """rows: (view_key, control_path, model_a_path, model_b_path)."""
    n = len(rows) * 3
    lines = [
        f"There are **{n}** videos in **{len(rows)} triplets** (same camera order as Cosmos Transfer multiview).",
        "Each triplet is: **(1) control** with boxes/lanes → **(2) Model A generated** → **(3) Model B generated**.",
        "",
    ]
    v = 1
    for view_key, cpath, path_a, path_b in rows:
        lines.append(
            f"**{view_key}** — video {v}: control `{cpath.name}`; "
            f"video {v + 1}: Model A `{path_a.name}`; "
            f"video {v + 2}: Model B `{path_b.name}`."
        )
        v += 3
    lines.append("")
    lines.append(
        "When comparing variants, contrast video (k+1) vs (k+2) within the same triplet; "
        "both follow the same control clip (k)."
    )
    return "\n".join(lines)


def load_triplets(
    cgv: Any,
    *,
    control_dir: Path,
    model_a_dir: Path,
    model_b_dir: Path,
    auto_multiview_path: Path | None,
) -> list[tuple[str, Path, Path, Path]]:
    pairs_a = cgv.load_multiview_pairs(
        control_dir=control_dir,
        generated_dir=model_a_dir,
        auto_multiview_path=auto_multiview_path,
    )
    pairs_b = cgv.load_multiview_pairs(
        control_dir=control_dir,
        generated_dir=model_b_dir,
        auto_multiview_path=auto_multiview_path,
    )
    if len(pairs_a) != len(pairs_b):
        raise ValueError("Model A and Model B multiview pair counts differ.")
    rows: list[tuple[str, Path, Path, Path]] = []
    for (vk_a, ctrl_a, gen_a), (vk_b, ctrl_b, gen_b) in zip(
        pairs_a, pairs_b, strict=True
    ):
        if vk_a != vk_b:
            raise ValueError(f"View order mismatch: {vk_a!r} vs {vk_b!r}")
        if ctrl_a.resolve() != ctrl_b.resolve():
            raise ValueError(f"Control path mismatch for view {vk_a!r}.")
        rows.append((vk_a, ctrl_a, gen_a, gen_b))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Critique and compare two Cosmos Transfer 2.5 multiview outputs (**Model A** vs **Model B**) "
            "with Cosmos-Reason2 offline inference. Same layout as critique_generated_video.py "
            "(seven control clips; each output dir with auto_multiview_<view>.mp4 or other known stems)."
        )
    )
    p.add_argument(
        "--control-dir",
        type=Path,
        required=True,
        help="Directory of seven control videos (same as Cosmos Transfer run).",
    )
    p.add_argument(
        "--model-a-dir",
        type=Path,
        required=True,
        dest="model_a_dir",
        help="Directory of seven generated videos for **Model A**.",
    )
    p.add_argument(
        "--model-b-dir",
        type=Path,
        required=True,
        dest="model_b_dir",
        help="Directory of seven generated videos for **Model B**.",
    )
    p.add_argument(
        "--auto-multiview-json",
        type=Path,
        default=None,
        help="Optional auto_multiview.json for view order and control basenames; can supply prompt via JSON.",
    )
    p.add_argument(
        "generation_prompt",
        type=str,
        nargs="?",
        default="",
        help="Text prompt shared by both runs. Omit if using --prompt-file or JSON prompt.",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="UTF-8 file with the shared generation prompt.",
    )
    p.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional extra context appended to the prompt (e.g. checkpoint ids).",
    )
    p.add_argument(
        "--notes-file",
        type=Path,
        default=None,
        help="Optional UTF-8 notes file.",
    )
    p.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="YAML prompt template path.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason2-2B",
        help="Hugging Face model id or path for offline inference.",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Max model length for vLLM.",
    )
    p.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable reasoning-style sampling for longer analysis.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose inference logging.")
    p.add_argument(
        "-o",
        "--output-debug",
        type=Path,
        default=None,
        help="Dump preprocessed frames for debugging.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write only the Assistant markdown to this file (no echoed system/user prompt or separators); "
            "the full log still prints to stdout."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    template = args.template.expanduser().resolve()
    if not template.is_file():
        print(f"Template not found: {template}", file=sys.stderr)
        sys.exit(1)

    json_prompt: str | None = None
    auto_path: Path | None = None
    if args.auto_multiview_json is not None:
        auto_path = args.auto_multiview_json.expanduser().resolve()
        if not auto_path.is_file():
            print(f"--auto-multiview-json not found: {auto_path}", file=sys.stderr)
            sys.exit(1)
        try:
            meta = json.loads(auto_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                pr = meta.get("prompt")
                if isinstance(pr, str) and pr.strip():
                    json_prompt = pr.strip()
        except (json.JSONDecodeError, OSError) as e:
            print(f"Could not read {auto_path}: {e}", file=sys.stderr)
            sys.exit(1)

    if args.prompt_file is not None:
        pf = args.prompt_file.expanduser().resolve()
        if not pf.is_file():
            print(f"--prompt-file not found: {pf}", file=sys.stderr)
            sys.exit(1)
        generation_prompt = pf.read_text(encoding="utf-8")
    elif args.generation_prompt.strip():
        generation_prompt = args.generation_prompt
    elif json_prompt is not None:
        generation_prompt = json_prompt
    else:
        print(
            "Provide the shared generation prompt (positional, --prompt-file, or prompt in JSON).",
            file=sys.stderr,
        )
        sys.exit(1)

    if not generation_prompt.strip():
        print("Generation prompt is empty.", file=sys.stderr)
        sys.exit(1)

    optional_notes = ""
    notes = args.notes
    if args.notes_file is not None:
        nf = args.notes_file.expanduser().resolve()
        if not nf.is_file():
            print(f"--notes-file not found: {nf}", file=sys.stderr)
            sys.exit(1)
        notes = nf.read_text(encoding="utf-8")
    if notes.strip():
        optional_notes = "**Additional operator notes:**\n" + notes.strip()

    cdir = args.control_dir.expanduser().resolve()
    dir_a = args.model_a_dir.expanduser().resolve()
    dir_b = args.model_b_dir.expanduser().resolve()
    for name, d in (
        ("--control-dir", cdir),
        ("--model-a-dir", dir_a),
        ("--model-b-dir", dir_b),
    ):
        if not d.is_dir():
            print(f"{name} is not a directory: {d}", file=sys.stderr)
            sys.exit(1)

    cgv = _load_critique_generated_video()
    try:
        rows = load_triplets(
            cgv,
            control_dir=cdir,
            model_a_dir=dir_a,
            model_b_dir=dir_b,
            auto_multiview_path=auto_path,
        )
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"Failed to resolve videos: {e}", file=sys.stderr)
        sys.exit(1)

    media_layout = build_media_layout_triplets(rows)
    user_prompt = load_user_prompt(
        template,
        generation_prompt=generation_prompt,
        media_layout=media_layout,
        optional_notes=optional_notes,
    )

    videos: list[Path] = []
    for _, ctrl, vid_a, vid_b in rows:
        videos.extend([ctrl, vid_a, vid_b])

    tmp_yaml = write_prompt_file(user_prompt)
    try:
        cmd, run_env = cgv.build_offline_command(
            videos=videos,
            input_yaml=tmp_yaml,
            model=args.model,
            max_model_len=args.max_model_len,
            reasoning=args.reasoning,
            verbose=args.verbose,
            output_debug=args.output_debug,
        )
        if args.verbose:
            print("Command:", " ".join(cmd), file=sys.stderr)
        code = cgv.run_offline_inference(cmd, run_env, output_text=args.output)
        raise SystemExit(code)
    finally:
        tmp_yaml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
