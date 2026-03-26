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

"""Critique control–generated alignment from seven layered (control+gen) multiview clips.

Example:
    python3 scripts/critique_layered_control_alignment.py \
        --layered-dir assets/overlay_output/b_trained_golden_hour \
        --auto-multiview-json assets/inference_results/b_trained_golden_hour/auto_multiview.json \
        --output assets/overlay_output/critique_results/critique_b_trained_golden_hour.md

Layered videos are resolved like multiview outputs (see critique_generated_video), and by default
also try stems with ``_overlay`` (e.g. ``FRONT_CENTER_overlay.mp4``). Override with ``--overlay-suffix``.
"""

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
DEFAULT_TEMPLATE = ROOT / "prompts" / "cosmos_transfer_layered_control_alignment.yaml"


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
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="critique_layered_", text=True)
    os.close(fd)
    p = Path(path)
    p.write_text(yaml.safe_dump({"user_prompt": user_prompt}), encoding="utf-8")
    return p


def build_media_layout_layered(rows: list[tuple[str, Path]]) -> str:
    """rows: (view_key, layered_video_path)."""
    n = len(rows)
    lines = [
        f"There are **{n}** videos: **one layered clip per camera** (control graphics composited with generated video).",
        "Watch in order; each file corresponds to a single multiview camera.",
        "",
    ]
    for i, (view_key, lpath) in enumerate(rows, start=1):
        lines.append(f"**{view_key}** — video {i}: `{lpath.name}`")
    lines.append("")
    lines.append(
        "For each clip, judge whether bounding boxes, lane lines, and road boundaries "
        "match the generated objects and road geometry, and whether anything appears in the "
        "generated scene **without** a plausible control cue (spurious vehicles, etc.)."
    )
    return "\n".join(lines)


def resolve_layered_video(
    layered_dir: Path,
    view_key: str,
    control_basename: str,
    cgv: Any,
    *,
    overlay_suffix: str,
) -> Path:
    """Like resolve_generated_video, but also tries ``{stem}{overlay_suffix}`` for each stem."""
    control_stem = Path(control_basename).stem
    stems = [f"auto_multiview_{view_key}"]
    stems.extend(cgv.VIEW_OUTPUT_STEMS.get(view_key, [view_key]))
    if control_stem not in stems:
        stems.append(control_stem)
    seen: set[str] = set()
    unique_stems: list[str] = []
    for s in stems:
        if s not in seen:
            seen.add(s)
            unique_stems.append(s)

    for stem in unique_stems:
        name_variants = [stem]
        if overlay_suffix:
            name_variants.append(f"{stem}{overlay_suffix}")
        for candidate in name_variants:
            found = cgv.pick_video_in_dir(layered_dir, candidate)
            if found is not None:
                return found.resolve()

    expected = f"auto_multiview_{view_key}".lower()
    if layered_dir.is_dir():
        for p in layered_dir.iterdir():
            if not p.is_file() or p.suffix not in cgv.VIDEO_EXTENSIONS:
                continue
            st = p.stem.lower()
            if st == expected:
                return p.resolve()
            if overlay_suffix and st == f"{expected}{overlay_suffix.lower()}":
                return p.resolve()

    present = cgv.list_video_basenames(layered_dir)
    tried = unique_stems
    if overlay_suffix:
        tried = [f"{s} (+ {s}{overlay_suffix})" for s in unique_stems]
    hint = (
        f" Files in {layered_dir} with video extensions: {present}."
        if present
        else f" No video files found directly in {layered_dir} (non-recursive)."
    )
    raise FileNotFoundError(
        f"No layered video in {layered_dir} for view {view_key!r}; "
        f"tried stems {tried} with extensions {cgv.VIDEO_EXTENSIONS}.{hint}"
    )


def load_layered_rows(
    cgv: Any,
    *,
    layered_dir: Path,
    auto_multiview_path: Path | None,
    overlay_suffix: str,
) -> list[tuple[str, Path]]:
    if auto_multiview_path is not None:
        raw = json.loads(auto_multiview_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected JSON object in {auto_multiview_path}")
        spec = cgv.parse_auto_multiview_views(raw)
    else:
        spec = cgv.DEFAULT_MULTIVIEW

    rows: list[tuple[str, Path]] = []
    for view_key, control_name in spec:
        layered = resolve_layered_video(
            layered_dir,
            view_key,
            control_name,
            cgv,
            overlay_suffix=overlay_suffix,
        )
        rows.append((view_key, layered))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Score control-to-generated alignment from seven layered (control overlay + generated) "
            "videos using Cosmos-Reason2 offline inference."
        )
    )
    p.add_argument(
        "--layered-dir",
        type=Path,
        required=True,
        help=(
            "Directory with seven layered videos per view. "
            "Uses multiview stem rules plus optional overlay suffix (default _overlay), "
            "e.g. FRONT_CENTER_overlay.mp4."
        ),
    )
    p.add_argument(
        "--overlay-suffix",
        type=str,
        default="_overlay",
        help=(
            "Append to each candidate basename before resolving (default: _overlay). "
            "Use empty string for names without a suffix."
        ),
    )
    p.add_argument(
        "--auto-multiview-json",
        type=Path,
        default=None,
        help="Optional auto_multiview.json for view order; may supply prompt via JSON.",
    )
    p.add_argument(
        "generation_prompt",
        type=str,
        nargs="?",
        default="",
        help="Text prompt used for generation. Omit if using --prompt-file or JSON prompt.",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="UTF-8 file with the generation prompt.",
    )
    p.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional extra context (e.g. overlay recipe, checkpoint id).",
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
        help="Write only the Assistant markdown to this file.",
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
            "Provide the generation prompt (positional, --prompt-file, or prompt in JSON).",
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

    layered_dir = args.layered_dir.expanduser().resolve()
    if not layered_dir.is_dir():
        print(f"--layered-dir is not a directory: {layered_dir}", file=sys.stderr)
        sys.exit(1)

    cgv = _load_critique_generated_video()
    try:
        rows = load_layered_rows(
            cgv,
            layered_dir=layered_dir,
            auto_multiview_path=auto_path,
            overlay_suffix=args.overlay_suffix,
        )
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"Failed to resolve layered videos: {e}", file=sys.stderr)
        sys.exit(1)

    media_layout = build_media_layout_layered(rows)
    user_prompt = load_user_prompt(
        template,
        generation_prompt=generation_prompt,
        media_layout=media_layout,
        optional_notes=optional_notes,
    )

    videos = [lpath for _, lpath in rows]

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
