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

"""Run Cosmos-Reason2 offline critique on Cosmos Transfer outputs (multiview control + generated)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from string import Template
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
# Directory on PYTHONPATH so `python -m cosmos_reason2_utils.script.inference` works without a venv entrypoint.
COSMOS_REASON2_UTILS_ROOT = ROOT / "cosmos_reason2_utils"
DEFAULT_TEMPLATE = ROOT / "prompts" / "cosmos_transfer_critique.yaml"
NUM_CAMERAS = 7

# Cosmos Transfer `auto_multiview.json` uses view keys; control files use uppercase rig names.
# Generated outputs may use `auto_multiview_<view_key>.mp4` (e.g. auto_multiview_cross_left.mp4)
# or the view key / rig stem alone (e.g. front_wide.mp4, FRONT_CENTER.mp4).
VIEW_OUTPUT_STEMS: dict[str, list[str]] = {
    "front_wide": ["front_wide", "FRONT_CENTER"],
    "rear": ["rear", "REAR_CENTER"],
    "rear_left": ["rear_left", "REAR_LEFT"],
    "rear_right": ["rear_right", "REAR_RIGHT"],
    "cross_left": ["cross_left", "FRONT_LEFT"],
    "cross_right": ["cross_right", "FRONT_RIGHT"],
    "front_tele": ["front_tele", "FRONT_CENTER_NARROW"],
}

# Default view order and control filenames when no auto_multiview.json is passed.
DEFAULT_MULTIVIEW: list[tuple[str, str]] = [
    ("front_wide", "FRONT_CENTER.mp4"),
    ("rear", "REAR_CENTER.mp4"),
    ("rear_left", "REAR_LEFT.mp4"),
    ("rear_right", "REAR_RIGHT.mp4"),
    ("cross_left", "FRONT_LEFT.mp4"),
    ("cross_right", "FRONT_RIGHT.mp4"),
    ("front_tele", "FRONT_CENTER_NARROW.mp4"),
]

AUTO_MULTIVIEW_SKIP_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "prompt",
        "prompt_path",
        "negative_prompt",
        "seed",
        "guidance",
        "num_conditional_frames",
        "control_weight",
        "fps",
        "num_steps",
        "enable_autoregressive",
        "num_chunks",
        "chunk_overlap",
        "save_combined_views",
    }
)

VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".MP4", ".mov", ".MOV", ".webm", ".WEBM")


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
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="critique_", text=True)
    os.close(fd)
    p = Path(path)
    p.write_text(yaml.safe_dump({"user_prompt": user_prompt}), encoding="utf-8")
    return p


def pick_video_in_dir(directory: Path, stem: str) -> Path | None:
    for ext in VIDEO_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    exact = directory / stem
    if exact.is_file():
        return exact
    return None


def list_video_basenames(directory: Path, *, limit: int = 40) -> list[str]:
    """Basenames of files under directory that look like video (shallow, non-recursive)."""
    names: list[str] = []
    if not directory.is_dir():
        return names
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix in VIDEO_EXTENSIONS:
            names.append(p.name)
        if len(names) >= limit:
            break
    return names


def resolve_generated_video(generated_dir: Path, view_key: str, control_basename: str) -> Path:
    control_stem = Path(control_basename).stem
    stems = [f"auto_multiview_{view_key}"]
    stems.extend(VIEW_OUTPUT_STEMS.get(view_key, [view_key]))
    if control_stem not in stems:
        stems.append(control_stem)
    seen: set[str] = set()
    unique_stems = []
    for s in stems:
        if s not in seen:
            seen.add(s)
            unique_stems.append(s)
    for stem in unique_stems:
        found = pick_video_in_dir(generated_dir, stem)
        if found is not None:
            return found
    expected = f"auto_multiview_{view_key}".lower()
    if generated_dir.is_dir():
        for p in generated_dir.iterdir():
            if (
                p.is_file()
                and p.suffix in VIDEO_EXTENSIONS
                and p.stem.lower() == expected
            ):
                return p.resolve()
    present = list_video_basenames(generated_dir)
    hint = (
        f" Files in {generated_dir} with video extensions: {present}."
        if present
        else f" No video files found directly in {generated_dir} (non-recursive)."
    )
    raise FileNotFoundError(
        f"No generated video in {generated_dir} for view {view_key!r}; "
        f"tried stems {unique_stems} with extensions {VIDEO_EXTENSIONS}.{hint}"
    )


def parse_auto_multiview_views(data: dict[str, Any]) -> list[tuple[str, str]]:
    """Return [(view_key, control_basename), ...] in JSON key order."""
    views: list[tuple[str, str]] = []
    for key, value in data.items():
        if key in AUTO_MULTIVIEW_SKIP_KEYS or not isinstance(value, dict):
            continue
        cp = value.get("control_path")
        if not cp or not isinstance(cp, str):
            continue
        basename = Path(cp).name
        views.append((key, basename))
    if len(views) != NUM_CAMERAS:
        raise ValueError(
            f"Expected {NUM_CAMERAS} views with control_path in auto_multiview JSON; found {len(views)}."
        )
    return views


def load_multiview_pairs(
    *,
    control_dir: Path,
    generated_dir: Path,
    auto_multiview_path: Path | None,
) -> list[tuple[str, Path, Path]]:
    if auto_multiview_path is not None:
        raw = json.loads(auto_multiview_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected JSON object in {auto_multiview_path}")
        spec = parse_auto_multiview_views(raw)
    else:
        spec = DEFAULT_MULTIVIEW

    pairs: list[tuple[str, Path, Path]] = []
    for view_key, control_name in spec:
        ctrl = control_dir / Path(control_name).name
        if not ctrl.is_file():
            raise FileNotFoundError(f"Control video missing for {view_key}: {ctrl}")
        gen = resolve_generated_video(generated_dir, view_key, control_name)
        pairs.append((view_key, ctrl.resolve(), gen.resolve()))
    return pairs


def build_media_layout_multiview(pairs: list[tuple[str, Path, Path]]) -> str:
    lines = [
        f"There are **{len(pairs) * 2}** videos: **seven camera pairs**. "
        "For each camera, the **first** clip of the pair is the **control** visualization "
        "(bounding boxes and map / lanes); the **second** is the **Cosmos Transfer generated** output.",
        "",
    ]
    i = 1
    for view_key, cpath, gpath in pairs:
        lines.append(
            f"**{view_key}** — video {i}: control `{cpath.name}`; "
            f"video {i + 1}: generated `{gpath.name}`."
        )
        i += 2
    lines.append("")
    lines.append(
        "Compare each generated clip to the control clip that immediately precedes it; "
        "then comment on consistency across cameras and with the text prompt."
    )
    return "\n".join(lines)


def build_media_layout_legacy_eight() -> str:
    lines = [
        "There are **8** videos: **7** control-camera streams then **1** generated output.",
        "",
    ]
    for i in range(NUM_CAMERAS):
        lines.append(
            f"{i + 1}. **Control** — camera slot {i} (boxes + map / lanes)."
        )
    lines.append(
        f"{NUM_CAMERAS + 1}. **Generated** — single Cosmos Transfer output to critique."
    )
    return "\n".join(lines)


def build_media_layout_legacy_single() -> str:
    return (
        "There is **one** video: the **generated** output only "
        "(no separate control-camera inputs in this run)."
    )


def read_control_video_paths(list_file: Path) -> list[Path]:
    text = list_file.read_text(encoding="utf-8")
    paths: list[Path] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(Path(line).expanduser())
    return paths


def build_offline_command(
    *,
    videos: list[Path],
    input_yaml: Path,
    model: str,
    max_model_len: int,
    reasoning: bool,
    verbose: bool,
    output_debug: Path | None,
) -> tuple[list[str], dict[str, str] | None]:
    """Build CLI argv and optional env for subprocess (PYTHONPATH when not using the entrypoint)."""
    inference = shutil.which("cosmos-reason2-inference")
    run_env: dict[str, str] | None = None
    if inference:
        cmd = [inference, "offline"]
    else:
        cmd = [sys.executable, "-m", "cosmos_reason2_utils.script.inference", "offline"]
        pkg = COSMOS_REASON2_UTILS_ROOT.resolve()
        if not (pkg / "cosmos_reason2_utils").is_dir():
            raise FileNotFoundError(
                f"Expected cosmos-reason2-utils package under {pkg}; "
                "install the repo (uv sync) or run from the repository root."
            )
        run_env = os.environ.copy()
        prev = run_env.get("PYTHONPATH", "")
        run_env["PYTHONPATH"] = str(pkg) + (os.pathsep + prev if prev else "")
    cmd += [
        "--model",
        model,
        "--max-model-len",
        str(max_model_len),
        "-i",
        str(input_yaml),
    ]
    for v in videos:
        cmd.extend(["--videos", str(v.resolve())])
    if reasoning:
        cmd.append("--reasoning")
    if verbose:
        cmd.append("-v")
    if output_debug is not None:
        output_debug.mkdir(parents=True, exist_ok=True)
        cmd += ["-o", str(output_debug.resolve())]
    return cmd, run_env


# Must match cosmos_reason2_utils.script.inference.SEPARATOR.
_INFERENCE_SEP_LINE = "-" * 20


def extract_assistant_markdown(captured_stdout: str) -> tuple[str, bool]:
    """Parse offline inference stdout; return (body, True) if an Assistant block was found."""
    lines = captured_stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip() != "Assistant:":
            continue
        if i == 0 or lines[i - 1].strip() != _INFERENCE_SEP_LINE:
            continue
        j = i + 1
        out: list[str] = []
        while j < len(lines):
            if lines[j].strip() == _INFERENCE_SEP_LINE:
                break
            ln = lines[j]
            if ln.startswith("  "):
                out.append(ln[2:])
            else:
                out.append(ln)
            j += 1
        return "\n".join(out).strip(), True
    return captured_stdout.strip(), False


def run_offline_inference(
    cmd: list[str],
    run_env: dict[str, str] | None,
    *,
    output_text: Path | None,
) -> int:
    """Run offline inference. If output_text is set, write only the Assistant markdown there; console still shows full stdout."""
    if output_text is None:
        return subprocess.run(cmd, env=run_env).returncode
    out_path = output_text.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    parts: list[str] = []
    for line in proc.stdout:
        sys.stdout.write(line)
        parts.append(line)
    code = proc.wait()
    full = "".join(parts)
    body, found = extract_assistant_markdown(full)
    if not found:
        print(
            "Warning: no 'Assistant:' block found in inference stdout; "
            "writing full stdout to --output file.",
            file=sys.stderr,
        )
    out_path.write_text((body + "\n") if body else "", encoding="utf-8")
    return code


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Critique Cosmos Transfer multiview results with Cosmos-Reason2. "
            "Primary mode: --control-dir and --generated-dir (seven cameras each), "
            "plus a text prompt. Optional auto_multiview.json fixes view order and control filenames. "
            "Requires GPU and cosmos-reason2-inference offline."
        )
    )
    p.add_argument(
        "--control-dir",
        type=Path,
        default=None,
        help=(
            "Directory of seven control videos (e.g. FRONT_CENTER.mp4). "
            "Use with --generated-dir for multiview critique."
        ),
    )
    p.add_argument(
        "--generated-dir",
        type=Path,
        default=None,
        help=(
            "Directory of seven generated videos per camera "
            "(e.g. front_wide.mp4 matching view keys from auto_multiview.json)."
        ),
    )
    p.add_argument(
        "--auto-multiview-json",
        type=Path,
        default=None,
        help=(
            "Cosmos Transfer auto_multiview.json: defines view order and control_path basenames. "
            "Prompt may be read from this file if you omit --prompt-file and the positional prompt."
        ),
    )
    p.add_argument(
        "generated_video",
        type=Path,
        nargs="?",
        default=None,
        help="Legacy: single generated file when not using --control-dir / --generated-dir.",
    )
    p.add_argument(
        "generation_prompt",
        type=str,
        nargs="?",
        default="",
        help="Text prompt used for generation. Omit with --prompt-file or when JSON supplies prompt.",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Read generation prompt from a UTF-8 file.",
    )
    p.add_argument(
        "--control-video",
        type=Path,
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Legacy: pass seven control videos in camera order 0..6. "
            "Uses a single generated_video plus seven controls (not multiview dirs)."
        ),
    )
    p.add_argument(
        "--control-videos-list",
        type=Path,
        default=None,
        help=f"Legacy: file with {NUM_CAMERAS} control video paths, one per line.",
    )
    p.add_argument(
        "--control",
        type=str,
        default="",
        help="Optional free-text operator notes.",
    )
    p.add_argument(
        "--control-file",
        type=Path,
        default=None,
        help="Optional operator notes from a UTF-8 file.",
    )
    p.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="YAML template ($generation_prompt, $media_layout, $optional_notes).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason2-2B",
        help="Hugging Face model id or local path.",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Passed through to offline inference.",
    )
    p.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable chain-of-thought style reasoning.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose inference logging.")
    p.add_argument(
        "-o",
        "--output-debug",
        type=Path,
        default=None,
        help="Save preprocessed vision tensors under this directory.",
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

    multiview = args.control_dir is not None and args.generated_dir is not None
    if args.control_dir is not None and args.generated_dir is None:
        print("When using --control-dir, also pass --generated-dir.", file=sys.stderr)
        sys.exit(1)
    if args.generated_dir is not None and args.control_dir is None:
        print("When using --generated-dir, also pass --control-dir.", file=sys.stderr)
        sys.exit(1)

    if multiview and args.generated_video is not None:
        print("Do not pass a positional generated_video when using --control-dir / --generated-dir.", file=sys.stderr)
        sys.exit(1)
    if multiview and (args.control_video or args.control_videos_list):
        print("Multiview mode uses directories; do not use --control-video or --control-videos-list.", file=sys.stderr)
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
                p = meta.get("prompt")
                if isinstance(p, str) and p.strip():
                    json_prompt = p.strip()
        except (json.JSONDecodeError, OSError) as e:
            print(f"Could not read JSON prompt from {auto_path}: {e}", file=sys.stderr)
            sys.exit(1)

    if args.prompt_file is not None:
        pf = args.prompt_file.expanduser().resolve()
        if not pf.is_file():
            print(f"Prompt file not found: {pf}", file=sys.stderr)
            sys.exit(1)
        generation_prompt = pf.read_text(encoding="utf-8")
    elif args.generation_prompt.strip():
        generation_prompt = args.generation_prompt
    elif json_prompt is not None:
        generation_prompt = json_prompt
    else:
        print(
            "Provide a generation prompt (positional, --prompt-file, or prompt field in --auto-multiview-json).",
            file=sys.stderr,
        )
        sys.exit(1)

    if not generation_prompt.strip():
        print("Generation prompt is empty.", file=sys.stderr)
        sys.exit(1)

    control_notes = args.control
    if args.control_file is not None:
        cf = args.control_file.expanduser().resolve()
        if not cf.is_file():
            print(f"Control notes file not found: {cf}", file=sys.stderr)
            sys.exit(1)
        control_notes = cf.read_text(encoding="utf-8")

    optional_notes = ""
    if control_notes.strip():
        optional_notes = "**Additional operator notes:**\n" + control_notes.strip()

    video_args: list[Path]
    media_layout: str

    if multiview:
        cdir = args.control_dir.expanduser().resolve()
        gdir = args.generated_dir.expanduser().resolve()
        if not cdir.is_dir():
            print(f"--control-dir is not a directory: {cdir}", file=sys.stderr)
            sys.exit(1)
        if not gdir.is_dir():
            print(f"--generated-dir is not a directory: {gdir}", file=sys.stderr)
            sys.exit(1)
        try:
            pairs = load_multiview_pairs(
                control_dir=cdir,
                generated_dir=gdir,
                auto_multiview_path=auto_path,
            )
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"Multiview pairing failed: {e}", file=sys.stderr)
            sys.exit(1)
        media_layout = build_media_layout_multiview(pairs)
        video_args = []
        for _, cpath, gpath in pairs:
            video_args.append(cpath)
            video_args.append(gpath)
    else:
        if args.generated_video is None:
            print(
                "Either pass --control-dir and --generated-dir, or a positional generated_video (legacy).",
                file=sys.stderr,
            )
            sys.exit(1)
        generated = args.generated_video.expanduser().resolve()
        if not generated.is_file():
            print(f"Generated video not found: {generated}", file=sys.stderr)
            sys.exit(1)

        from_cli = list(args.control_video) if args.control_video else []
        from_file: list[Path] = []
        if args.control_videos_list is not None:
            lf = args.control_videos_list.expanduser().resolve()
            if not lf.is_file():
                print(f"Control videos list not found: {lf}", file=sys.stderr)
                sys.exit(1)
            from_file = read_control_video_paths(lf)

        if from_cli and from_file:
            print("Use either --control-video or --control-videos-list, not both.", file=sys.stderr)
            sys.exit(1)

        control_paths = from_cli if from_cli else from_file
        if len(control_paths) not in (0, NUM_CAMERAS):
            print(
                f"Legacy mode: provide exactly {NUM_CAMERAS} control videos or none; got {len(control_paths)}.",
                file=sys.stderr,
            )
            sys.exit(1)

        for i, cp in enumerate(control_paths):
            p = cp.expanduser().resolve()
            if not p.is_file():
                print(f"Control video {i} not found: {p}", file=sys.stderr)
                sys.exit(1)
            control_paths[i] = p

        if control_paths:
            media_layout = build_media_layout_legacy_eight()
            video_args = [p.resolve() for p in control_paths] + [generated.resolve()]
        else:
            media_layout = build_media_layout_legacy_single()
            video_args = [generated.resolve()]

    user_prompt = load_user_prompt(
        template,
        generation_prompt=generation_prompt,
        media_layout=media_layout,
        optional_notes=optional_notes,
    )

    tmp_yaml = write_prompt_file(user_prompt)
    try:
        cmd, run_env = build_offline_command(
            videos=video_args,
            input_yaml=tmp_yaml,
            model=args.model,
            max_model_len=args.max_model_len,
            reasoning=args.reasoning,
            verbose=args.verbose,
            output_debug=args.output_debug,
        )
        if args.verbose:
            print("Command:", " ".join(cmd), file=sys.stderr)
        code = run_offline_inference(cmd, run_env, output_text=args.output)
        raise SystemExit(code)
    finally:
        tmp_yaml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
