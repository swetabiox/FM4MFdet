# scripts/train_faster_rcnn_uni_midogpp.py

"""
Training script for Faster R-CNN with UNI backbone on MIDOG++.

This script:
- loads the MMDetection config file
- verifies the dataset split is patient-stratified before training starts
- verifies the config has an EarlyStoppingHook
- initialises the MMDetection runner and starts training

All model architecture, dataset, and optimisation settings live in the config
file. This script only wires the config to the runner and runs two safety
checks that protect the validity of the experiment.

PATIENT STRATIFICATION
----------------------
In MIDOG++ one slide is one patient. Patches from the same slide must never be
split across train/val/test, or validation and test metrics leak training
information. Before training, this script calls the same leakage check used by
`check_split_leakage.py`. If a slide crosses splits, training is aborted with a
clear message - generate the splits with `make_patient_splits.py` first.

EARLY STOPPING
--------------
Training no longer runs a fixed number of epochs. The config defines an
`EarlyStoppingHook` that monitors the validation metric (coco/bbox_mAP) and
stops once it stops improving, restoring the best checkpoint. `max_epochs` acts
only as an upper bound.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner


CONFIG_NAME = "faster_rcnn_uni_midogpp.py"


def _slide_id(file_name: str) -> str:
    """Derive the slide (== patient) id from a patch file name.

    Kept in sync with `scripts/make_patient_splits.py`. A trailing tile / patch
    / coordinate suffix is stripped from the file stem; whatever remains is the
    slide id.
    """
    import re
    stem = Path(file_name).stem
    tile_suffix = re.compile(
        r"(?:[_-](?:x\d+[_-]y\d+|tile[_-]?\d+|patch[_-]?\d+|"
        r"\d+[_-]\d+|\d+))+$",
        re.IGNORECASE,
    )
    stripped = tile_suffix.sub("", stem)
    return stripped if stripped else stem


def _slides_of(coco_path: Path):
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    slides = defaultdict(int)
    for img in coco["images"]:
        slides[_slide_id(img["file_name"])] += 1
    return set(slides.keys())


def verify_patient_stratification(cfg: Config, project_root: Path) -> None:
    """Abort if any slide appears in more than one split.

    The annotation paths are read straight from the config so this check always
    matches what training will actually load.
    """
    data_root = Path(cfg.train_dataloader["dataset"]["data_root"])
    if not data_root.is_absolute():
        data_root = project_root / data_root

    paths = {
        "train": data_root / cfg.train_dataloader["dataset"]["ann_file"],
        "val": data_root / cfg.val_dataloader["dataset"]["ann_file"],
        "test": data_root / cfg.test_dataloader["dataset"]["ann_file"],
    }

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print("WARNING: cannot verify patient stratification, annotation "
              "file(s) not found:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("Generate them with scripts/make_patient_splits.py.",
              file=sys.stderr)
        raise SystemExit(1)

    slides = {name: _slides_of(p) for name, p in paths.items()}
    leaks = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = slides[a] & slides[b]
        if shared:
            leaks.append((a, b, sorted(shared)))

    if leaks:
        print("ABORTING: dataset split is NOT patient-stratified.",
              file=sys.stderr)
        for a, b, shared in leaks:
            print(f"  {len(shared)} slide(s) shared between {a} and {b}: "
                  f"{shared[:5]}{' ...' if len(shared) > 5 else ''}",
                  file=sys.stderr)
        print("Re-generate splits with scripts/make_patient_splits.py.",
              file=sys.stderr)
        raise SystemExit(1)

    total = len(slides["train"] | slides["val"] | slides["test"])
    print(f"Patient stratification OK: {total} slides, none crossing splits.")


def verify_early_stopping(cfg: Config) -> None:
    """Warn if no EarlyStoppingHook is configured."""
    hooks = cfg.get("custom_hooks", []) or []
    has_es = any(
        (h.get("type") if isinstance(h, dict) else getattr(h, "type", None))
        == "EarlyStoppingHook"
        for h in hooks
    )
    if has_es:
        print("Early stopping is enabled (EarlyStoppingHook found).")
    else:
        print("WARNING: no EarlyStoppingHook in config; training will run for "
              "the full max_epochs.", file=sys.stderr)


def main():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    cfg_path = project_root / "configs" / CONFIG_NAME
    cfg = Config.fromfile(str(cfg_path))

    init_default_scope("mmdet")

    # Safety checks before any GPU work.
    verify_patient_stratification(cfg, project_root)
    verify_early_stopping(cfg)

    # Optional: resume training from the latest checkpoint.
    # cfg.resume = True

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
