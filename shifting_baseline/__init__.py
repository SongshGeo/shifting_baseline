#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""shifting_baseline — comparing historical Chinese climate archives against
tree-ring reconstructions to study Shifting Baseline Syndrome (SBS).

Importing the package loads a local ``.env`` (if present) so that machine-specific
paths referenced from the Hydra configs (``${oc.env:SBS_ROOT}`` etc.) resolve
without any personal absolute path being committed. See ``.env.example``.
"""

from pathlib import Path

from dotenv import load_dotenv

# Populate os.environ from the repo-root .env when present (resolved relative to this
# file so it works from any working directory — notebooks, `python -m`, pytest).
# Never overrides variables already set in the real environment.
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_FILE if _ENV_FILE.exists() else None, override=False)
