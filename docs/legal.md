# TactiGen — Legal and Licensing Notes

This document summarizes the terms attached to the external data, models, and
services TactiGen can use. It is informational, not legal advice; always consult
the upstream licenses before any distribution or commercial use.

## SoccerNet (video and action labels)
SoccerNet data is distributed for **research purposes only**. Access to the video
material requires registration and agreement to SoccerNet's terms (a password /
NDA-style agreement for the raw videos). Do not redistribute SoccerNet videos.
Use them only for research and cite the SoccerNet papers. See
https://www.soccer-net.org/ for current terms.

## StatsBomb Open Data
StatsBomb's open event data is released for public use under the **StatsBomb
Public Data User Agreement**, which permits research, education, and public
analysis **with attribution to StatsBomb** and prohibits certain commercial uses.
The repository's `source_registry.csv` tags this source for convenience, but the
authoritative terms are StatsBomb's user agreement in their `open-data`
repository — review `LICENSE`/`README` at
https://github.com/statsbomb/open-data and attribute StatsBomb wherever their data
is used.

## YOLOv8 / Ultralytics
Ultralytics YOLOv8 is licensed under **AGPL-3.0**. AGPL-3.0 is a strong copyleft
license: if you run a modified version as a network service, you must make the
corresponding source available to users of that service. For closed-source or
commercial deployment that cannot meet AGPL obligations, obtain a commercial
**Ultralytics Enterprise License**. Factor this into any productization decision.

## OpenAI API (GPT-4o)
Report generation can call the OpenAI API. Use is governed by OpenAI's terms of
use and usage policies. Clip-derived evidence (metrics, patterns, timestamps) is
sent to the API when generation is enabled; do not send personal data or anything
you are not permitted to transmit. The API key must be supplied via the `.env`
file and is never committed (see `.gitignore`). The system runs fully offline via
the deterministic template fallback when no key is set.

## Raw football video in the repository
Broadcast football video is copyrighted by leagues, clubs, and broadcasters.
**Do not commit raw video** to a public repository. `.gitignore` already excludes
`*.mp4`, `*.mkv`, `*.mov`, and `*.avi` under `data/`. Store video outside version
control (local disk, private object storage, or DVC) and keep only derived,
non-infringing artifacts (coordinates, metrics, reports) in the repo.

## Model weights
Pretrained weights (YOLOv8, ResNet18, any trained transformer checkpoint) are not
committed; they are excluded by `.gitignore` (`*.pt`, `*.pth`, `*.ckpt`,
`outputs/models/`). Distribute large weights via releases or a model registry, and
respect each weight's own license.
