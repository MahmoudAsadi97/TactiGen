# TactiGen — Limitations

TactiGen is a research-preview MVP. Its conclusions should be treated as
decision-support, not ground truth. The most important limitations follow.

## Localization: YOLOv8 vs. proper football localization
The system uses a generic YOLOv8 person detector, not a football-specific
tracker. It does not perform multi-object tracking with persistent identities,
does not resolve occlusions or player congestion, and cannot distinguish players
from referees, coaches, or ball boys. A production system would use a detector
fine-tuned on football broadcast footage plus a tracker (e.g. ByteTrack) to
maintain stable identities across frames.

## Anticipation: heuristic vs. full FAANTRA
No trained transformer checkpoint ships with the repository, so action
anticipation falls back to a rule-based heuristic over tactical metrics. The
transformer architecture is implemented and ready, but until it is trained on
labelled SoccerNet sequences its predictions are not learned and should be read as
plausibility heuristics rather than model outputs.

## Homography approximation
Without per-clip camera calibration, image-to-pitch mapping uses a crude linear
fallback that assumes the frame maps directly onto the full pitch rectangle. Real
broadcast footage has moving, zooming cameras and partial pitch views, so
world-coordinate estimates — and therefore every metric derived from them — carry
significant error unless a proper homography is calibrated per clip.

## Knowledge base size
The RAG corpus is ten original documents covering core concepts. It is
intentionally small, so retrieval can only ground reports in those concepts;
queries outside this scope return the nearest available passage, which may be only
loosely relevant. Expanding and curating the knowledge base is required before the
grounding claims generalize.

## Residual LLM hallucination risk
The evidence-first design and strict prompt substantially reduce, but do not
eliminate, hallucination. When GPT-4o generation is enabled the model can still
phrase claims loosely or over-interpret the evidence. The confidence gate,
six-section structure, and groundedness checks are mitigations, not guarantees;
human review remains necessary, especially for any clip flagged
`low_confidence_review`.

## Clip-length constraint
The design targets short clips (roughly 15-30 seconds, analysed in ~5-6 second
windows). It does not model full-match context, score state, fatigue over 90
minutes, or longer tactical trends. Conclusions are local to the analysed window.

## Team identification
Teams are split by a naive halfway-line heuristic (attackers right of center,
defenders left), not by jersey/color clustering. This is only valid for the
synthetic setup and for clips that happen to match that spatial assumption. Real
analysis needs team assignment by appearance, and direction-of-play detection, to
attribute metrics to the correct side.
