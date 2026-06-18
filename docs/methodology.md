# TactiGen — Methodology

This document describes the modelling choices behind each stage of TactiGen and
the reasoning for them.

## 1. Player localization

Players are detected per frame with **YOLOv8** (`yolov8m.pt`, person class only).
YOLOv8 was chosen because it is a strong, well-maintained, real-time object
detector with a permissive deployment path and an extremely simple Python API
(`ultralytics`), which makes it practical for an MVP that must run on CPU. It
gives reliable person detections out of the box without any football-specific
training, so the team can focus engineering effort on the tactical reasoning layer
rather than on training a bespoke detector.

For each detection the bottom-center of the bounding box is taken as the player's
ground contact point (the feet), because that point projects most accurately onto
the pitch plane.

When YOLOv8 or its weights are unavailable, the localizer falls back to a
deterministic synthetic generator so the rest of the pipeline remains testable.

### Homography / coordinate mapping

Image pixels are mapped to pitch world coordinates (meters) by a planar
homography. Given four or more correspondences between image points and known
pitch points, `cv2.findHomography` (with RANSAC) estimates a 3x3 matrix `H` such
that a homogeneous image point `p = [u, v, 1]^T` maps to a pitch point
`q ~ H p`, recovered as `(x, y) = (q_0/q_2, q_1/q_2)`. When no calibration is
supplied, a linear fallback maps the image extents directly onto the pitch
rectangle (length 105 m, width 68 m); this is a coarse approximation used only so
that downstream metrics have meters to work with.

## 2. Action anticipation — Temporal Transformer

The anticipation model is a lightweight temporal transformer inspired by the
FAANTRA family of football action-anticipation models.

- **Frame features:** each of up to `sequence_len = 8` frames is encoded by a
  pretrained **ResNet18** with its classification head removed, yielding a
  512-dimensional feature vector per frame.
- **Projection + position:** features are linearly projected to `d_model = 256`
  and given sinusoidal positional encodings,
  `PE(pos, 2i) = sin(pos / 10000^{2i/d})`,
  `PE(pos, 2i+1) = cos(pos / 10000^{2i/d})`.
- **Encoder:** a 2-layer `TransformerEncoder` (`nhead = 4`,
  feed-forward = 512, dropout = 0.1, `batch_first`).
- **Head:** mean pooling over the time dimension followed by a linear layer to
  the 9 action classes (`pass, shot, cross, dribble, duel, clearance,
  interception, foul, ball_progression`).

Because no trained checkpoint ships with the repository (training requires
labelled SoccerNet sequences), inference falls back to the rule-based
`HeuristicBaseline`, which maps tactical metrics to a likely next action (for
example, a strong wide overload anticipates a `cross`). This keeps predictions
sensible and fully reproducible in the MVP while leaving a clean slot for a
trained model.

## 3. Tactical metrics (MVP scope: four metrics)

All metrics operate on world coordinates `(x, y)` in meters. Teams are split by
the halfway line: attackers have `x >= 52.5`, defenders `x < 52.5`.

- **Team width** `W = max_i x_i - min_i x_i` over the attacking team — the
  horizontal spread in meters.
- **Compactness** `C = clamp_{[0,1]}(1 - A_hull / A_pitch)`, where `A_hull` is the
  convex-hull area of the defending team (via `scipy.spatial.ConvexHull`) and
  `A_pitch = 105 x 68 = 7140` m^2. `C = 1` means perfectly compact, `C -> 0`
  means spread across the whole pitch.
- **Defensive line height** `H = median(x_(1..4))` of the four deepest defenders
  (smallest `x` when attacking right) — how high the back line sits.
- **Overload ratio** per channel `c`: `R_c = |A_c| / max(|D_c|, 1)`, where `A_c`
  and `D_c` are attackers and defenders whose `y` falls in channel `c`. Channels
  are left `[0, 22.67)`, center `[22.67, 45.33)`, right `[45.33, 68)` meters. The
  pipeline reports the channel with the highest ratio.

## 4. Evidence-first design

The single most important design decision is that an `EvidenceObject` is compiled
**before** any natural-language generation. The compiler maps metrics to an
ontology pattern (e.g. `right_side_overload`, `compact_block`,
`broken_defensive_line`) with a confidence score and runs a **confidence gate**:
if localization confidence < 0.65 or tactical-pattern confidence < 0.50, the
report is flagged `low_confidence_review` and a manual-review limitation is
attached. The generator may only reference fields that exist in this object, so
every tactical claim is traceable to a metric, a model output, or a retrieved
passage.

## 5. RAG strategy

The knowledge base is ten original tactical documents. `rag/build_index.py`
splits each into **400-character chunks with 80-character overlap**, embeds them
with **sentence-transformers/all-MiniLM-L6-v2** (384-dim), and stores them in a
**FAISS** index plus a JSON metadata file. Retrieval (`k = 4`) uses vector
similarity when FAISS is present and degrades to a keyword-overlap search over the
metadata when it is not, so retrieval always returns passages.

## 6. Prompt design

Two prompt files drive generation. The **system prompt** fixes ten strict rules:
every claim must cite a metric/timestamp/model output, missing or low-confidence
values must be stated, only ontology concepts may be used, the report must have
exactly six numbered sections, and a low-confidence banner must precede the
sections when the evidence is flagged. The **template** injects the clip id, the
serialized evidence object, the retrieved passages, and the valid ontology terms,
and asks the model to append a machine-readable `STRUCTURED_REPORT` JSON block.
Because that template contains a literal JSON example with braces, placeholder
substitution is done explicitly rather than with `str.format`, which would
misinterpret the braces.
