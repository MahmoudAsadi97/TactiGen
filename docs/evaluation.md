# TactiGen — Evaluation

TactiGen is evaluated per module. The four scripts in `evaluation/` read the most
recent run from `outputs/run_logs/` and the artifacts in `outputs/reports/`,
compute the metrics below, and write `eval_<module>_<clip_id>.json` back into
`outputs/run_logs/`. Because the MVP ships no trained checkpoint and no labelled
ground truth, accuracy-style metrics return explicit placeholders with a note,
while the structural and grounding metrics are computed for real.

## Metric definitions

### Localization — mAP@delta and detection stats
For action/detection quality with ground truth, mean Average Precision at a
tolerance is used. A detection counts as a true positive when it matches a ground
truth within a spatial tolerance `delta` (IoU >= 0.5 for boxes). Precision and
recall are `P = TP / (TP + FP)` and `R = TP / (TP + FN)`; Average Precision is the
area under the precision-recall curve, and `mAP@delta` averages AP over classes.
Without labelled frames the script reports the **confidence distribution**
(mean, std, min, max, median), **mean detections per frame**, and a
**visual-inspection flag** (raised when mean confidence < 0.65).

### Retrieval — precision@k
`precision@k = (relevant items in top k) / k`. TactiGen retrieves `k = 4`
passages per query. The RAG script also reports **citation coverage**
(`cited_source_ids / retrieved_source_ids`) and the **unsupported-claim rate**
(report sentences containing neither a number nor a source_id, divided by total
sentences).

### Reports — groundedness and hallucination rate
`groundedness = (sentences referencing a number or timestamp) / (total sentences)`.
`hallucination_rate = (claims not supported by evidence or retrieval) / (total claims)`;
in the MVP this is approximated by the unsupported-claim rate and by the count of
analyst-flagged hallucinations from `outputs/feedback/`. The report script also
verifies the **six-section structure** and the **presence of confidence values**.

## Baselines (action anticipation)
- **Majority class** — always predicts `pass`, the most frequent football action.
- **Last action** — predicts the most recently observed event type.
- **Heuristic** — rule-based mapping from tactical metrics to the next action
  (e.g. wide overload -> `cross`, low compactness -> `shot`).
- **Temporal Transformer** — the learned model; in the MVP it falls back to the
  heuristic because no checkpoint is shipped.

A model is only interesting if it beats the majority-class and last-action
baselines on top-1 accuracy once labels are available.

## Evaluation protocol
1. Run the pipeline on a clip (or the synthetic clip) to produce run-logs and reports.
2. Run all four evaluation scripts.
3. Inspect the `eval_*` JSON files in `outputs/run_logs/`.
4. For accuracy metrics, align predictions to SoccerNet action-spotting labels.
5. Collect analyst feedback through the Streamlit feedback tab for human metrics.

## Expected performance ranges (targets, not yet measured)
These are indicative targets drawn from the literature for a fully trained system,
not measured results for this MVP:

| Metric | Indicative target |
|--------|-------------------|
| Person detection mAP@0.5 (YOLOv8m, broadcast) | ~0.55-0.75 |
| Action anticipation top-1 (9 classes) vs majority | beat ~0.40-0.45 baseline |
| Retrieval precision@4 (in-domain queries) | ~0.75-1.0 |
| Report groundedness | >= 0.7 |
| Hallucination rate | <= 0.1 |

## Human evaluation
Analysts rate each report on a 1-5 scale for **pattern accuracy** and
**recommendation usefulness**, and flag any **unsupported claim**, with an optional
comment. Feedback is captured in the Streamlit UI and persisted (to
`analyst_feedback` in PostgreSQL, or to JSON when the database is offline) so that
groundedness and hallucination findings can be validated against human judgement.
