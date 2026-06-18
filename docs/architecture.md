# TactiGen — System Architecture

TactiGen turns a short football clip (or, when no video is available, a synthetic
clip of known player positions) into a structured, evidence-grounded tactical
report. The system is an ordered chain of eight agents that pass a single shared
`context` dictionary from stage to stage. Each agent is independent, logs its own
step, and is wrapped in error handling so a single failure degrades gracefully
instead of crashing the run.

## Pipeline diagram

```
                          Football clip (mp4/mov/avi)  OR  synthetic_clip_001.json
                                          |
                                          v
                              +-----------------------+
                              | 1. PreprocessingAgent |  OpenCV + FFmpeg
                              |   frames, segments    |
                              +-----------+-----------+
                                          |
                                          v
                              +-----------------------+
                              | 2. LocalizationAgent  |  YOLOv8 + homography
                              |   world coordinates   |  (synthetic fallback)
                              +-----------+-----------+
                                          |
                       +------------------+------------------+
                       v                                     v
        +----------------------------+         +----------------------------+
        | 4. TacticalAnalysisAgent   |         | 3. ActionPredictionAgent   |
        |  width/compactness/line/   |         |  Temporal Transformer or   |
        |  overload (NumPy, SciPy)   |         |  heuristic baseline        |
        +-------------+--------------+         +-------------+--------------+
                      |                                      |
                      +------------------+-------------------+
                                         v
                           +-------------------------------+
                           | 5. EvidenceCompilerAgent      |  confidence gate +
                           |    EvidenceObject (Pydantic)  |  pattern detection
                           +---------------+---------------+
                                           |
                                           v
                           +-------------------------------+
                           | 6. RAGRetrievalAgent          |  FAISS / MiniLM
                           |    retrieved passages         |  (keyword fallback)
                           +---------------+---------------+
                                           |
                                           v
                           +-------------------------------+
                           | 7. ReportGenerationAgent      |  GPT-4o or template
                           |    structured + text report   |
                           +---------------+---------------+
                                           |
                      +--------------------+--------------------+
                      v                                         v
        +----------------------------+            +----------------------------+
        | 8. VisualizationFeedback   |            |  Storage (Postgres / JSON) |
        |    heatmap + trajectories  |            |  DatabaseManager           |
        +-------------+--------------+            +----------------------------+
                      |
                      v
              Streamlit UI (6 tabs) + outputs/run_logs + MLflow
```

## Component table

| # | Agent | Input | Output (context keys) | Primary tools |
|---|-------|-------|-----------------------|---------------|
| 1 | PreprocessingAgent | video path or synthetic clip | `frame_paths`, `segments`, `video_meta`, `synthetic_path` | OpenCV, FFmpeg |
| 2 | LocalizationAgent | frames / synthetic clip | `localization_records`, `localization_confidence` | YOLOv8 (Ultralytics), OpenCV homography |
| 3 | ActionPredictionAgent | frames (+ metrics) | `anticipation_result` | PyTorch (ResNet18 + Transformer), heuristic baseline |
| 4 | TacticalAnalysisAgent | `localization_records` | `tactical_metrics` | NumPy, SciPy (ConvexHull) |
| 5 | EvidenceCompilerAgent | metrics + anticipation + confidence | `evidence` | Pydantic, confidence gate |
| 6 | RAGRetrievalAgent | detected pattern | `retrieved_passages` | FAISS, SentenceTransformers (MiniLM), LangChain |
| 7 | ReportGenerationAgent | evidence + passages + ontology | `structured_report`, `text_report` | OpenAI GPT-4o, template fallback |
| 8 | VisualizationFeedbackAgent | `localization_records` | `heatmap_path`, `trajectory_path` | Matplotlib, Seaborn |

## Data flow

The orchestrator (`orchestration/pipeline.py`) seeds `context` with `clip_id`,
`video_path`, and an empty `limitations` list, then runs each agent in order. The
deliberate ordering guarantees that the **Evidence Compiler always runs before the
language model**: tactical metrics and model confidences are frozen into a typed
`EvidenceObject` first, and only that object (plus retrieved knowledge passages)
is exposed to report generation. This is the structural mechanism that prevents
the LLM from inventing unsupported tactical claims.

Each run writes a JSON run-log to `outputs/run_logs/<run_id>.json` capturing the
clip id, video source, report status, and per-agent timings, and (when available)
logs parameters, metrics and the report artifact to MLflow.

## Tech stack

| Layer | Technology |
|-------|------------|
| Computer vision | OpenCV, Ultralytics YOLOv8, FFmpeg |
| Deep learning | PyTorch, torchvision (ResNet18 + Transformer encoder) |
| Tactical analysis | NumPy, SciPy |
| Data contracts | Pydantic v2 |
| Retrieval / RAG | FAISS, SentenceTransformers (all-MiniLM-L6-v2), LangChain |
| Generation | OpenAI GPT-4o (optional), deterministic template fallback |
| Visualization | Matplotlib, Seaborn |
| Storage | PostgreSQL + SQLAlchemy, JSON-lines fallback |
| Frontend | Streamlit |
| Experiment tracking | MLflow |
| Orchestration | Custom agent layer over a shared context dict |
