CREATE TABLE IF NOT EXISTS raw_video_clips (
    clip_id TEXT PRIMARY KEY,
    file_path TEXT,
    fps FLOAT,
    duration_seconds FLOAT,
    resolution TEXT,
    uploaded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw_player_localizations (
    id SERIAL PRIMARY KEY,
    clip_id TEXT REFERENCES raw_video_clips(clip_id) ON DELETE CASCADE,
    frame_id INTEGER,
    timestamp FLOAT,
    player_id INTEGER,
    x_world FLOAT,
    y_world FLOAT,
    confidence FLOAT,
    bbox JSONB,
    source TEXT DEFAULT 'yolov8'
);

CREATE TABLE IF NOT EXISTS raw_action_predictions (
    id SERIAL PRIMARY KEY,
    clip_id TEXT REFERENCES raw_video_clips(clip_id) ON DELETE CASCADE,
    timestamp FLOAT,
    predicted_action TEXT,
    confidence FLOAT,
    model_used TEXT,
    anticipation_window_seconds INTEGER DEFAULT 5
);

CREATE TABLE IF NOT EXISTS stg_tactical_metrics (
    id SERIAL PRIMARY KEY,
    clip_id TEXT REFERENCES raw_video_clips(clip_id) ON DELETE CASCADE,
    time_window_start FLOAT,
    time_window_end FLOAT,
    team_width_meters FLOAT,
    compactness_score FLOAT,
    defensive_line_height_meters FLOAT,
    overload_ratio FLOAT,
    overload_channel TEXT,
    attacking_count INTEGER,
    defending_count INTEGER
);

CREATE TABLE IF NOT EXISTS mart_clip_tactical_summary (
    clip_id TEXT PRIMARY KEY,
    main_phase TEXT,
    primary_pattern TEXT,
    anticipated_action TEXT,
    confidence FLOAT,
    report_text TEXT,
    structured_report JSONB,
    report_status TEXT DEFAULT 'ok',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analyst_feedback (
    id SERIAL PRIMARY KEY,
    clip_id TEXT,
    report_id TEXT,
    pattern_accuracy_score INTEGER CHECK (pattern_accuracy_score BETWEEN 1 AND 5),
    recommendation_usefulness_score INTEGER CHECK (recommendation_usefulness_score BETWEEN 1 AND 5),
    hallucination_flag BOOLEAN DEFAULT FALSE,
    reviewer_comment TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
