-- Initialize Airflow database within PostgreSQL instance
SELECT 'CREATE DATABASE airflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec

GRANT ALL PRIVILEGES ON DATABASE airflow TO mlflow;

-- Initialize Observability & Drift Metrics Store in mlflow database
\connect mlflow;

CREATE TABLE IF NOT EXISTS monitoring_drift_records (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    batch_id VARCHAR(100),
    feature_name VARCHAR(100),
    metric_type VARCHAR(50),
    statistic_value DOUBLE PRECISION,
    threshold_value DOUBLE PRECISION,
    p_value DOUBLE PRECISION,
    drift_detected BOOLEAN,
    reference_sample_count INT,
    current_sample_count INT,
    details JSONB
);

CREATE INDEX IF NOT EXISTS idx_drift_timestamp ON monitoring_drift_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_drift_batch_id ON monitoring_drift_records(batch_id);
