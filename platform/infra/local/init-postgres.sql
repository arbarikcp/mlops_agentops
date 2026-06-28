-- Bootstrap databases and per-service users.
-- Threat model E-01: separate roles per component.

-- MLflow gets its own database (created by POSTGRES_DB env var).
-- Additional databases for future components:

CREATE DATABASE IF NOT EXISTS feast;
CREATE DATABASE IF NOT EXISTS evidently;

-- Read-only role for monitoring queries
CREATE ROLE mlops_reader WITH LOGIN PASSWORD 'reader_placeholder';
GRANT CONNECT ON DATABASE mlflow TO mlops_reader;
GRANT USAGE ON SCHEMA public TO mlops_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mlops_reader;

-- NOTE: Replace 'reader_placeholder' via secrets manager before production.
