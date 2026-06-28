#!/bin/sh
# Bootstrap MinIO buckets. Runs once via minio-init service.
# Threat model I-03: buckets are private by default (no public policy set).

set -e

MC="mc"
ALIAS="local"

# Wait for MinIO to be ready
until $MC alias set $ALIAS http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" > /dev/null 2>&1; do
  echo "Waiting for MinIO..."
  sleep 2
done

# Create buckets (private by default — no mc policy set public)
for bucket in mlflow-artifacts dvc-data feast-offline monitoring-reports; do
  if ! $MC ls "$ALIAS/$bucket" > /dev/null 2>&1; then
    $MC mb "$ALIAS/$bucket"
    echo "Created bucket: $bucket"
  else
    echo "Bucket already exists: $bucket"
  fi
done

echo "MinIO bucket init complete."
