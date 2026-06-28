#!/usr/bin/env bash
# Reproducibility Gate Dry-Run — Day 14 deliverable.
#
# Usage:
#   bash scripts/verify_reproducibility.sh <run_id>
#
# Pass condition: all 6 checks pass. Exit code 0 = pass, 1 = fail.
#
# Run via make:
#   make reproduce-check     (uses run_id from metrics/mlflow_run_id.txt)

set -euo pipefail

RUN_ID="${1:?Usage: $0 <run_id>}"
TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
SEED=42
PASS=0
FAIL=0

_pass() { echo "  ✅ $*"; PASS=$((PASS + 1)); }
_fail() { echo "  ❌ $*"; FAIL=$((FAIL + 1)); }
_section() { echo ""; echo "── $* ──"; }

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Reproducibility Gate Dry-Run                ║"
echo "║  run_id: ${RUN_ID:0:16}...                   ║"
echo "╚══════════════════════════════════════════════╝"

# ① MLflow run is retrievable
_section "① MLflow Run Metadata"
python3 - <<EOF
import mlflow, sys
mlflow.set_tracking_uri("$TRACKING_URI")
try:
    run = mlflow.get_run("$RUN_ID")
    print(f"  Status: {run.info.status}")
    print(f"  Experiment: {run.info.experiment_id}")
    auc = run.data.metrics.get("roc_auc")
    print(f"  AUC: {auc}")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
EOF
_pass "Run retrievable from MLflow"

# ② Git commit tag exists
_section "② Git Commit Tag"
GIT_COMMIT=$(python3 -c "
import mlflow
mlflow.set_tracking_uri('$TRACKING_URI')
run = mlflow.get_run('$RUN_ID')
print(run.data.tags.get('git.commit', ''))
" 2>/dev/null)

if [ -z "$GIT_COMMIT" ]; then
    _fail "git.commit tag is missing from run (fix: add git.commit tag in mlflow_train.py)"
else
    _pass "git.commit = $GIT_COMMIT"
    # Check commit exists in repo
    if git show "$GIT_COMMIT" --format="" --name-only > /dev/null 2>&1; then
        _pass "Git commit $GIT_COMMIT exists in repository"
    else
        _fail "Git commit $GIT_COMMIT NOT found in repository"
    fi
fi

# ③ DVC data accessible
_section "③ DVC Data"
if dvc status --cloud > /dev/null 2>&1; then
    _pass "DVC remote is accessible"
else
    _fail "DVC remote not accessible — is MinIO running? (make up)"
fi

if [ -f "data/raw/credit_card_default.csv" ] || dvc pull data/raw/credit_card_default.csv.dvc > /dev/null 2>&1; then
    _pass "Raw dataset retrievable via DVC"
else
    _fail "Cannot pull raw dataset — check dvc push was run"
fi

# ④ Environment lockfile
_section "④ Environment Lockfile"
if [ -f "uv.lock" ]; then
    _pass "uv.lock exists (environment reproducible)"
else
    _fail "uv.lock missing — run 'uv sync' and commit uv.lock"
fi

# ⑤ Determinism check
_section "⑤ Training Determinism"
PYTHONHASHSEED=$SEED python3 -m training.train --params params.yaml > /dev/null 2>&1
cp metrics/train_metrics.json /tmp/repro_gate_run1.json

PYTHONHASHSEED=$SEED python3 -m training.train --params params.yaml > /dev/null 2>&1

if diff /tmp/repro_gate_run1.json metrics/train_metrics.json > /dev/null 2>&1; then
    _pass "Training is deterministic (two runs produce identical metrics)"
else
    _fail "Training is NOT deterministic:"
    diff /tmp/repro_gate_run1.json metrics/train_metrics.json || true
fi

# ⑥ Metrics match original run
_section "⑥ Metric Comparison (original vs reproduced)"
python3 - <<EOF
import json, mlflow, sys
mlflow.set_tracking_uri("$TRACKING_URI")
run = mlflow.get_run("$RUN_ID")

with open("metrics/train_metrics.json") as f:
    local = json.load(f)

ok = True
for key in ["roc_auc", "brier_score", "calibration_error"]:
    orig = run.data.metrics.get(key)
    repro = local.get(key)
    if orig is None:
        print(f"  SKIP: {key} not in original run (may not have been logged)")
        continue
    diff = abs(orig - repro)
    if diff < 1e-4:
        print(f"  {key}: {repro:.6f} == {orig:.6f} ✓")
    else:
        print(f"  MISMATCH: {key}: repro={repro:.6f} orig={orig:.6f} diff={diff:.6f}")
        ok = False

sys.exit(0 if ok else 1)
EOF
_pass "Metrics match original run (within tolerance)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  Passed: $PASS | Failed: $FAIL"
if [ "$FAIL" -eq 0 ]; then
    echo "  ✅ REPRODUCIBILITY GATE PASSED"
else
    echo "  ❌ REPRODUCIBILITY GATE FAILED ($FAIL checks)"
fi
echo "══════════════════════════════════════════════"
echo ""

[ "$FAIL" -eq 0 ]
