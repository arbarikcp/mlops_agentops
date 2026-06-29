# Runbook: bad-artifact-pushed

## Alert
`ModelPredictionPSI` fires when `model_prediction_psi_score > 0.2` for `5m`.

## Immediate Steps (first 5 minutes)
1. Acknowledge alert in PagerDuty / Slack channel `#ml-oncall`
2. Open Grafana dashboard: **ML Serving → Prediction Distribution**
3. Confirm PSI is elevated: `model_prediction_psi_score{alias="production"}` > 0.2
4. Note which model version is in production: check `model_version` label on metrics

## Root Cause Investigation
- [ ] Check recent registry promotions:
  ```bash
  mlflow models search-model-versions \
      --filter "name='credit-risk' AND tags.promoted_at > '1h ago'"
  ```
- [ ] Check AUC of current production version vs previous:
  ```bash
  mlflow runs get --run-id <current_run_id> | jq '.data.metrics.val_auc'
  ```
- [ ] Compare with CI gate logs: was AUC guard bypassed?

## Recovery
1. Find last good version (gate_passed=true):
   ```bash
   mlflow models search-model-versions \
       --filter "name='credit-risk' AND tags.gate_passed='true'" \
       | jq 'sort_by(.version)[-1].version'
   ```
2. Roll back production alias:
   ```bash
   mlflow models set-alias \
       --name credit-risk --alias production --version <LAST_GOOD>
   ```
3. Verify KServe downloads previous version (allow 60s):
   ```bash
   kubectl get inferenceservice credit-risk -n ml-serving \
       -o jsonpath='{.status.modelStatus.modelRevisionStates}'
   ```
4. Confirm PSI drops below 0.1:
   - Grafana: `model_prediction_psi_score{alias="production"} < 0.1`

## Escalate if
- PSI stays elevated > 30 min after rollback
- Multiple models affected simultaneously
- Data loss or audit trail gap suspected

## Postmortem trigger
Any SLO breach, or if detection time > 15 min from promotion.
