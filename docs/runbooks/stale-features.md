# Runbook: stale-features

## Alert
`FeatureStaleness` fires when `feature_freshness_lag_s > 3600` for `10m`.

## Immediate Steps (first 5 minutes)
1. Acknowledge alert in Slack `#ml-oncall`
2. Check feature freshness dashboard: **Feature Store → Freshness**
3. Confirm lag: `feature_freshness_lag_s{feature_set="credit_features"}` > 3600
4. Note which entity set is stale

## Root Cause Investigation
- [ ] Check Dagster materialization status:
  ```bash
  dagster job status --location credit-risk --job materialize_credit_features
  ```
- [ ] Check failure reason in Dagster UI or logs:
  ```bash
  dagster run logs --run-id <latest_run_id>
  ```
- [ ] Check if Redis is healthy:
  ```bash
  redis-cli -h redis.mlops.svc ping
  ```
- [ ] Check for OOM in materializer pod:
  ```bash
  kubectl get events -n ml-training --field-selector reason=OOMKilling
  ```

## Recovery
1. Re-trigger materialization:
   ```bash
   dagster job launch \
       --location credit-risk \
       --job materialize_credit_features
   ```
2. Monitor progress in Dagster UI (ETA: 5–15 min)
3. Verify feature freshness in Redis:
   ```bash
   redis-cli hgetall credit_risk:features:entity_001 | grep -A1 timestamp
   ```
4. Confirm alert clears: `feature_freshness_lag_s < 60`

## Escalate if
- Re-materialization fails twice
- Redis OOM (requires infra team)
- Freshness lag > 6 hours (upstream data pipeline issue)

## Postmortem trigger
Any lag > 2 hours reaching production serving.
