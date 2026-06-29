# Runbook: broken-retriever

## Alert
`RetrievalEmpty` fires when `retrieval_empty_rate > 0.05` for `5m`.

## Immediate Steps (first 5 minutes)
1. Acknowledge alert in Slack `#llm-oncall`
2. Check dashboard: **RAG Pipeline → Retrieval Quality**
3. Confirm: `retrieval_context_size_avg{service="rag"}` ≈ 0
4. Note affected queries (check trace IDs in Jaeger / OTEL)

## Root Cause Investigation
- [ ] Check embedding sidecar:
  ```bash
  kubectl get pods -n llm-serving -l app=embedding-sidecar
  kubectl describe pod <embedding-pod> -n llm-serving
  ```
- [ ] Look for OOMKilled:
  ```bash
  kubectl get events -n llm-serving \
      --field-selector reason=OOMKilling \
      --sort-by='.lastTimestamp' | tail -5
  ```
- [ ] Test embedding endpoint directly:
  ```bash
  kubectl exec -n llm-serving deploy/embedding-sidecar -- \
      curl -s -X POST http://localhost:8001/embed \
      -H 'Content-Type: application/json' \
      -d '{"text": "test query"}'
  ```
- [ ] Check vector store (Chroma / Pinecone) connectivity:
  ```bash
  curl http://vector-store.llm-serving.svc/health
  ```

## Recovery
1. If OOMKilled — increase memory limit:
   ```bash
   kubectl set resources deployment/embedding-sidecar \
       --containers=embedding \
       --limits=memory=4Gi \
       --requests=memory=2Gi \
       -n llm-serving
   ```
2. Wait for pod restart:
   ```bash
   kubectl rollout status deployment/embedding-sidecar -n llm-serving
   ```
3. Re-test retrieval:
   ```bash
   curl http://rag-service.llm-serving.svc/retrieve \
       -d '{"query": "late payment penalty"}' | jq '.chunks | length'
   # Should return > 0
   ```
4. Confirm alert clears: `retrieval_empty_rate < 0.01`

## Escalate if
- OOM repeats within 30 min of memory increase
- Vector store unreachable (DB/network team)
- Embedding model corrupted (ML team to re-deploy)

## Postmortem trigger
Any > 5 min of 100% retrieval failure reaching users.
