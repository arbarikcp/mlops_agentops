# Day 62 — Storage on K8s: PVCs, Model-Storage Strategies, Init-Container Pulls

## The Storage Drag Problem

ML models are large binary files (100 MB – 10 GB). Every pod restart that must
re-download the model adds latency to scaling events:

```
Without storage strategy:
  Pod starts → pulls 2 GB model from S3 → serving ready in 3 min

With init-container + PVC:
  Pod starts → reads model from PVC (already present) → serving ready in 15s

With node-local cache (PVC per node):
  First pod on node: downloads 2 GB once
  Subsequent pods: reads from node cache → serving ready in 15s
```

---

## Kubernetes Storage Primitives

```mermaid
graph TD
    PVC[PersistentVolumeClaim] -->|requests| PV[PersistentVolume]
    PV -->|backed by| SC[StorageClass]
    SC -->|provisions| EBS[AWS EBS]
    SC -->|provisions| NFS[NFS / EFS]
    SC -->|provisions| Local[Local disk]
    SC -->|provisions| MinIO[MinIO / S3]
    Pod -->|mounts| PVC
```

### StorageClass

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ml-model-storage
provisioner: kubernetes.io/no-provisioner   # local for kind
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Retain   # don't delete PV when PVC deleted
```

### PVC for model storage

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: ml-serving
spec:
  accessModes: [ReadOnlyMany]   # multiple pods can read same PV
  resources:
    requests:
      storage: 5Gi
  storageClassName: ml-model-storage
```

---

## Three Model-Storage Strategies

| Strategy | How | Cold-start | Pod lifecycle | Best for |
|---|---|---|---|---|
| **emptyDir** | Download in init-container per pod | Slow (per pod) | Lost on pod death | Dev / small models |
| **PVC (ReadOnlyMany)** | Download once → shared PVC | Fast (one-time) | Survives restarts | CPU serving, medium models |
| **Node-local PV** | First pod downloads, rest read cache | Fast after first | Node-local | GPU serving (DaemonSet) |

---

## Init-Container Pull Pattern

```mermaid
sequenceDiagram
    participant K8s
    participant Init as init-container (model-downloader)
    participant Vol as emptyDir volume
    participant API as main container (api)

    K8s->>Init: start init-container
    Init->>S3: aws s3 cp s3://models/v1.pkl /model/
    S3-->>Init: download complete
    Init->>Vol: write model.pkl
    Init-->>K8s: exit 0
    K8s->>API: start main container
    API->>Vol: read /model/model.pkl
    API-->>K8s: readinessProbe passes ✅
```

---

## PVC Strategy YAML

```yaml
# Shared model PVC — download once, mounted ReadOnly by all pods
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
spec:
  accessModes: [ReadOnlyMany]
  resources:
    requests:
      storage: 5Gi
---
# Job that downloads model once into PVC
apiVersion: batch/v1
kind: Job
metadata:
  name: model-seeder
spec:
  template:
    spec:
      containers:
        - name: seeder
          image: amazon/aws-cli:2.15.0
          command: [aws, s3, cp, "$(MODEL_S3_PATH)", /model/model.pkl]
          volumeMounts:
            - name: model-pvc
              mountPath: /model
      volumes:
        - name: model-pvc
          persistentVolumeClaim:
            claimName: model-cache
      restartPolicy: OnFailure
```

---

## StorageStrategy Enum and VolumeSpec (Python)

```mermaid
classDiagram
    class StorageStrategy {
        <<enumeration>>
        EMPTY_DIR
        PVC
        NODE_LOCAL
    }

    class VolumeSpec {
        +name: str
        +strategy: StorageStrategy
        +mount_path: str
        +pvc_name: str
        +storage_size: str
        +access_modes: list
        +to_volume_dict() dict
        +to_volume_mount_dict() dict
        +to_pvc_manifest() dict
    }

    VolumeSpec --> StorageStrategy
```
