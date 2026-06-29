# Day 60 — kind Cluster: Deploy Service + Ingress

## What is kind?

**kind** (Kubernetes IN Docker) runs a full K8s cluster inside Docker containers.
It is the standard local-dev K8s for ML teams — no VMs, no cloud bill, reproducible CI.

```
Host machine
└── Docker
    ├── kind-control-plane container  ← kube-apiserver, etcd, scheduler
    ├── kind-worker-1 container       ← runs ML serving pods
    └── kind-worker-2 container       ← runs ML serving pods
```

---

## Cluster Config

```yaml
# infra/kind/cluster.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: mlops-local
nodes:
  - role: control-plane
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "ingress-ready=true"
    extraPortMappings:
      - containerPort: 80
        hostPort: 8080      # http://localhost:8080 → nginx ingress
      - containerPort: 443
        hostPort: 8443
  - role: worker
    labels:
      node-type: ml-serving
  - role: worker
    labels:
      node-type: ml-serving
```

---

## Ingress

Ingress is a Layer-7 HTTP router that maps hostnames/paths to services:

```mermaid
graph LR
    Browser -->|http://localhost:8080/predict| Ingress[NGINX Ingress Controller]
    Ingress -->|/predict| SVC[credit-risk-api Service]
    Ingress -->|/metrics| SVC2[prometheus-metrics Service]
    SVC --> Pod1[API Pod]
    SVC --> Pod2[API Pod]
    SVC --> Pod3[API Pod]
```

```yaml
# infra/k8s/base/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-serving-ingress
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
    - host: localhost
      http:
        paths:
          - path: /predict
            pathType: Prefix
            backend:
              service:
                name: credit-risk-api
                port:
                  number: 80
          - path: /health
            pathType: Prefix
            backend:
              service:
                name: credit-risk-api
                port:
                  number: 80
```

---

## Deployment Sequence: git push → running in kind

```mermaid
sequenceDiagram
    participant Dev
    participant Make as Makefile
    participant Kind as kind CLI
    participant K8s as kubectl
    participant Cluster

    Dev->>Make: make kind-up
    Make->>Kind: kind create cluster --config infra/kind/cluster.yaml
    Kind-->>Make: cluster ready ✅
    Make->>K8s: kubectl apply -f infra/k8s/base/
    K8s->>Cluster: create Namespace, ConfigMap, Secret, Deployment, Service
    Cluster-->>K8s: resources created
    Make->>K8s: kubectl rollout status deployment/credit-risk-api -n ml-serving
    K8s-->>Make: deployment complete ✅
    Dev->>Make: make k8s-smoke
    Make->>Cluster: curl http://localhost:8080/health
    Cluster-->>Make: {"status": "ok"} ✅
```

---

## Key kind + kubectl Commands

```bash
# Create cluster
kind create cluster --name mlops-local --config infra/kind/cluster.yaml

# Load local Docker image into kind (avoids registry)
kind load docker-image credit-risk-api:v1 --name mlops-local

# Apply all base manifests
kubectl apply -f infra/k8s/base/ -n ml-serving

# Watch rollout
kubectl rollout status deployment/credit-risk-api -n ml-serving

# Port-forward for quick testing (no ingress needed)
kubectl port-forward svc/credit-risk-api 8080:80 -n ml-serving

# Rollback
kubectl rollout undo deployment/credit-risk-api -n ml-serving

# Delete cluster
kind delete cluster --name mlops-local
```

---

## IngressSpec Builder (Python)

```mermaid
classDiagram
    class IngressRule {
        +host: str
        +path: str
        +path_type: str
        +service_name: str
        +service_port: int
        +to_dict() dict
    }

    class IngressSpec {
        +name: str
        +namespace: str
        +rules: list~IngressRule~
        +ingress_class: str
        +annotations: dict
        +to_manifest() dict
    }

    IngressSpec --> "1..*" IngressRule
```
