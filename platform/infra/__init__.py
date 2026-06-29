"""infra package — Kubernetes, Helm, and cloud infrastructure builders.

Phase 9 (Days 59–70):
  infra/k8s_manifests.py    — ResourceRequirements, ContainerSpec, DeploymentSpec, ServiceSpec, K8sManifestSet
  infra/ingress.py          — IngressRule, IngressSpec (NGINX kind cluster)
  infra/helm_chart.py       — HelmValues, HelmChart (values builder + CLI command renderer)
  infra/k8s_gpu_storage.py  — StorageStrategy, VolumeSpec, GPUToleration, GPUWorkloadSpec
  infra/kserve.py           — InferenceServiceSpec, CanaryConfig (traffic splitting)
  infra/k8s_autoscaling.py  — HPAMetric, HPASpec, KEDAScaledObject, KueueJobConfig
  infra/k8s_observability.py — PolicyRule, ClusterRoleSpec, ServiceMonitorSpec, SecretThreatChecker
"""
