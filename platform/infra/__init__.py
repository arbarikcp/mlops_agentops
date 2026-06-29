"""infra package — Kubernetes, Helm, GitOps, and cloud infrastructure builders.

Phase 9 (Days 59–70):
  infra/k8s_manifests.py    — ResourceRequirements, ContainerSpec, DeploymentSpec, ServiceSpec, K8sManifestSet
  infra/ingress.py          — IngressRule, IngressSpec (NGINX kind cluster)
  infra/helm_chart.py       — HelmValues, HelmChart (values builder + CLI command renderer)
  infra/k8s_gpu_storage.py  — StorageStrategy, VolumeSpec, GPUToleration, GPUWorkloadSpec
  infra/kserve.py           — InferenceServiceSpec, CanaryConfig (traffic splitting)
  infra/k8s_autoscaling.py  — HPAMetric, HPASpec, KEDAScaledObject, KueueJobConfig
  infra/k8s_observability.py — PolicyRule, ClusterRoleSpec, ServiceMonitorSpec, SecretThreatChecker

Phase 11 (Days 74–77):
  infra/gitops.py              — AppHealthStatus, SyncPolicy, AppSyncResult, ArgoCDApp
  infra/progressive_delivery.py — CanaryStep, RolloutStrategy, AnalysisMetric, AnalysisTemplate, ArgoRollout
  infra/ct_automation.py       — TriggerType, CTTrigger, CTWorkflowStep, CTWorkflowSpec, CTRun

Phase 12 (Days 78–90) — AWS Deep + GCP Mapped + Milestone 2 Gate:
  infra/aws/__init__.py         — re-exports all AWS builders
  infra/aws/foundations.py      — IAMStatement, IAMPolicyDoc, ECRLifecycleRule, ECRRepository, SubnetConfig, VPCEndpoint, VPCConfig
  infra/aws/sagemaker_training.py — DataChannel, SMTrainingJob, ProcessingInput, ProcessingOutput, SMProcessingJob, SMTrialComponent, SMExperiment
  infra/aws/sagemaker_serving.py  — SMModelPackage, SMEndpointConfig (real-time/serverless/async/batch), SMEndpoint
  infra/aws/sagemaker_pipeline.py — SMPipelineStep, SMPipelineParameter, SMPipeline, SMModelApproval
  infra/aws/sagemaker_monitor.py  — MonitoringConstraints, SMDataQualityMonitor, SMModelQualityMonitor, SMClarifyBiasConfig, SMClarifyConfig
  infra/aws/serving.py            — EKSResourceSpec, EKSInferenceConfig, BedrockGuardrailConfig, BedrockConfig
  infra/aws/security.py           — SpotConfig, KMSKeyPolicy, KMSConfig, BudgetAlert, BudgetGuardrail, PrivateLinkConfig
  infra/terraform_config.py       — TFVariable, TFResource, TFOutput, TFModule, TFConfig
  infra/gcp_vertex.py             — VertexMachineSpec, VertexTrainingJob, VertexModelPackage, VertexEndpoint, VertexPipelineComponent, VertexPipeline
  infra/portability.py            — MatrixEntry, PortabilityMatrix, CloudAdapter, PortabilityScore
  infra/aws_deployment.py         — DeploymentStage, DeploymentReport, AWSDeploymentPlan
  infra/milestone2_gate.py        — M2GateCheck, M2GateReport, Milestone2Gate (15 checks, 6 gate dimensions)
"""
