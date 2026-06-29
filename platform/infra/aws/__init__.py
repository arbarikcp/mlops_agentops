"""infra.aws — AWS-specific MLOps infrastructure builders (Phase 12, Days 79–85).

All builders return plain Python dicts (no AWS SDK calls) so they are testable
without credentials and deployable via IaC tooling.

Modules:
  foundations.py        — IAMPolicyDoc, ECRLifecycleRule, VPCConfig
  sagemaker_training.py — SMTrainingJob, SMProcessingJob, SMExperiment
  sagemaker_serving.py  — SMModelPackage, SMEndpointConfig, SMEndpoint
  sagemaker_pipeline.py — SMPipelineStep, SMPipeline, SMModelApproval
  sagemaker_monitor.py  — SMDataQualityMonitor, SMModelQualityMonitor, SMClarifyConfig
  serving.py            — EKSInferenceConfig, BedrockConfig
  security.py           — SpotConfig, KMSConfig, BudgetGuardrail, PrivateLinkConfig
"""

from .foundations import IAMPolicyDoc, ECRLifecycleRule, VPCConfig
from .sagemaker_training import SMTrainingJob, SMProcessingJob, SMExperiment
from .sagemaker_serving import SMModelPackage, SMEndpointConfig, SMEndpoint
from .sagemaker_pipeline import SMPipelineStep, SMPipeline, SMModelApproval
from .sagemaker_monitor import SMDataQualityMonitor, SMModelQualityMonitor, SMClarifyConfig
from .serving import EKSInferenceConfig, BedrockConfig
from .security import SpotConfig, KMSConfig, BudgetGuardrail, PrivateLinkConfig

__all__ = [
    "IAMPolicyDoc", "ECRLifecycleRule", "VPCConfig",
    "SMTrainingJob", "SMProcessingJob", "SMExperiment",
    "SMModelPackage", "SMEndpointConfig", "SMEndpoint",
    "SMPipelineStep", "SMPipeline", "SMModelApproval",
    "SMDataQualityMonitor", "SMModelQualityMonitor", "SMClarifyConfig",
    "EKSInferenceConfig", "BedrockConfig",
    "SpotConfig", "KMSConfig", "BudgetGuardrail", "PrivateLinkConfig",
]
