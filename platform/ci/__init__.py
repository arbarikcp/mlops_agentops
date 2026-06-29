"""ci package — ML CI/CD pipeline stages for code, data, and model quality.

Phase 8 (Days 54–58):
  ci/ml_pipeline.py     — CIStage, CIResult, CIPipelineRun, MLCIPipeline
  ci/ml_tests.py        — DataContractChecker, BehavioralChecker, SmokeTrainer, AUCGuard
  ci/gitlab_pipeline.py — CacheConfig, ArtifactConfig, GitLabJob, GitLabPipeline
  ci/signing.py         — SBOMEntry, SBOMDocument, ArtifactProvenanceRecord, ArtifactSigner
  ci/milestone1_gate.py — TraceabilityRecord, GateCheck, GateReport, Milestone1Gate
"""
