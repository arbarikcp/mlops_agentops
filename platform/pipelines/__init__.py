# pipelines package — orchestration, lineage, pipeline DAG definitions
"""
Phase 5 — Orchestration & Pipelines modules:

  dag.py               Core primitives: DagStep, SimpleDag, RunContext, BackfillPlanner
  dagster_pipeline.py  Dagster-style asset pipeline (PipelineConfig, TrainingPipeline)
  zenml_pipeline.py    ZenML-style pipeline (StepDef, ZenPipeline, ArtifactStore)
  validation_gate.py   Data validation gate (SchemaCheck, StatisticalCheck, DataValidationGate)
  model_gate.py        Model validation gate (ModelGate, ChampionRegistry, GateThresholds)
  failure_modes.py     Failure classification, idempotency proof, lineage audit
  pipeline_gate.py     Pipeline gate dry-run + orchestration survey
  lineage.py           OpenLineage emission (Phase 1, Day 13)
"""
