# features package — feature store, views, materialization, streaming, monitoring, feedback loop
"""
Phase 6 — Feature Store & Closed Feedback Loop modules:

  feature_store.py    Core primitives: DataSource, OfflineStore, InMemoryOnlineStore,
                      FeatureRegistry, FeatureStore
  feature_views.py    Entity, Feature, FeatureView, FeatureService, PointInTimeJoin
                      + canonical credit-risk definitions
  materialization.py  MaterializationInterval, MaterializationJob, IncrementalMaterializer
  streaming.py        PushEvent, PushSchema, PushSource, OnDemandTransform, StreamProcessor
  feature_monitor.py  FreshnessChecker, FeatureQualityChecker, FeatureDriftMonitor,
                      FeatureMonitor, FeatureMonitorReport
  feedback_loop.py    GroundTruthJoiner, MetricRecomputer, RetrainDecider,
                      LabelFeedbackLoop — 8-step closed feedback loop
  skew_checker.py     TrainServeSkewChecker, TrainServeSkewReport — zero-skew consolidation
"""
