"""MLflow Model Registry operations — Day 11 deliverable.

Model Registry lifecycle:
  run_id (training run)
    → register_model()     → creates ModelVersion (version number assigned)
    → set_alias("champion") → marks this version as current production
    → get_model_by_alias()  → serving layer loads model by alias, not version number

Why aliases instead of stages:
  MLflow 2.x deprecated stages (Staging/Production/Archived) in favour of
  user-defined aliases. Aliases are more flexible:
    - "champion"    → currently serving 100% of traffic
    - "challenger"  → candidate being shadow-tested
    - "shadow"      → silent evaluation, no live traffic

Rollback:
  If the champion degrades (drift detected), set "champion" alias to the previous
  version in one API call — the serving layer picks it up without redeployment.

Threat checkpoint (Day 11):
  Artifact provenance — see threat_model_v0.md T-03, S-03:
  - ModelVersion.run_id traces back to the training run.
  - Training run has git commit tag, data version (DVC hash), params.
  - Together these form a complete provenance chain.

Usage:
    from training.registry import register_model, set_alias, compare_champion_challenger

    # After a training run:
    mv = register_model(run_id="abc123", model_name="credit-risk-model")
    set_alias("credit-risk-model", mv.version, "challenger")

    # After shadow/canary validation:
    result = compare_champion_challenger("credit-risk-model")
    if result.winner == "challenger":
        set_alias("credit-risk-model", result.challenger_version, "champion")

Debugging:
    - "RESOURCE_DOES_NOT_EXIST": Model not registered yet — run register_model first.
    - "INVALID_PARAMETER_VALUE": Alias contains spaces or special chars → use snake_case.
    - List all versions:
        python -m training.registry --list credit-risk-model
    - Get champion run_id:
        python -m training.registry --get-run-id credit-risk-model champion
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

log = logging.getLogger(__name__)


def _client() -> MlflowClient:
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    return MlflowClient(tracking_uri=uri)


# ── Registration ─────────────────────────────────────────────────────────────

def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
    description: str | None = None,
) -> ModelVersion:
    """Register a model from an MLflow run into the Model Registry.

    The model_uri pattern "runs:/<run_id>/<artifact_path>" points to the
    artifact logged during training. MLflow copies it to the registry store.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    log.info("Registering %s from %s...", model_name, model_uri)

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    if description:
        client = _client()
        client.update_model_version(
            name=model_name, version=mv.version, description=description
        )

    log.info("Registered %s v%s (run_id=%s)", model_name, mv.version, run_id)
    return mv


# ── Alias management ─────────────────────────────────────────────────────────

def set_alias(model_name: str, version: str | int, alias: str) -> None:
    """Assign an alias (champion / challenger / shadow) to a model version.

    The serving layer always loads by alias — changing the alias here
    transparently redirects traffic without redeployment.
    """
    _client().set_registered_model_alias(
        name=model_name, alias=alias, version=str(version)
    )
    log.info("Alias '%s' → %s v%s", alias, model_name, version)


def remove_alias(model_name: str, alias: str) -> None:
    _client().delete_registered_model_alias(name=model_name, alias=alias)
    log.info("Removed alias '%s' from %s", alias, model_name)


def get_version_by_alias(model_name: str, alias: str) -> ModelVersion:
    return _client().get_model_version_by_alias(name=model_name, alias=alias)


def get_run_id_by_alias(model_name: str, alias: str) -> str:
    mv = get_version_by_alias(model_name, alias)
    return mv.run_id


# ── Model loading ────────────────────────────────────────────────────────────

def load_model_by_alias(model_name: str, alias: str):
    """Load a registered model by alias. Works with any MLflow-flavoured model."""
    uri = f"models:/{model_name}@{alias}"
    log.info("Loading %s@%s", model_name, alias)
    return mlflow.pyfunc.load_model(uri)


# ── Champion / challenger comparison ─────────────────────────────────────────

@dataclass
class ComparisonResult:
    champion_version: str
    challenger_version: str
    champion_auc: float
    challenger_auc: float
    winner: str  # "champion" or "challenger"
    delta: float  # challenger_auc - champion_auc


def compare_champion_challenger(
    model_name: str,
    metric: str = "roc_auc",
) -> ComparisonResult:
    """Compare champion vs challenger on a logged metric.

    Returns ComparisonResult.winner = "challenger" if the challenger is
    strictly better. Ties go to "champion" (conservative policy for credit risk).
    """
    client = _client()

    def _get(alias: str) -> tuple[str, float]:
        mv = client.get_model_version_by_alias(name=model_name, alias=alias)
        run = client.get_run(mv.run_id)
        val = run.data.metrics.get(metric, 0.0)
        log.info("%s: %s v%s → %s=%.4f", alias, model_name, mv.version, metric, val)
        return mv.version, val

    champ_ver, champ_auc = _get("champion")
    chal_ver, chal_auc = _get("challenger")

    delta = chal_auc - champ_auc
    winner = "challenger" if delta > 0 else "champion"

    log.info(
        "Champion AUC=%.4f vs Challenger AUC=%.4f → Δ=%.4f → winner: %s",
        champ_auc, chal_auc, delta, winner,
    )
    return ComparisonResult(
        champion_version=champ_ver,
        challenger_version=chal_ver,
        champion_auc=champ_auc,
        challenger_auc=chal_auc,
        winner=winner,
        delta=delta,
    )


def list_versions(model_name: str) -> None:
    """Print all registered versions with their aliases and run IDs."""
    client = _client()
    versions = client.search_model_versions(f"name='{model_name}'")
    print(f"\n{'Version':<10} {'Aliases':<30} {'Run ID':<15} {'State'}")
    print("-" * 70)
    for mv in sorted(versions, key=lambda x: int(x.version)):
        aliases = ",".join(mv.aliases) if mv.aliases else "-"
        print(f"  v{mv.version:<8} {aliases:<30} {mv.run_id[:8]:<15} {mv.current_stage}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="MLflow Registry CLI")
    parser.add_argument("--list", metavar="MODEL_NAME", help="List all versions")
    parser.add_argument("--get-run-id", nargs=2, metavar=("MODEL_NAME", "ALIAS"),
                        help="Get run_id for a model alias")
    parser.add_argument("--compare", metavar="MODEL_NAME", help="Compare champion vs challenger")
    args = parser.parse_args()

    if args.list:
        list_versions(args.list)
    elif args.get_run_id:
        model_name, alias = args.get_run_id
        print(get_run_id_by_alias(model_name, alias))
    elif args.compare:
        compare_champion_challenger(args.compare)
