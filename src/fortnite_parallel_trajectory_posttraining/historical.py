from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from .common import (
    ANALYSIS_DIRECTORY,
    CHILD_RUN,
    EXPECTED_ENCODER_STATE_SHA256,
    EXPECTED_FINAL_ROUTE_NLL,
    EXPECTED_FINAL_STEP,
    FINALIZATION_DIRECTORY,
    REPOSITORY_ROOT,
    current_utc,
    file_record,
    make_file_read_only,
    read_json,
    require,
    self_seal,
    verify_self_seal,
    write_new_json,
)


OUTPUT_PATH = ANALYSIS_DIRECTORY / "historical_comparability.json"
ROTATION_RUN = REPOSITORY_ROOT / "runs" / "rotation-placeholder"
OLD_DECODER_RUN = (
    REPOSITORY_ROOT / "runs" / "parallel-trajectory-decoder-v1-20260813-run-1-resume-1"
)
OLD_DECODER_EVALUATION = (
    REPOSITORY_ROOT
    / "runs"
    / "parallel-trajectory-decoder-v1-20260813-run-1-evaluation-20260816"
)
FNCS_ENCODER_RUN = REPOSITORY_ROOT / "runs" / "encoder-fncs-pilot-20260817-fresh-run-3"


def build_historical_comparability() -> dict[str, Any]:
    if OUTPUT_PATH.is_file():
        existing = read_json(OUTPUT_PATH)
        verify_self_seal(existing)
        return {"status": "already_complete", "artifact": file_record(OUTPUT_PATH)}

    rotation_config = read_json(ROTATION_RUN / "resolved_config.json")
    rotation_summary = read_json(ROTATION_RUN / "final_summary.json")
    rotation_best = read_json(ROTATION_RUN / "best.json")
    old_resolved = read_json(OLD_DECODER_RUN / "resolved_config.json")
    old_best = read_json(OLD_DECODER_RUN / "best.json")
    old_eval = read_json(OLD_DECODER_EVALUATION / "final_evaluation_summary.json")
    encoder_resolved = read_json(FNCS_ENCODER_RUN / "resolved_training_config.json")
    encoder_objective = read_json(FNCS_ENCODER_RUN / "objective_definition.json")
    encoder_best = read_json(FNCS_ENCODER_RUN / "best.json")
    current_lineage = read_json(FINALIZATION_DIRECTORY / "completed_lineage.json")
    verify_self_seal(current_lineage)

    require(
        rotation_summary.get("status") == "completed"
        and rotation_best.get("value") == 17.46510212045403,
        "rotation-placeholder historical value changed",
    )
    old_training = old_resolved.get("training_configuration", {})
    require(
        old_best.get("value") == 11.876513356512243
        and old_best.get("optimizer_step") == 860
        and old_training.get("architecture_id") == "parallel_trajectory_decoder_v1"
        and old_training.get("target_contract_version") == "parallel-trajectory-targets:1.0",
        "old parallel-decoder historical binding changed",
    )
    old_reproduced_nll = old_eval.get("probabilistic_calibration", {}).get(
        "validation", {}
    ).get("route_level_mixture_nll")
    require(
        isinstance(old_reproduced_nll, (int, float))
        and math.isclose(old_reproduced_nll, old_best["value"], rel_tol=0.0, abs_tol=3e-8),
        "old decoder validation NLL did not reproduce",
    )
    require(
        encoder_best.get("value") == 7.842962951838928
        and encoder_objective.get("reductions", {}).get("future_position")
        == "independent masked mean cross entropy for each 15/30/60-second horizon, then sum the three horizon means",
        "FNCS encoder historical objective changed",
    )
    current_state = current_lineage.get("completed_state", {})
    require(
        current_state.get("best_checkpoint_step") == EXPECTED_FINAL_STEP
        and current_state.get("best_validation_route_nll") == EXPECTED_FINAL_ROUTE_NLL,
        "current final historical binding changed",
    )

    experiments = {
        "rotation_placeholder_41_session_corpus": {
            "artifact": file_record(ROTATION_RUN / "final_summary.json"),
            "model_family": "RotationModel encoder plus four classification heads",
            "trajectory_decoder_present": False,
            "training_sessions": rotation_config["train_session_count"],
            "validation_sessions": rotation_config["validation_session_count"],
            "test_sessions": rotation_config["test_session_count"],
            "dataset": rotation_config["dataset_root"],
            "horizons_seconds": [15, 30, 60],
            "position_output": "1025-way grid/sentinel classification independently at each horizon",
            "reported_value": 17.46510212045403,
            "reported_metric": "future_position",
            "objective_and_reduction": (
                "Sum of three independently masked mean categorical cross-entropies at 15/30/60 s."
            ),
            "optimizer_steps": rotation_summary["optimizer_steps"],
            "encoder_policy": "encoder trained jointly from a fresh/random run state",
            "use_in_current_report": "historical context only",
        },
        "parallel_trajectory_decoder_20260813": {
            "artifact": file_record(OLD_DECODER_RUN / "best.pt"),
            "model_family": old_training["architecture_id"],
            "trajectory_decoder_present": True,
            "training_sessions": old_training["expected_split_counts"]["train"],
            "validation_sessions": old_training["expected_split_counts"]["validation"],
            "test_sessions": old_training["expected_split_counts"]["test"],
            "dataset": old_training["dataset_root"],
            "horizons_seconds": list(range(5, 61, 5)),
            "target_contract": old_training["target_contract_version"],
            "reported_value": old_best["value"],
            "reproduced_value": old_reproduced_nll,
            "reported_metric": "validation_route_mixture_nll",
            "objective_and_reduction": (
                "Same five-mode route-level bivariate-Gaussian NLL family: valid horizon "
                "log densities sum within a coherent mode, then one mode logsumexp and route mean."
            ),
            "optimizer_steps": old_best["optimizer_step"],
            "encoder_policy": (
                "Transferred encoder/congestion state frozen only in epoch 1; all transferred "
                "components became trainable in epochs 2-20."
            ),
            "encoder_identity": old_resolved["transfer_proof"]["component_state_sha256"]["encoder"],
            "validation_point_metrics_meters": old_eval["point_forecast_accuracy"]["validation_primary"],
            "old_validation_static_meters": old_eval["validation_performance"]["performance"]["static"],
            "use_in_current_report": "same-objective historical context, not a controlled comparison",
        },
        "fncs_encoder_20260817": {
            "artifact": file_record(FNCS_ENCODER_RUN / "best.pt"),
            "model_family": encoder_objective["model"],
            "trajectory_decoder_present": False,
            "training_sessions": 920,
            "validation_sessions": 115,
            "test_sessions": 115,
            "dataset": encoder_resolved["dataset_root"],
            "horizons_seconds": encoder_objective["class_contract"]["horizons_seconds"],
            "position_output": "1025-way grid/sentinel classification independently at each horizon",
            "reported_value": encoder_best["value"],
            "reported_metric": encoder_best["metric"],
            "objective_and_reduction": encoder_objective["reductions"]["future_position"],
            "optimizer_steps": encoder_best["optimizer_step"],
            "encoder_state_selected_for_current_decoder": EXPECTED_ENCODER_STATE_SHA256,
            "use_in_current_report": "encoder-pretraining context only",
        },
        "fncs_frozen_parallel_decoder_final": {
            "artifact": file_record(CHILD_RUN / "best.pt"),
            "model_family": "parallel_trajectory_decoder_v1",
            "trajectory_decoder_present": True,
            "training_sessions": 920,
            "validation_sessions": 115,
            "test_sessions": 115,
            "dataset": encoder_resolved["dataset_root"],
            "horizons_seconds": list(range(5, 61, 5)),
            "target_contract": "parallel-trajectory-targets:1.0",
            "reported_value": EXPECTED_FINAL_ROUTE_NLL,
            "reported_metric": "validation_route_mixture_nll",
            "objective_and_reduction": (
                "Five-mode route-level bivariate-Gaussian NLL, summed over valid horizons "
                "within route mode and averaged equally over valid routes."
            ),
            "optimizer_steps": EXPECTED_FINAL_STEP,
            "encoder_policy": "canonical FNCS encoder frozen byte-identically for every update",
            "encoder_identity": EXPECTED_ENCODER_STATE_SHA256,
            "use_in_current_report": "primary finalized development result",
        },
    }
    matrix = [
        {
            "comparison": "17.46510212045403 versus 7.004335993010065",
            "verdict": "not numerically comparable",
            "reasons": [
                "categorical classification cross-entropy versus continuous route-mixture density NLL",
                "three independent horizons versus one coherent twelve-horizon route",
                "different output spaces, likelihood measures, masks, populations, and model families",
                "a smaller raw loss number cannot establish improvement across these objectives",
            ],
        },
        {
            "comparison": "17.46510212045403 versus 7.842962951838928",
            "verdict": "same broad future-position CE definition, contextual only",
            "reasons": [
                "both sum independently masked 15/30/60 s categorical position cross-entropies",
                "dataset generation, tensorization/lifecycle handling, train/validation populations, and compute differ",
                "the difference cannot be attributed to session count or architecture in isolation",
            ],
        },
        {
            "comparison": "11.876513356512243 versus 7.004335993010065",
            "verdict": "same route-NLL family and units, not a controlled performance comparison",
            "reasons": [
                "architecture, target contract, horizons, grid normalization, and reduction are aligned",
                "validation populations are 11 versus 115 different sessions",
                "training populations are 86 versus 920 different sessions and data vintages",
                "the old encoder/congestion stack was fine-tuned after epoch 1; the current encoder stayed frozen",
                "optimizer schedules and update counts are 860 versus 9200",
            ],
        },
        {
            "comparison": "current fixed-compute N=41/115/230/460/920 endpoints",
            "verdict": "controlled within the locked scale study",
            "reasons": [
                "nested subsets of one 920-session training universe",
                "same initialization, frozen encoder, architecture, targets, objective, sampler, absolute LR history, and 1840 updates",
                "same locked 115-session validation queries",
                "remaining uncertainty includes one initialization and one subset ordering",
            ],
        },
    ]
    payload = self_seal(
        {
            "schema_version": "fncs-decoder-historical-comparability:1.0",
            "created_utc": current_utc(),
            "read_only_artifact_review_only": True,
            "no_dataset_session_opened": True,
            "no_test_partition_accessed": True,
            "experiments": experiments,
            "comparability_matrix": matrix,
            "bottom_line": (
                "Only values sharing the route-mixture likelihood definition have the same mathematical "
                "units; only the newly locked fixed-compute nested-subset endpoints support a controlled "
                "dataset-size comparison."
            ),
            "source_artifacts": [
                file_record(ROTATION_RUN / "resolved_config.json"),
                file_record(ROTATION_RUN / "final_summary.json"),
                file_record(OLD_DECODER_RUN / "resolved_config.json"),
                file_record(OLD_DECODER_RUN / "best.json"),
                file_record(OLD_DECODER_EVALUATION / "final_evaluation_summary.json"),
                file_record(FNCS_ENCODER_RUN / "resolved_training_config.json"),
                file_record(FNCS_ENCODER_RUN / "objective_definition.json"),
                file_record(FNCS_ENCODER_RUN / "best.json"),
                file_record(FINALIZATION_DIRECTORY / "completed_lineage.json"),
            ],
        }
    )
    write_new_json(OUTPUT_PATH, payload)
    make_file_read_only(OUTPUT_PATH)
    return {"status": "completed", "artifact": file_record(OUTPUT_PATH)}


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    print(json.dumps(build_historical_comparability(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["OUTPUT_PATH", "build_historical_comparability"]
