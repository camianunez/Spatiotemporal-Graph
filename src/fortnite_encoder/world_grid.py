from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


WORLD_GRID_PROFILE_SCHEMA_VERSION = "2.0"
WORLD_GRID_AUDIT_SCHEMA_VERSION = "2.0"
WORLD_GRID_PROFILE_ID = "model-world-grid-v1"
WORLD_GRID_PROFILE_KIND = "engineered_model_world_partition"
WORLD_GRID_AUDIT_ID = "model-world-grid-v1-publication-audit"
WORLD_IDENTITY = "/Hera_Map/Maps/Hera_V2_Terrain"
WORLD_X_MIN = -131_072
WORLD_X_MAX = 131_072
WORLD_Y_MIN = -131_072
WORLD_Y_MAX = 131_072
GRID_ROWS = 32
GRID_COLUMNS = 32
LATTICE_ORIGIN = 0
LATTICE_SPACING_WORLD_UNITS = 512
CELL_WIDTH_WORLD_UNITS = 8_192
CELL_HEIGHT_WORLD_UNITS = 8_192
METERS_PER_WORLD_UNIT = 0.01
COMPATIBLE_BUILDS = ("41.10", "41.20")
EXPECTED_CHANGELISTS = {"41.10": 55_434_016, "41.20": 55_798_846}

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_EXPECTED_EVIDENCE_SUBJECTS = frozenset(
    (
        "model_world_partition",
        "reported_metadata_containment",
        "model_visible_coordinate_coverage",
    )
)
_EXPECTED_COORDINATE_SOURCES = frozenset(
    (
        "player_position",
        "player_event_position",
        "eligible_player_centroid",
        "zone_phase_source",
        "zone_phase_target",
        "zone_sample_current",
        "zone_sample_target",
    )
)
_PROFILE_FIELDS = {
    "schema_version",
    "profile_id",
    "profile_kind",
    "authoritative_terrain_bounds",
    "compatible_builds",
    "world_identity",
    "world_x_min",
    "world_x_max",
    "world_y_min",
    "world_y_max",
    "grid_rows",
    "grid_columns",
    "column_axis",
    "row_axis",
    "column_direction",
    "row_direction",
    "inclusive_maximum_bounds",
    "cell_ordering",
    "lattice_origin_x",
    "lattice_origin_y",
    "lattice_spacing_world_units",
    "cell_width_world_units",
    "cell_height_world_units",
    "world_unit_scale",
    "publication_audit",
    "evidence",
    "profile_hash",
}
_AUDIT_FIELDS = {
    "schema_version",
    "audit_id",
    "decoder_fork_revision",
    "inspected_replay_count",
    "replay_counts_by_build",
    "required_source_fields",
    "replays",
    "unique_values_by_build",
    "coordinate_validation_by_build",
    "lattice_canonicalization",
    "engineered_partition",
    "coordinate_audit",
    "engineered_validation",
    "compatibility_decision",
    "publication",
    "evidence",
    "audit_hash",
}
_DATASET_BINDING_FIELDS = {
    "ingestion_report_schema_version",
    "dataset_schema_version",
    "accepted_session_count",
    "accepted_session_counts_by_build",
    "accepted_sessions",
    "complete",
    "ingestion_binding_sha256",
}
_SESSION_BINDING_FIELDS = {
    "session_id",
    "replay_sha256",
    "build",
    "changelist",
    "disposition",
}


class WorldGridProfileError(ValueError):
    """Raised when a world-grid profile or publication binding is invalid."""


class _RawJsonFloat(float):
    """A parsed float that retains System.Text.Json's source token spelling."""

    def __new__(cls, token: str):
        value = super().__new__(cls, token)
        value.token = token
        return value


def _canonical_json(value: Any) -> str:
    def string(item: str) -> str:
        encoded = json.dumps(item, ensure_ascii=False)
        return (
            encoded.replace('\\"', "\\u0022")
            .replace("&", "\\u0026")
            .replace("<", "\\u003C")
            .replace(">", "\\u003E")
        )

    def write(item: Any) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if isinstance(item, str):
            return string(item)
        if isinstance(item, _RawJsonFloat):
            return item.token
        if type(item) is int:
            return str(item)
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("canonical JSON forbids nonfinite numbers")
            encoded = json.dumps(item, allow_nan=False)
            return re.sub(r"(?<=\d)e(?=[+-]\d)", "E", encoded)
        if isinstance(item, (list, tuple)):
            return "[" + ",".join(write(entry) for entry in item) + "]"
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("canonical JSON object keys must be strings")
            return "{" + ",".join(
                f"{string(key)}:{write(item[key])}"
                for key in sorted(item)
            ) + "}"
        raise TypeError(f"unsupported canonical JSON value {type(item)!r}")

    return write(value)


def _payload_hash(value: dict[str, Any], hash_field: str) -> str:
    payload = dict(value)
    payload.pop(hash_field, None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_object(
    raw: Any,
    *,
    location: str,
    fields: set[str] | frozenset[str],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise WorldGridProfileError(f"{location} must be a JSON object")
    unknown = sorted(set(raw) - fields)
    missing = sorted(fields - set(raw))
    if unknown:
        raise WorldGridProfileError(
            f"{location} contains unknown fields: {', '.join(unknown)}"
        )
    if missing:
        raise WorldGridProfileError(
            f"{location} is missing fields: {', '.join(missing)}"
        )
    return raw


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorldGridProfileError(f"{location} must be a nonempty string")
    return value


def _exact_integer(value: Any, location: str) -> int:
    if type(value) is not int:
        raise WorldGridProfileError(f"{location} must be an integer")
    return value


def _finite(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorldGridProfileError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise WorldGridProfileError(f"{location} must be finite")
    return result


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise WorldGridProfileError(f"{location} must be lowercase SHA-256")
    return value


def _read_json(
    path: str | Path,
    location: str,
    *,
    preserve_float_tokens: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            parse_float=_RawJsonFloat if preserve_float_tokens else float,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise WorldGridProfileError(f"cannot read {location} {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise WorldGridProfileError(f"{location} must be a JSON object")
    return raw


def ingestion_binding_from_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise WorldGridProfileError("ingestion report must be a JSON object")
    if report.get("schema_version") != "2.0":
        raise WorldGridProfileError("ingestion report schema_version must be 2.0")
    if report.get("dataset_schema_version") != "2.0.0":
        raise WorldGridProfileError("ingestion dataset schema_version must be 2.0.0")
    items = report.get("items")
    if not isinstance(items, list):
        raise WorldGridProfileError("ingestion report items must be an array")

    sessions: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        disposition = item.get("disposition")
        if disposition not in {"included_attested", "included_unattested"}:
            continue
        session_id = _string(
            item.get("session_id"), f"ingestion.items[{index}].session_id"
        )
        replay_sha256 = _sha256(
            item.get("replay_sha256"),
            f"ingestion.items[{index}].replay_sha256",
        )
        build_profile = item.get("build_profile")
        if not isinstance(build_profile, dict):
            raise WorldGridProfileError(
                f"ingestion.items[{index}].build_profile must be an object"
            )
        build = _string(
            build_profile.get("build"),
            f"ingestion.items[{index}].build_profile.build",
        )
        changelist = _exact_integer(
            build_profile.get("changelist"),
            f"ingestion.items[{index}].build_profile.changelist",
        )
        sessions.append(
            {
                "session_id": session_id,
                "replay_sha256": replay_sha256,
                "build": build,
                "changelist": changelist,
                "disposition": disposition,
            }
        )
    sessions.sort(key=lambda item: item["session_id"])
    if len({item["session_id"] for item in sessions}) != len(sessions):
        raise WorldGridProfileError("ingestion report duplicates included session_id")
    if len({item["replay_sha256"] for item in sessions}) != len(sessions):
        raise WorldGridProfileError("ingestion report duplicates included replay_sha256")
    counts = dict(sorted(Counter(item["build"] for item in sessions).items()))
    binding = {
        "ingestion_report_schema_version": "2.0",
        "dataset_schema_version": "2.0.0",
        "accepted_session_count": len(sessions),
        "accepted_session_counts_by_build": counts,
        "accepted_sessions": sessions,
        "complete": True,
        "ingestion_binding_sha256": "",
    }
    binding["ingestion_binding_sha256"] = _payload_hash(
        binding, "ingestion_binding_sha256"
    )
    return binding


@dataclass(frozen=True, slots=True)
class WorldUnitScale:
    meters_per_world_unit: float
    semantic_source: str
    source_url: str


@dataclass(frozen=True, slots=True)
class PublicationAuditReference:
    report_id: str
    audit_sha256: str
    coordinate_audit_sha256: str


@dataclass(frozen=True, slots=True)
class WorldGridEvidence:
    kind: str
    subject: str
    source: str
    semantics: str


@dataclass(frozen=True, slots=True)
class WorldGridProfile:
    schema_version: str
    profile_id: str
    profile_kind: str
    authoritative_terrain_bounds: bool
    compatible_builds: tuple[str, ...]
    world_identity: str
    world_x_min: int
    world_x_max: int
    world_y_min: int
    world_y_max: int
    grid_rows: int
    grid_columns: int
    column_axis: Literal["x"]
    row_axis: Literal["y"]
    column_direction: Literal["increasing"]
    row_direction: Literal["increasing"]
    inclusive_maximum_bounds: bool
    cell_ordering: Literal["row_major"]
    lattice_origin_x: int
    lattice_origin_y: int
    lattice_spacing_world_units: int
    cell_width_world_units: int
    cell_height_world_units: int
    world_unit_scale: WorldUnitScale
    publication_audit: PublicationAuditReference
    evidence: tuple[WorldGridEvidence, ...]
    profile_hash: str

    def __post_init__(self) -> None:
        expected = {
            "schema_version": WORLD_GRID_PROFILE_SCHEMA_VERSION,
            "profile_id": WORLD_GRID_PROFILE_ID,
            "profile_kind": WORLD_GRID_PROFILE_KIND,
            "authoritative_terrain_bounds": False,
            "compatible_builds": COMPATIBLE_BUILDS,
            "world_identity": WORLD_IDENTITY,
            "world_x_min": WORLD_X_MIN,
            "world_x_max": WORLD_X_MAX,
            "world_y_min": WORLD_Y_MIN,
            "world_y_max": WORLD_Y_MAX,
            "grid_rows": GRID_ROWS,
            "grid_columns": GRID_COLUMNS,
            "column_axis": "x",
            "row_axis": "y",
            "column_direction": "increasing",
            "row_direction": "increasing",
            "inclusive_maximum_bounds": True,
            "cell_ordering": "row_major",
            "lattice_origin_x": LATTICE_ORIGIN,
            "lattice_origin_y": LATTICE_ORIGIN,
            "lattice_spacing_world_units": LATTICE_SPACING_WORLD_UNITS,
            "cell_width_world_units": CELL_WIDTH_WORLD_UNITS,
            "cell_height_world_units": CELL_HEIGHT_WORLD_UNITS,
        }
        for name, expected_value in expected.items():
            if getattr(self, name) != expected_value:
                raise WorldGridProfileError(
                    f"{name} must equal engineered profile value "
                    f"{expected_value!r}"
                )
        if (
            self.world_x_max - self.world_x_min
            != self.grid_columns * self.cell_width_world_units
            or self.world_y_max - self.world_y_min
            != self.grid_rows * self.cell_height_world_units
            or self.cell_width_world_units % self.lattice_spacing_world_units
            or self.cell_height_world_units % self.lattice_spacing_world_units
        ):
            raise WorldGridProfileError("integer lattice/cell geometry is inconsistent")
        if (
            _finite(
                self.world_unit_scale.meters_per_world_unit,
                "world_unit_scale.meters_per_world_unit",
            )
            != METERS_PER_WORLD_UNIT
        ):
            raise WorldGridProfileError(
                "world_unit_scale.meters_per_world_unit must be 0.01"
            )
        _string(
            self.world_unit_scale.semantic_source,
            "world_unit_scale.semantic_source",
        )
        _string(self.world_unit_scale.source_url, "world_unit_scale.source_url")
        if self.publication_audit.report_id != WORLD_GRID_AUDIT_ID:
            raise WorldGridProfileError("publication_audit.report_id is invalid")
        _sha256(
            self.publication_audit.audit_sha256,
            "publication_audit.audit_sha256",
        )
        _sha256(
            self.publication_audit.coordinate_audit_sha256,
            "publication_audit.coordinate_audit_sha256",
        )
        subjects = {item.subject for item in self.evidence}
        if not _EXPECTED_EVIDENCE_SUBJECTS <= subjects:
            raise WorldGridProfileError(
                "profile evidence must include engineered definition, reported "
                "metadata containment, and coordinate coverage"
            )
        for item in self.evidence:
            for name in item.__slots__:
                _string(getattr(item, name), f"evidence.{name}")
            if item.subject == "model_world_partition" and item.kind != (
                "engineered_definition"
            ):
                raise WorldGridProfileError(
                    "model partition bounds require engineered_definition evidence"
                )
        _sha256(self.profile_hash, "profile_hash")
        if self.compute_hash() != self.profile_hash:
            raise WorldGridProfileError("profile_hash does not match profile content")

    @classmethod
    def from_dict(cls, raw: Any) -> WorldGridProfile:
        value = _require_object(raw, location="profile", fields=_PROFILE_FIELDS)
        scale = _require_object(
            value["world_unit_scale"],
            location="profile.world_unit_scale",
            fields={
                "meters_per_world_unit",
                "semantic_source",
                "source_url",
            },
        )
        audit = _require_object(
            value["publication_audit"],
            location="profile.publication_audit",
            fields={
                "report_id",
                "audit_sha256",
                "coordinate_audit_sha256",
            },
        )
        builds = value["compatible_builds"]
        if not isinstance(builds, list):
            raise WorldGridProfileError("compatible_builds must be a JSON array")
        raw_evidence = value["evidence"]
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise WorldGridProfileError("profile.evidence must be a nonempty array")
        evidence: list[WorldGridEvidence] = []
        for index, item in enumerate(raw_evidence):
            entry = _require_object(
                item,
                location=f"profile.evidence[{index}]",
                fields={"kind", "subject", "source", "semantics"},
            )
            evidence.append(
                WorldGridEvidence(
                    *(
                        _string(entry[name], f"profile.evidence[{index}].{name}")
                        for name in ("kind", "subject", "source", "semantics")
                    )
                )
            )
        integer_fields = (
            "world_x_min",
            "world_x_max",
            "world_y_min",
            "world_y_max",
            "grid_rows",
            "grid_columns",
            "lattice_origin_x",
            "lattice_origin_y",
            "lattice_spacing_world_units",
            "cell_width_world_units",
            "cell_height_world_units",
        )
        integers = {
            name: _exact_integer(value[name], name) for name in integer_fields
        }
        for name in (
            "authoritative_terrain_bounds",
            "inclusive_maximum_bounds",
        ):
            if type(value[name]) is not bool:
                raise WorldGridProfileError(f"{name} must be boolean")
        return cls(
            schema_version=_string(value["schema_version"], "schema_version"),
            profile_id=_string(value["profile_id"], "profile_id"),
            profile_kind=_string(value["profile_kind"], "profile_kind"),
            authoritative_terrain_bounds=value["authoritative_terrain_bounds"],
            compatible_builds=tuple(
                _string(build, "compatible_builds[]") for build in builds
            ),
            world_identity=_string(value["world_identity"], "world_identity"),
            **integers,
            column_axis=value["column_axis"],
            row_axis=value["row_axis"],
            column_direction=value["column_direction"],
            row_direction=value["row_direction"],
            inclusive_maximum_bounds=value["inclusive_maximum_bounds"],
            cell_ordering=value["cell_ordering"],
            world_unit_scale=WorldUnitScale(
                _finite(
                    scale["meters_per_world_unit"],
                    "world_unit_scale.meters_per_world_unit",
                ),
                _string(
                    scale["semantic_source"],
                    "world_unit_scale.semantic_source",
                ),
                _string(scale["source_url"], "world_unit_scale.source_url"),
            ),
            publication_audit=PublicationAuditReference(
                _string(audit["report_id"], "publication_audit.report_id"),
                _sha256(
                    audit["audit_sha256"],
                    "publication_audit.audit_sha256",
                ),
                _sha256(
                    audit["coordinate_audit_sha256"],
                    "publication_audit.coordinate_audit_sha256",
                ),
            ),
            evidence=tuple(evidence),
            profile_hash=_sha256(value["profile_hash"], "profile_hash"),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_hash: str | None = None,
        audit_path: str | Path | None = None,
        ingestion_report_path: str | Path | None = None,
        audit_binding_mode: Literal["canonical", "independent"] = "canonical",
    ) -> WorldGridProfile:
        if audit_binding_mode not in {"canonical", "independent"}:
            raise WorldGridProfileError(
                "audit_binding_mode must be canonical or independent"
            )
        profile = cls.from_dict(_read_json(path, "WorldGridProfile"))
        if expected_hash is not None and profile.profile_hash != _sha256(
            expected_hash, "expected_profile_hash"
        ):
            raise WorldGridProfileError(
                "profile hash does not match the run configuration"
            )
        if audit_path is not None:
            if ingestion_report_path is None:
                raise WorldGridProfileError(
                    "publication audit verification requires the current "
                    "ingestion-report path"
                )
            profile.verify_audit(
                audit_path,
                ingestion_report_path,
                binding_mode=audit_binding_mode,
            )
        elif ingestion_report_path is not None:
            raise WorldGridProfileError(
                "ingestion-report binding requires publication audit verification"
            )
        return profile

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "profile_kind": self.profile_kind,
            "authoritative_terrain_bounds": self.authoritative_terrain_bounds,
            "compatible_builds": list(self.compatible_builds),
            "world_identity": self.world_identity,
            "world_x_min": self.world_x_min,
            "world_x_max": self.world_x_max,
            "world_y_min": self.world_y_min,
            "world_y_max": self.world_y_max,
            "grid_rows": self.grid_rows,
            "grid_columns": self.grid_columns,
            "column_axis": self.column_axis,
            "row_axis": self.row_axis,
            "column_direction": self.column_direction,
            "row_direction": self.row_direction,
            "inclusive_maximum_bounds": self.inclusive_maximum_bounds,
            "cell_ordering": self.cell_ordering,
            "lattice_origin_x": self.lattice_origin_x,
            "lattice_origin_y": self.lattice_origin_y,
            "lattice_spacing_world_units": self.lattice_spacing_world_units,
            "cell_width_world_units": self.cell_width_world_units,
            "cell_height_world_units": self.cell_height_world_units,
            "world_unit_scale": {
                "meters_per_world_unit": (
                    self.world_unit_scale.meters_per_world_unit
                ),
                "semantic_source": self.world_unit_scale.semantic_source,
                "source_url": self.world_unit_scale.source_url,
            },
            "publication_audit": {
                "report_id": self.publication_audit.report_id,
                "audit_sha256": self.publication_audit.audit_sha256,
                "coordinate_audit_sha256": (
                    self.publication_audit.coordinate_audit_sha256
                ),
            },
            "evidence": [
                {
                    "kind": item.kind,
                    "subject": item.subject,
                    "source": item.source,
                    "semantics": item.semantics,
                }
                for item in self.evidence
            ],
            "profile_hash": self.profile_hash,
        }

    def compute_hash(self) -> str:
        return _payload_hash(self.to_dict(), "profile_hash")

    def verify_audit(
        self,
        path: str | Path,
        ingestion_report_path: str | Path,
        *,
        binding_mode: Literal["canonical", "independent"] = "canonical",
    ) -> None:
        if binding_mode not in {"canonical", "independent"}:
            raise WorldGridProfileError(
                "binding_mode must be canonical or independent"
            )
        audit = _require_object(
            _read_json(
                path,
                "world-grid publication audit",
                preserve_float_tokens=True,
            ),
            location="publication audit",
            fields=_AUDIT_FIELDS,
        )
        if audit["schema_version"] != WORLD_GRID_AUDIT_SCHEMA_VERSION:
            raise WorldGridProfileError("unsupported publication audit schema_version")
        if audit["audit_id"] != self.publication_audit.report_id:
            raise WorldGridProfileError("publication audit identity mismatch")
        audit_hash = _sha256(audit["audit_hash"], "audit_hash")
        if (
            audit_hash != self.publication_audit.audit_sha256
            or _payload_hash(audit, "audit_hash") != audit_hash
        ):
            raise WorldGridProfileError("publication audit hash mismatch")

        coordinate_audit = audit["coordinate_audit"]
        if not isinstance(coordinate_audit, dict):
            raise WorldGridProfileError("coordinate_audit must be an object")
        coordinate_hash = _sha256(
            coordinate_audit.get("coordinate_audit_hash"),
            "coordinate_audit.coordinate_audit_hash",
        )
        if (
            coordinate_hash != self.publication_audit.coordinate_audit_sha256
            or _payload_hash(
                coordinate_audit, "coordinate_audit_hash"
            )
            != coordinate_hash
        ):
            raise WorldGridProfileError("coordinate audit hash mismatch")
        sources = coordinate_audit.get("coordinate_sources")
        if not isinstance(sources, dict) or set(sources) != _EXPECTED_COORDINATE_SOURCES:
            raise WorldGridProfileError(
                "coordinate audit must contain all seven absolute-position streams"
            )

        publication = audit["publication"]
        if (
            not isinstance(publication, dict)
            or publication.get("status") != "published"
            or publication.get("required_coordinate_audit_sha256")
            != coordinate_hash
            or publication.get("approved_coordinate_audit_sha256")
            != coordinate_hash
            or WORLD_GRID_PROFILE_ID + ".json"
            not in publication.get("profile_files", [])
        ):
            raise WorldGridProfileError(
                "publication audit does not authorize this profile"
            )
        compatibility = audit["compatibility_decision"]
        engineered_validation = audit["engineered_validation"]
        if (
            not isinstance(compatibility, dict)
            or compatibility.get("status") != "compatible"
            or not isinstance(engineered_validation, dict)
            or engineered_validation.get("passed") is not True
        ):
            raise WorldGridProfileError(
                "publication audit engineered validation did not pass"
            )
        self._verify_engineered_partition(audit["engineered_partition"])

        dataset_binding = _require_object(
            coordinate_audit.get("dataset_binding"),
            location="coordinate_audit.dataset_binding",
            fields=_DATASET_BINDING_FIELDS,
        )
        binding_hash = _sha256(
            dataset_binding["ingestion_binding_sha256"],
            "dataset_binding.ingestion_binding_sha256",
        )
        if (
            dataset_binding.get("complete") is not True
            or _payload_hash(
                dataset_binding, "ingestion_binding_sha256"
            )
            != binding_hash
        ):
            raise WorldGridProfileError("dataset binding hash mismatch")
        current_report = _read_json(ingestion_report_path, "ingestion report")
        current_binding = ingestion_binding_from_report(current_report)
        if binding_mode == "canonical":
            if current_binding != dataset_binding:
                raise WorldGridProfileError(
                    "current ingestion report does not match publication audit binding"
                )
            self._verify_replay_bindings(audit["replays"], current_binding)
            return

        # An independent corpus is not allowed to redefine the published
        # profile. First prove that the publication's embedded binding is
        # internally sound, then check every current accepted session against
        # the same reviewed build/world/map-grid contract.
        self._verify_replay_bindings(audit["replays"], dataset_binding)
        self._verify_independent_ingestion(
            current_report,
            current_binding,
            audit.get("unique_values_by_build"),
        )

    def _verify_independent_ingestion(
        self,
        report: dict[str, Any],
        binding: dict[str, Any],
        raw_unique_values: Any,
    ) -> None:
        if not isinstance(raw_unique_values, dict):
            raise WorldGridProfileError(
                "publication audit unique_values_by_build must be an object"
            )
        items = report.get("items")
        if not isinstance(items, list):
            raise WorldGridProfileError("ingestion report items must be an array")
        included: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(items):
            if not isinstance(raw, dict) or raw.get("disposition") not in {
                "included_attested",
                "included_unattested",
            }:
                continue
            session_id = raw.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise WorldGridProfileError(
                    f"ingestion.items[{index}].session_id must be nonempty"
                )
            if session_id in included:
                raise WorldGridProfileError(
                    f"ingestion report duplicates included session {session_id}"
                )
            included[session_id] = raw

        geometry_fields = {
            "start_x": "world_x_min",
            "end_x": "world_x_max",
            "start_y": "world_y_min",
            "end_y": "world_y_max",
            "spacing_x": "spacing_x",
            "spacing_y": "spacing_y",
            "count_x": "count_x",
            "count_y": "count_y",
            "total_size_x": "total_size_x",
            "total_size_y": "total_size_y",
        }
        for session in binding["accepted_sessions"]:
            session_id = session["session_id"]
            raw = included.get(session_id)
            if raw is None:
                raise WorldGridProfileError(
                    f"independent ingestion session {session_id} is missing"
                )
            build = session["build"]
            changelist = session["changelist"]
            world_identity = raw.get("world_identity")
            if (
                EXPECTED_CHANGELISTS.get(build) != changelist
                or not isinstance(world_identity, str)
                or not self.supports(build, world_identity)
            ):
                raise WorldGridProfileError(
                    f"session {session_id} has unsupported build/world identity"
                )
            allowed = raw_unique_values.get(build)
            if not isinstance(allowed, dict):
                raise WorldGridProfileError(
                    f"publication audit has no geometry contract for build {build}"
                )
            allowed_worlds = allowed.get("world_identity")
            if (
                not isinstance(allowed_worlds, list)
                or world_identity not in allowed_worlds
            ):
                raise WorldGridProfileError(
                    f"session {session_id} world identity is not publication-approved"
                )
            map_grid = raw.get("map_grid")
            if not isinstance(map_grid, dict):
                raise WorldGridProfileError(
                    f"session {session_id} has no map-grid geometry"
                )
            for report_name, audit_name in geometry_fields.items():
                value = map_grid.get(report_name)
                candidates = allowed.get(audit_name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not isinstance(candidates, list)
                    or not any(
                        not isinstance(candidate, bool)
                        and isinstance(candidate, (int, float))
                        and math.isfinite(float(candidate))
                        and float(candidate) == float(value)
                        for candidate in candidates
                    )
                ):
                    raise WorldGridProfileError(
                        f"session {session_id} map-grid {report_name} differs "
                        "from the publication audit"
                    )

    @staticmethod
    def _verify_engineered_partition(raw: Any) -> None:
        if not isinstance(raw, dict):
            raise WorldGridProfileError("engineered_partition must be an object")
        expected = {
            "profile_id": WORLD_GRID_PROFILE_ID,
            "profile_kind": WORLD_GRID_PROFILE_KIND,
            "authoritative_terrain_bounds": False,
            "world_x_min": WORLD_X_MIN,
            "world_x_max": WORLD_X_MAX,
            "world_y_min": WORLD_Y_MIN,
            "world_y_max": WORLD_Y_MAX,
            "grid_rows": GRID_ROWS,
            "grid_columns": GRID_COLUMNS,
            "column_axis": "x",
            "row_axis": "y",
            "column_direction": "increasing",
            "row_direction": "increasing",
            "inclusive_maximum_bounds": True,
            "cell_ordering": "row_major",
            "lattice_origin_x": LATTICE_ORIGIN,
            "lattice_origin_y": LATTICE_ORIGIN,
            "lattice_spacing_world_units": LATTICE_SPACING_WORLD_UNITS,
            "cell_width_world_units": CELL_WIDTH_WORLD_UNITS,
            "cell_height_world_units": CELL_HEIGHT_WORLD_UNITS,
            "meters_per_world_unit": METERS_PER_WORLD_UNIT,
        }
        if raw != expected:
            raise WorldGridProfileError(
                "publication audit engineered geometry does not match profile"
            )

    @staticmethod
    def _verify_replay_bindings(
        raw_replays: Any,
        dataset_binding: dict[str, Any],
    ) -> None:
        if not isinstance(raw_replays, list):
            raise WorldGridProfileError("publication audit replays must be an array")
        by_hash: dict[str, dict[str, Any]] = {}
        for raw in raw_replays:
            if not isinstance(raw, dict):
                raise WorldGridProfileError("publication audit replay is not an object")
            replay_hash = _sha256(
                raw.get("replay_sha256"), "audit replay_sha256"
            )
            if replay_hash in by_hash:
                raise WorldGridProfileError(
                    f"publication audit duplicates replay {replay_hash}"
                )
            by_hash[replay_hash] = raw
        for session in dataset_binding["accepted_sessions"]:
            audited = by_hash.get(session["replay_sha256"])
            if (
                audited is None
                or audited.get("session_id") != session["session_id"]
                or audited.get("build") != session["build"]
                or audited.get("changelist") != session["changelist"]
                or audited.get("world_identity") != WORLD_IDENTITY
                or EXPECTED_CHANGELISTS.get(session["build"])
                != session["changelist"]
            ):
                raise WorldGridProfileError(
                    f"session {session['session_id']} has no exact "
                    "replay/build/world publication binding"
                )

    def supports(self, build: str, world_identity: str) -> bool:
        return build in self.compatible_builds and world_identity == self.world_identity

    def normalize(self, world_xy: Iterable[float]) -> tuple[float, float]:
        values = tuple(world_xy)
        if len(values) != 2:
            raise WorldGridProfileError("world_xy must contain exactly two values")
        x = _finite(values[0], "world_xy.x")
        y = _finite(values[1], "world_xy.y")
        return (
            (x - self.world_x_min) / (self.world_x_max - self.world_x_min),
            (y - self.world_y_min) / (self.world_y_max - self.world_y_min),
        )

    def cell(self, world_xy: Iterable[float]) -> int | None:
        values = tuple(world_xy)
        if len(values) != 2:
            raise WorldGridProfileError("world_xy must contain exactly two values")
        x, y = values
        x = _finite(x, "world_xy.x")
        y = _finite(y, "world_xy.y")
        if (
            x < self.world_x_min
            or x > self.world_x_max
            or y < self.world_y_min
            or y > self.world_y_max
        ):
            return None
        column = min(
            self.grid_columns - 1,
            math.floor((x - self.world_x_min) / self.cell_width_world_units),
        )
        row = min(
            self.grid_rows - 1,
            math.floor((y - self.world_y_min) / self.cell_height_world_units),
        )
        return row * self.grid_columns + column
