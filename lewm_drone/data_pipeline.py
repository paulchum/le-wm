"""Secure data-pipeline primitives for reference-platform evidence flows."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .interfaces import JsonValue, to_jsonable
from .reference_platform import DataResidencyPolicy


@dataclass(frozen=True)
class DataArtifact:
    """One local artifact considered for secure mission/evidence export."""

    name: str
    path: str
    artifact_type: str
    classification_label: str = "unclassified"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactDigest:
    """Hash and size metadata for an artifact."""

    name: str
    path: str
    artifact_type: str
    classification_label: str
    sha256: str
    size_bytes: int
    metadata: Mapping[str, JsonValue]


@dataclass(frozen=True)
class EncryptionMetadata:
    """Encryption/KMS expectations for a pipeline bundle."""

    required: bool
    status: str
    algorithm: str = "external_kms_envelope"
    key_residency: str = "canada"
    notes: str = "Repo workflow records encryption requirements; production encryption is externalized."


@dataclass(frozen=True)
class PipelineBundle:
    """Auditable bundle manifest emitted by the secure data pipeline."""

    bundle_id: str
    created_at_s: float
    artifacts: tuple[ArtifactDigest, ...]
    manifest_hash: str
    policy: Mapping[str, JsonValue]
    encryption: EncryptionMetadata
    export_status: str
    review_gates: tuple[str, ...]

    def to_record(self) -> dict[str, JsonValue]:
        return to_jsonable(self)  # type: ignore[return-value]


class SecurityGateError(ValueError):
    """Raised when an artifact violates the selected security posture."""


class SecureDataPipeline:
    """Validate, hash, and manifest mission/evidence artifacts.

    This intentionally does not handle classified material. It creates
    unclassified/protected evidence manifests and blocks labels that require a
    separate secure facility, Controlled Goods review, or classified network.
    """

    def __init__(self, policy: DataResidencyPolicy | None = None):
        self.policy = policy or DataResidencyPolicy()

    def create_bundle(
        self,
        artifacts: Iterable[DataArtifact],
        *,
        bundle_id: str,
        output_path: str | Path | None = None,
        encryption_status: str = "external_kms_required",
    ) -> PipelineBundle:
        digests = tuple(self._digest_artifact(artifact) for artifact in artifacts)
        manifest_payload = {
            "bundle_id": bundle_id,
            "created_at_s": time.time(),
            "artifacts": to_jsonable(digests),
            "policy": to_jsonable(self.policy),
            "export_status": "review_required",
            "review_gates": self.review_gates(),
        }
        manifest_hash = _hash_json(manifest_payload)
        bundle = PipelineBundle(
            bundle_id=bundle_id,
            created_at_s=float(manifest_payload["created_at_s"]),
            artifacts=digests,
            manifest_hash=manifest_hash,
            policy=to_jsonable(self.policy),  # type: ignore[arg-type]
            encryption=EncryptionMetadata(
                required=self.policy.encryption_required,
                status=encryption_status,
            ),
            export_status="review_required",
            review_gates=tuple(self.review_gates()),
        )
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(bundle.to_record(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return bundle

    def review_gates(self) -> list[str]:
        gates = ["hash_chain_verification", "model_provenance_review"]
        if self.policy.export_review_required:
            gates.append("export_control_review")
        if self.policy.audit_required:
            gates.append("audit_log_review")
        gates.extend(("controlled_goods_screen", "classified_network_exclusion"))
        return gates

    def _digest_artifact(self, artifact: DataArtifact) -> ArtifactDigest:
        label = artifact.classification_label.lower()
        if label in self.policy.blocked_labels:
            raise SecurityGateError(f"blocked_classification_label:{label}")
        if label not in self.policy.allowed_labels:
            raise SecurityGateError(f"unknown_classification_label:{label}")

        path = Path(artifact.path)
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise ValueError(f"artifact_not_file:{path}")

        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)

        return ArtifactDigest(
            name=artifact.name,
            path=str(path),
            artifact_type=artifact.artifact_type,
            classification_label=label,
            sha256=digest.hexdigest(),
            size_bytes=size,
            metadata=to_jsonable(artifact.metadata),  # type: ignore[arg-type]
        )


def _hash_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
