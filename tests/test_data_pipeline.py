import shutil
import unittest
from pathlib import Path

from lewm_drone import DataArtifact, SecureDataPipeline, SecurityGateError


class SecureDataPipelineTest(unittest.TestCase):
    def setUp(self):
        self.workdir = Path("artifacts") / "test_data_pipeline"
        self.workdir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_creates_review_gated_bundle_manifest(self):
        artifact_path = self.workdir / "mission.jsonl"
        artifact_path.write_text('{"sequence_id":0}\n', encoding="utf-8")

        bundle = SecureDataPipeline().create_bundle(
            [
                DataArtifact(
                    name="mission",
                    path=str(artifact_path),
                    artifact_type="mission_log",
                    classification_label="protected_b",
                )
            ],
            bundle_id="bundle-001",
            output_path=self.workdir / "bundle.json",
        )

        self.assertEqual(bundle.bundle_id, "bundle-001")
        self.assertEqual(bundle.export_status, "review_required")
        self.assertEqual(len(bundle.artifacts), 1)
        self.assertIn("controlled_goods_screen", bundle.review_gates)
        self.assertTrue((self.workdir / "bundle.json").exists())

    def test_blocks_classified_and_controlled_goods_labels(self):
        artifact_path = self.workdir / "payload.bin"
        artifact_path.write_text("blocked", encoding="utf-8")

        for label in ("classified", "secret", "controlled_goods", "top_secret"):
            with self.subTest(label=label):
                with self.assertRaises(SecurityGateError):
                    SecureDataPipeline().create_bundle(
                        [
                            DataArtifact(
                                name="payload",
                                path=str(artifact_path),
                                artifact_type="payload_data",
                                classification_label=label,
                            )
                        ],
                        bundle_id="blocked",
                    )


if __name__ == "__main__":
    unittest.main()
