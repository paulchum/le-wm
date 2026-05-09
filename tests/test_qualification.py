import unittest

from lewm_drone import default_bvlos_l1c_evidence_pack, evidence_dashboard_summary


class QualificationEvidenceTest(unittest.TestCase):
    def test_default_pack_tracks_standard_922_and_security_gates(self):
        pack = default_bvlos_l1c_evidence_pack()
        identifiers = {item.identifier for item in pack.items}

        self.assertIn("922-08-containment", identifiers)
        self.assertIn("922-09-c2-link", identifiers)
        self.assertIn("922-10-daa", identifiers)
        self.assertIn("922-11-control-station", identifiers)
        self.assertIn("classified-path-gates", identifiers)
        self.assertEqual(pack.approval_claim, "none")

    def test_release_blockers_include_missing_environmental_evidence(self):
        pack = default_bvlos_l1c_evidence_pack()
        summary = evidence_dashboard_summary(pack)

        self.assertIn("922-12-environmental-envelope", pack.release_blockers())
        self.assertIn("missing", summary["status_counts"])


if __name__ == "__main__":
    unittest.main()
