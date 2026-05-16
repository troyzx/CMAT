import json
import os
from pathlib import Path
import tempfile
import unittest

import matplotlib
import numpy as np


matplotlib.use("Agg")

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "cmat-test-mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_NOTEBOOK = REPO_ROOT / "example.ipynb"


class NotebookSmokeTests(unittest.TestCase):
    def test_example_notebook_uses_preferred_simulation_constructor(self):
        notebook = json.loads(EXAMPLE_NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn("cmat.TTVSimulation(", code)
        self.assertNotIn("cmat.ttv_sim(", code)

    def test_example_notebook_runs_all_code_cells_in_smoke_mode(self):
        notebook = json.loads(EXAMPLE_NOTEBOOK.read_text(encoding="utf-8"))
        namespace = {"__name__": "__main__"}
        original_cwd = Path.cwd()
        original_smoke = os.environ.get("CMAT_NOTEBOOK_SMOKE")

        os.environ["CMAT_NOTEBOOK_SMOKE"] = "1"
        try:
            os.chdir(REPO_ROOT)
            for cell in notebook["cells"]:
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                if not source.strip():
                    continue
                exec(compile(source, str(EXAMPLE_NOTEBOOK), "exec"), namespace)
        finally:
            os.chdir(original_cwd)
            if original_smoke is None:
                os.environ.pop("CMAT_NOTEBOOK_SMOKE", None)
            else:
                os.environ["CMAT_NOTEBOOK_SMOKE"] = original_smoke
            import matplotlib.pyplot as plt

            plt.close("all")

        rs = namespace["rs"]
        mp2s = namespace["mp2s"]
        chi2_surface = namespace["chi2_surface"]
        reduced_chi2_surface = namespace["reduced_chi2_surface"]
        rms_surface = namespace["rms_surface"]
        results2d = namespace["results2d"]
        rlt_chi2_curve = namespace["rlt_chi2_curve"]
        rlt_rms_curve = namespace["rlt_rms_curve"]

        self.assertEqual(len(namespace["ttv_sim"].ttv_results), len(rs) * len(mp2s))
        self.assertEqual(chi2_surface.shape, (len(mp2s), len(rs)))
        self.assertEqual(reduced_chi2_surface.shape, (len(mp2s), len(rs)))
        self.assertEqual(rms_surface.shape, (len(mp2s), len(rs)))
        self.assertEqual(results2d.shape, (len(mp2s), len(rs)))
        self.assertEqual(rlt_chi2_curve.shape, rs.shape)
        self.assertEqual(rlt_rms_curve.shape, rs.shape)
        self.assertTrue(np.isfinite(chi2_surface).any())
        self.assertTrue(np.isfinite(reduced_chi2_surface).any())
        self.assertTrue(np.isfinite(results2d).all())


if __name__ == "__main__":
    unittest.main()
