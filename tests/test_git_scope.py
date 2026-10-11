"""Check the repository data boundary without changing the real Git index."""
import io
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/"scripts/check_git_scope.py"

class GitScopeTests(unittest.TestCase):
    def test_ignored_tracked_file_is_rejected(self):
        with patch("subprocess.check_output",side_effect=[b"data/reviews.csv\\0",b"data/reviews.csv\\0"]), \
             patch("sys.stderr",new=io.StringIO()):
            with self.assertRaises(SystemExit) as result:runpy.run_path(str(SCRIPT))
        self.assertEqual(result.exception.code,1)

    def test_clean_file_inventory_passes(self):
        with patch("subprocess.check_output",side_effect=[b"",b"src/common.py\\0"]), \
             patch("sys.stdout",new=io.StringIO()) as output:
            runpy.run_path(str(SCRIPT))
        self.assertIn("passed",output.getvalue())

if __name__=="__main__":unittest.main()
