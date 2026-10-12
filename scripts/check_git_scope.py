"""Reject ignored tracked files and notebook outputs before a Git checkpoint."""
from pathlib import Path
import json
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
def files(args):
    return [p for p in subprocess.check_output(["git",*args,"-z"],cwd=root).decode().split("\0") if p]
errors=[f"Ignored file is tracked: {p}" for p in files(["ls-files","-ci","--exclude-standard"])]
for name in files(["ls-files"]):
    if name.endswith(".ipynb") and (root/name).exists():
        notebook=json.loads((root/name).read_text())
        if any(c.get("outputs") or c.get("execution_count") is not None
               for c in notebook.get("cells",[]) if c.get("cell_type")=="code"):
            errors.append(f"Notebook outputs must stay local: {name}")
if "deliverables/team_handoff.zip" in files(["ls-files"]):
    sys.path.insert(0,str(root))
    try:
        from src.package_handoff import verify_public
        verify_public(root/"deliverables/team_handoff.zip")
    except Exception as e:
        errors.append(f"Public handoff bundle failed content verification: {e}")
if errors:
    print("\n".join(errors),file=sys.stderr);sys.exit(1)
print("Git scope check passed: ignored data stays local; tracked notebooks have no outputs.")
