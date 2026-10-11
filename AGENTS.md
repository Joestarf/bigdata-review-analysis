# Repository instructions

- Run in archlinux WSL with the existing virtual environment.
- Respect the existing .gitignore. data/, models/, generated CSV/JSON, full pair output and sealed annotation material stay local. Never force-add ignored files without explicit authorization naming those files.
- Git push authorization covers code, documentation and safe figures; it does not override data exclusions.
- Publish notebooks only with empty outputs and execution_count=null. Keep executed notebooks local.
- Before committing or pushing, run .venv/bin/python scripts/check_git_scope.py and inspect the staged file list.
- Real-pair labels are AI-assisted judgments, never a human gold standard. Never invent missing labels or use test labels before the model lock.
- Preserve the original sample and original annotation batches. Do not adjust models using the final test.
