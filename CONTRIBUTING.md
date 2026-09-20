# Five-person collaboration

1. Clone the private repository to your own computer.
2. Create a task branch, for example `git switch -c codex/data-audit`.
3. Keep changes focused on your assigned module; agree on shared field names, dimensions and units first.
4. Run `python -m pytest -q`; inspect `git diff` and `git status --short`.
5. Stage specific files, commit, push your branch and open a pull request.
6. Have another teammate review before merging. B+C review timing; C+D review training boundaries; A+E review portfolio constraints.
7. Synchronize main after merging. Do not force-push shared branches.

Do not commit raw competition data, private keys, `.env`, model binaries, cached embeddings, output tables, CVs or original provider documents. `.gitignore` cannot remove secrets from old history. Google Drive is for team documents; GitHub is for code and versioned configuration.

Run the full test suite after changing module names, imports or configuration paths. Keep runtime files named by function.
