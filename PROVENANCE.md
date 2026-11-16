# Project provenance

The original local experiment was prepared and run on 16 November 2026. Training reports retain their UTC timestamps, exact package versions and pinned model revision.

The first Git commit imports the existing implementation, synthetic data and measured evidence. Later commits record actual cache-portability fixes, report verification, response-grading tools, tests and publication documentation. Commit dates reflect when those changes were recorded.

The experiment plan hash in the original environment report identifies the version in the initial import commit. Later editorial updates preserve the recorded experimental design and hyperparameters. The original plan can be recovered with `git show 8315e45:EXPERIMENT_PLAN.md` and checked against that hash.

The original experiment reports remain separate from the publication checks. `reports/publication_smoke.json` records a later three-task SFT inference check. `reports/publication_verification.json` recomputes grades and checks saved reports without claiming that CI reran training or inspected adapter weights.

Local adapter inspection is recorded in `reports/evidence_audit.json`. Adapter files and the base checkpoint are omitted from Git and can be regenerated using the pinned model and commands. Absolute cache paths in original logs describe the original machine and are not configuration requirements.

All tasks in this repository are generated from the specifications in `post_training_lab/data.py`. No private employer tasks, client prompts or benchmark solutions are included. Preferences are synthetic. Human review fields are blank, and no human annotation study is claimed.
