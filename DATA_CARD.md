# Synthetic task and preference data

All 216 examples are newly generated from bounded specifications. They contain no employer content, copied benchmark solutions, personal records or collected human judgments.

There are three arithmetic families (addition, subtraction, multiplication), three corresponding Python-expression families, and three simulated routing families (weather, calculator, search). Each family has 16 training groups, 4 validation groups and 4 test groups. Group IDs 0–15 are training, 16–19 validation and 20–23 test. Arguments and operands derive from group IDs; split parameter ranges therefore differ. Families and instruction templates are shared, so this is compositional/range holdout, not unseen-family generalization.

The preferred response is generated from the specification. Rejected arithmetic answers are off by one, coding answers contain the wrong constant, and routing answers select the wrong tool. This makes deterministic correctness checks possible but produces a limited and artificial preference distribution. There are no natural-language stylistic preferences, annotator disagreements, multilingual data, multi-turn conversations or genuine human-rated comparisons.

Synthetic pairwise labels enable an RLHF-style mechanics demonstration; they do not establish alignment with real human preferences. The blank human-review queue randomizes response order with a fixed seed and supplies reviewer/rationale fields. It is excluded from training. A genuine human-feedback study would require actual reviews, reviewer instructions, disagreement analysis and independently reserved test data.

Every preferred and near-miss answer is checked by an independent payload-based grader. Exact prompt sets and parameter groups are disjoint across splits. The manifest records hashes of the JSONL files. The tests also challenge the graders with unsafe AST nodes, malformed integers, duplicate JSON keys, wrong tool choices and extra schema fields.
