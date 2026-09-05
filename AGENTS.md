# MerchantShield

## Objective

Build a defensive merchant-onboarding risk investigator for the Razorpay AI Buildathon.

## Required architecture

- Use a hybrid of deterministic rules, tabular ML, graph analysis, LLM reasoning, and human review.
- Implement a system-level sparse mixture of experts, not a newly trained neural MoE.
- Use LangGraph for state, routing, and human-review interruptions.
- Use at most two LLM agents: an investigator and an evidence-grounding reviewer.
- Deterministic code owns numerical risk scoring and permitted actions.
- LLMs must not autonomously blacklist or permanently reject merchants.

## Required evaluation

- Use a labelled synthetic dataset split by fraud-ring ID or connected component to prevent leakage.
- Report precision, recall, PR-AUC, false-positive rate, ring recall, manual-review rate, and false-positive cost.
- Compare rules-only, tabular-only, graph-only, hybrid, and routed-MoE systems.
- Never present synthetic metrics as production Razorpay performance.

## Safety

- Use synthetic identities and documents only.
- Do not implement offensive fraud-generation capabilities.
- Do not commit API keys, PII, uploaded documents, databases, or virtual environments.
- Route high-risk applications to human review.
- Never treat missing evidence as clean evidence.

## Commands

- Install: `python -m pip install -r requirements.txt`
- Test: `python -m pytest -q`
- Run API: `./run.sh`

## Working rules

- Preserve existing passing tests and add tests for every behavioral change.
- Run the full test suite after each milestone.
- Make focused commits.
- Explain assumptions and known limitations.
- Do not add infrastructure unrelated to the September 4 demonstration.
