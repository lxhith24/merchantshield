# MerchantShield — five-minute, two-person demo script

Target runtime: **4:45–5:00**. Speak calmly at roughly 135 words per minute.
Use **Johan** and **Teammate** as placeholders if the second speaker wants a
different name.

## Before recording

- Start the app and confirm all three pages load.
- Keep `docs/merchantshield-architecture.svg` open in the first tab.
- Keep the deployed MerchantShield link open in the second tab.
- In Finder, select the six files inside `demo_uploads/six_file_ring/` so they
  can all be uploaded together.
- Use a clean browser window at 100% zoom and close unrelated tabs.
- Do not call the result a conviction, verified fraud or production metric.

---

## 0:00–0:38 — The problem

**[SCREEN: Architecture diagram. Do not open the app yet.]**

**Johan:**

“The Reserve Bank of India’s 2024–25 report recorded 13,516 card-and-internet
fraud cases—the largest fraud category by number.

Yet many risk systems begin after a suspicious merchant is already inside.
We asked: what if the risk is invisible in one application, but obvious in the
connections between many?”

**Teammate:**

“MerchantShield investigates that gap during onboarding, before a machine is
allowed to make a permanent decision.”

## 0:38–1:20 — Architecture in plain English

**[SCREEN: Point across the architecture diagram from left to right.]**

**Teammate:**

“A merchant application enters through our API. Shared accounts, devices, IPs,
addresses and timing become an evidence graph.

Rules catch obvious inconsistencies. A graph model understands relationships.
A tabular model studies the individual merchant.”

**Johan:**

“This is a system-level sparse mixture of experts—not a neural MoE or a room
full of chatbots. Rules and graph analysis always run. The tabular specialist
joins when evidence is uncertain, missing or contradictory. LangGraph manages
state, investigation and human interruption.

Deterministic code owns the score. A human owns the final action. An LLM may
explain cited evidence; it cannot blacklist or permanently reject.”

## 1:20–2:47 — Six-file live investigation

**[SCREEN: Open the deployed link. Click “Check merchants.”]**

**Johan:**

“Imagine an operations employee receives a day’s onboarding batch. Production
would supply these fields through an event stream. Here we use synthetic CSVs.”

**[ACTION: Click upload. Select all six CSV files from
`demo_uploads/six_file_ring/`. Pause so the six filenames are visible. Click
“Run live check.”]**

**Teammate:**

“These are six merchants with different names. MerchantShield does not find one
equality and declare fraud. It builds the component, weighs relationships,
checks corroboration and timing, then runs the real workflow.”

**[SCREEN: Results appear. Point to “Potential linked group,” “6 submitted
merchants,” the relationship graph and the similarity table.]**

**Teammate:**

“It found one new group containing all six merchants. Four evidence types
support it: a shared account token, shared devices, paired IPs and paired
addresses—all submitted within one minute.”

**Johan:**

“Notice the language: ‘Potential linked group’ and ‘Human review.’ This is a
risk signal, not a verdict. The batch is compared with itself and our synthetic
reference population, then discarded.”

**[OPTIONAL ACTION: Expand “Verified execution trace” for two seconds.]**

**Johan:**

“This trace shows the real assessment, investigation and routing—not a staged
animation.”

## 2:47–3:48 — Human-review workflow

**[SCREEN: Click “Review.” Then click “Review next case.”]**

**Teammate:**

“The live check discovers a group. The review desk is where a person decides.

We show the linked merchants, strongest similarity, coverage, timing and
supporting evidence. We also show a legitimate alternative, because a shared
office, kiosk or accountant can be genuine.”

**[SCREEN: Briefly show the shared-signal table and investigator explanation.
Scroll to “Your decision.” Choose “Confirm suspicious ring,” but do not click
the final confirmation until the line below.]**

**Johan:**

“The employee can confirm the relationship, request evidence, or clear it with
a verified explanation. Missing evidence is never clean evidence.”

**[ACTION: Click “Confirm and finish.”]**

**Johan:**

“This records a human hold, not an autonomous ban. Only human approval can
reach our simulated gateway.”

## 3:48–4:37 — Evaluation without hiding the weakness

**[SCREEN: Click “Model validation.” Pause on the headline and partition
counts. Scroll to the system comparison table.]**

**Johan:**

“We split labelled synthetic data by complete ring or connected component, so
one ring cannot leak into training and test.”

**Teammate:**

“On 841 untouched applications and 49 rings, the routed system achieved 61.2
percent precision, 82.4 percent recall and 83.5 percent PR-AUC. It found at
least one member in 43 of 49 rings.”

**[SCREEN: Point to false-positive rate, review rate and false-positive cost.]**

**Teammate:**

“False-positive rate was 29.9 percent, and 64.8 percent went to review. These
are synthetic—not production—results.

An earlier stress test reached 99.1 percent recall but only 39.3 percent
precision. We kept that result visible. It is why MerchantShield supports
human investigators instead of automatically rejecting merchants.”

## 4:37–5:00 — Close

**[SCREEN: Return to the top of the app or the architecture diagram.]**

**Johan:**

“Most onboarding systems ask: does this merchant look suspicious?”

**Teammate:**

“MerchantShield asks: who else is this merchant connected to, why, and what
should a reviewer verify next?”

**Together, or Johan:**

“MerchantShield: synthetic evidence, transparent reasoning, human decisions.”

---

## If the upload is slow

Do not fill the silence. Johan says:

“While it runs, the system is building the evidence graph and executing the
same routed workflow used by the review desk. The files remain transient.”

If Community Cloud has cold-started, pause the recording and restart the take.
Do not spend demo time debugging.

## Backup answers for likely judge questions

**Why not simple if/else statements?**

“Rules are one expert and an important baseline. The graph adds component
structure, corroboration and time. Fitted models add learned combinations of
signals. The ablation table measures each layer on the same held-out data.”

**Why LangGraph with no LLM key?**

“LangGraph is the workflow engine: it owns state, bounded routing, human
interruption and resume. The optional LLM is only one node and can be disabled
without disabling the workflow.”

**Is the live batch compared to training data?**

“It is compared with the frozen synthetic reference population and with the
other submitted merchants. The fitted experts were trained only on the training
partition. No fitting or label lookup happens during the upload.”

**Why is precision not higher?**

“Shared infrastructure exists in legitimate coworking, franchise, family and
accountant relationships. Aggressively increasing precision or recall without
independent evidence would hide that ambiguity. We report review workload and
false-positive cost because an operational detector must account for both.”

**Is this production-ready?**

“No. It is an end-to-end synthetic demonstration. Production requires real KYC
verification adapters, access control, retention policy, monitoring,
calibration and testing on representative onboarding data.”

## Source note — not spoken

The opening statistic comes from the Reserve Bank of India Annual Report
2024–25, Table VI.3, which reports 13,516 card/internet fraud cases and states
that digital payments predominated by number. The table concerns reported bank
frauds and is not a measurement of shell-merchant onboarding fraud. Project
metrics come from `data/evaluation/performance_report.json` and
`data/evaluation/expanded_report.json`; all are synthetic.

Official RBI source:
https://www.rbi.org.in/scripts/AnnualReportPublications.aspx?Id=1436
