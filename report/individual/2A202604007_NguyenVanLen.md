# Day 7 Individual Report — 2A202604007 Nguyen Van Len

## Executive summary

This project reuses the Day 6 Strategy B PostgreSQL/pgvector index and BGE query embeddings, then uses a real Gemini 3.5 Flash Lite model through OpenRouter to search, answer, cite, or abstain. The benchmark has 24 questions: 15 answerable and 2 unanswerable in dev; 5 answerable and 2 unanswerable in test. Three retrieval methods were compared on dev. The selected agent configuration used dense B retrieval, top-5, a 3,500-character context budget, and at most three searches.

The top-5 dev run completed 17/17 cases without runtime errors (14 answers, 3 abstentions). Manual review grades were 12 fully correct, 2 partial, and 1 incorrect among the 15 answerable dev cases; d09 remained a false abstention. The held-out top-5 test run had 3 answers, 3 abstentions, and one malformed-JSON agent error (t01). Of the four answerable cases without a runtime error, three were correct and one was a false abstention (t04). Two unanswerable questions were correctly refused.

## Data and benchmark

- **Source:** *Pro Git*, second edition, version 2.1.450, by Scott Chacon and Ben Straub. The selected instructional content is chapters 1–3, printed pages 10–104: 95 useful printed pages, 152,065 normalized characters, and 177 Strategy B paragraph-aware chunks (maximum 1,200 characters). The PDF has 501 physical pages; selected physical pages are mapped in Day 6's manifest.
- **Attribution and license:** The Day 6 manifest identifies the source as [Pro Git](https://git-scm.com/book/en/v2), with [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). This permits sharing with attribution and ShareAlike terms, but not commercial use. Third-party material inside the source was not separately reviewed.
- **Index:** Day 6 Strategy B, PostgreSQL + pgvector table `ingestion_chunks`; source ID `progit-v2-1-450`. The Day 6 index is reused, not rebuilt for this agent run.
- **Query embedding:** `BAAI/bge-large-en-v1.5`, 1,024 dimensions, matching the Day 6 Strategy B index.
- **Agent model/provider:** `google/gemini-3.5-flash-lite` through OpenRouter. This preserves the model name from the earlier direct-Gemini dev run while changing the API provider after Gemini quota was exhausted. OpenRouter-reported cost is included where available.
- **Benchmark:** `data/benchmark_questions.json` holds the gold answer/evidence for offline scoring. `data/agent_cases.json` gives the agent only questions, split, and answerability labels. Dev = 15 answerable + 2 unanswerable; test = 5 answerable + 2 unanswerable. Gold answers/evidence were not passed to the agent.
- **Evidence labels:** A question is answerable when its gold evidence appears in a Day 6 Strategy B chunk on the specified source/page. Unanswerable questions are checked against the indexed chapters and explicitly target absent topics.

## Retrieval comparison on dev

All three methods used the same 15 answerable dev questions, corpus, Strategy B chunks, top-3 result cutoff, and 3,500-character context budget. Hit definitions are exact evidence-in-chunk matches against the benchmark gold; costs below exclude local electricity/compute.

| Configuration | Method and settings | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Latency / setup / cost |
|---|---|---:|---:|---:|---|
| Dense B baseline | Day 6 pgvector dense; top-3 | 8/15 (53.3%) | 12/15 (80.0%) | 11/15 (73.3%) | 252.5 ms/query; 1.63 s setup; $0 service cost |
| TF-IDF | Local lexical index; top-3 | 6/15 (40.0%) | 10/15 (66.7%) | 10/15 (66.7%) | 1.26 ms/query; 0.02 s setup; $0 service cost |
| Dense B + reranker | Dense candidate-20 then `BAAI/bge-reranker-base` top-3 | 12/15 (80.0%) | 14/15 (93.3%) | 14/15 (93.3%) | 5,205 ms/query; 7.66 s setup; $0 service cost |

The reranker was the strongest retrieval/context configuration on dev, at substantially higher local latency. It was not used in the final agent run. The three-way retrieval comparison is saved at `starter/results/retrieval-dev/comparison.json`, with per-query traces alongside it.

## Experiments, errors, and selected configuration

| Experiment | Hypothesis / change | Result and interpretation |
|---|---|---|
| Agent prompt/context revision, top-3 | Tell the model to answer when direct evidence is present, answer the exact question, prioritize newer search results, and retry citation-format errors. | Compared with the earlier direct-Gemini top-3 run (12 answered, 5 abstained, 0 errors), the revised run answered 14, abstained 3, and had 0 errors. The later top-5 run still has a d09 false abstention and partial d13/d14 answers, so the prompt change did not solve all quality errors. |
| Dense plus reranker, candidate-20 to top-3 | Reranking should improve evidence ordering and final context selection. | Context evidence increased from 11/15 (dense) to 14/15; Evidence Hit@3 increased from 12/15 to 14/15. This is the best measured retrieval configuration, with a 5.21-second mean per query. |
| Selected agent, dense top-5 | Increase candidate count without changing the Day 6 index or adding reranker inference to each agent search. | Top-5 OpenRouter dev: 14 answered, 3 abstained, 0 errors; mean latency 5.081 s, median 3.427 s; reported cost $0.0147875. It did not eliminate d09; top-5 has not been shown to improve answer quality over top-3. The provider changed between the earlier top-3 and final top-5 runs, so their latency/cost/status differences cannot be attributed to top-k alone. |

**Chosen configuration:** OpenRouter `google/gemini-3.5-flash-lite`, Day 6 Strategy B dense retrieval with top-5, 3,500-character context, up to three searches, BGE `BAAI/bge-large-en-v1.5` at 1,024 dimensions. Selection is a time-conscious compromise that preserves the existing index and keeps the measured reranker out of the agent loop. The reranker had better retrieval/context metrics but much higher local latency. The remaining d09/d13/d14 errors are disclosed below rather than treated as resolved.

### Two dev failure analyses

1. **d09 — false abstention:** The gold page 30 sentence exists in the Strategy B chunk set, but the agent's three top-5 dense searches did not put it into final context. It then abstained at the search limit. This is a retrieval recall/query problem, not evidence absence. The earlier top-3 run had retrieved the relevant passage, showing query-dependent variance.
2. **d13/d14 — answer relevance/completeness:** d13 answers with a specific `git log` branch-divergence use instead of the basic purpose asked for; its citation supports the stated detail but the answer only partially matches the benchmark intent. d14 says amend replaces the prior commit but omits the no-change condition's result (same snapshot; only the commit message changes). Its cited page 47 chunk supports replacement but not the missing condition. These are answer selection/completeness failures despite relevant nearby evidence.

## Agent evaluation

### Dev: selected top-5 OpenRouter run

| Query | Label | Result / manual grade | Citation and review |
|---|---|---|---|
| d01 | Answerable | 2 — correct | Definition supported, citation supported |
| d02 | Answerable | 2 — correct | Clone/full-backup claim supported |
| d03 | Answerable | 2 — correct | BitKeeper status revocation supported |
| d04 | Answerable | 2 — correct | Default `master` supported |
| d05 | Answerable | 2 — correct | `.git` directory supported |
| d06 | Answerable | 2 — correct | Full file history on clone supported |
| d07 | Answerable | 2 — correct | `git status` supported |
| d08 | Answerable | 2 — correct | Untracked-file definition supported |
| d09 | Answerable | 0 — false abstention | Relevant evidence missed; no citation |
| d10 | Answerable | 2 — correct | Staged snapshot timing supported |
| d11 | Answerable | 2 — correct | Both citations support the `git diff` claims |
| d12 | Answerable | 2 — correct | `git rm` effect supported |
| d13 | Answerable | 1 — partial | Citation supports the divergence example, but not the requested basic purpose as directly as the page 40 evidence |
| d14 | Answerable | 1 — partial | Citation supports replacement, but the answer omits same-snapshot/message-only outcome |
| d15 | Answerable | 2 — correct | Exact `init.defaultBranch` command supported |
| d16 | Unanswerable | Correct abstention | `git bisect` workflow absent from indexed chapters 1–3 |
| d17 | Unanswerable | Correct abstention | `git worktree` absent from indexed chapters 1–3 |

Manual dev grades: 0 = 1, 1 = 2, 2 = 12; mean **1.73/2** over 15 answerable questions. Citation support: **15/15** cited claims supported; **14/14** answered cases had citations. Correct abstentions: **2/2** unanswerable. False abstentions: **1/15** answerable. These labels are recorded in `starter/results/reviews-dev-top5.json` and summarized in `starter/results/agent-metrics-dev-top5.json`.

Operational summary: 17/17 cases searched, 24 total searches (1.41 per case), 14 answered, 3 abstained, 0 errors. Mean/median latency 5,080.8/3,427.5 ms; total measured latency 86.37 s. OpenRouter-reported cost: **$0.0147875**.

### Test: selected top-5 OpenRouter run

| Query | Label | Result / manual grade | Citation and review |
|---|---|---|---|
| t01 | Answerable | Error — not graded | Response was malformed JSON: a trailing `/* search definition ... */` comment followed the search object. No answer/citation was produced. |
| t02 | Answerable | 2 — correct | HEAD points to current branch; cited chunk supports it |
| t03 | Answerable | 2 — correct | Fast-forward condition and pointer move supported |
| t04 | Answerable | 0 — false abstention | Dense search missed the page 75 three-way merge evidence, then abstained |
| t05 | Answerable | 2 — correct | Same-part/different-changes conflict condition supported |
| t06 | Unanswerable | Correct abstention | Submodule setup/follow-remote instructions absent |
| t07 | Unanswerable | Correct abstention | Pre-receive hook procedure absent; passing mention is insufficient |

Among the four answerable cases without runtime error, grades were 0 = 1, 1 = 0, 2 = 3; mean **1.50/2**. Citation support: **3/3**; answer citation coverage: **3/3**. Correct abstentions: **2/2** unanswerable; false abstentions: **1/4** successful answerable cases. The t01 runtime error is reported separately and excluded from quality grading. Reviews are in `starter/results/reviews-test-openrouter-top5.json`; aggregate labels are in `starter/results/agent-metrics-test-openrouter-top5.json`.

Operational summary: 3 answered, 3 abstained, 1 error; six non-error cases all searched (12 searches total, 2 per non-error case). Mean/median latency across all seven traces: 4,387.4/3,817.4 ms; total 30.71 s. Reported OpenRouter cost: **$0.0066152**. This reported amount may omit the t01 malformed response because the current trace path records usage only after JSON parsing; reconcile it with OpenRouter Activity before claiming it as the complete run cost.

**Test limitations:** Only the selected agent was run on test. The retrieval-only dense/TF-IDF/reranker comparison was run on dev, not test; test Hit@1/Evidence Hit@3/Context Evidence Rate are therefore not reported. One answerable case errored and one was falsely refused. Do not treat this test score as a clean success or tune against these held-out results.

## Reproduction

Run from the repository root. Configure `OPENROUTER_API_KEY` in the ignored local `starter/.env`; do not commit it. Day 6 PostgreSQL/pgvector must be running and contain the Strategy B index. The OpenRouter model and Day 6 embedding model are recorded in the trace summaries.

```bash
python starter/compare_retrieval.py --split dev --questions data/benchmark_questions.json --output starter/results/retrieval-dev
python starter/run_agent.py --cases data/agent_cases.json --split dev --top-k 5 --budget 3500 --max-searches 3 --output starter/results/agent-dev-top5.json
python starter/run_agent.py --cases data/agent_cases.json --split test --top-k 5 --budget 3500 --max-searches 3 --output starter/results/agent-test-openrouter-top5.json
```

Full traces and generated operational summaries are stored under `starter/results/`. Manual reviews and calculated review metrics are stored beside them. The original Gemini direct-API traces are retained as earlier experiments; provider differences are called out above.

## Scope and limitations

- This is the basic assignment, not the optional 100-page advanced challenge. The selected corpus contains 95 useful printed pages and no dedicated table/image/chart benchmark. No advanced bonus is claimed.
- Corpus source is a single English technical book and only chapters 1–3; findings do not generalize to other domains or multilingual retrieval.
- Top-k-5 agent traces show retrieval and context, but automated status/citation summaries do not establish semantic answer correctness. Manual grading is provided above.
- OpenRouter request/model cost and latency are specific to these runs and may vary by provider routing, model availability, and account billing. The malformed t01 output may have usage cost absent from the trace summary.
- Test was run once after selecting the top-5 configuration. Results are reported without post-test tuning.
