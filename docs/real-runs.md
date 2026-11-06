# Real runs

Runs with the real model (`gpt-6-luna`, reasoning effort `low`) on 2026-10-08, each with
`--max-sentences 20 --batch-size 10`, so 2 batches and 2 LLM calls per run. These runs are not
scored against gold labels (that needs the A4 `kaznerd_test` source); they check that the
system works end to end on different kinds of text and show what the normaliser repairs.

| Article | Status | Entities | For review | Tokens in / out | Cost |
|---|---|---|---|---|---|
| Абай Құнанбайұлы (biography) | completed | 46 | 0 | 8538 / 2333 | $0.0020 |
| Алматы (city) | completed | 52 | 1 | 8081 / 2244 | $0.0019 |
| Назарбаев Университеті (organisation) | completed | 84 | 0 | 9089 / 3742 | $0.0028 |
| Желтоқсан көтерілісі (historical event) | completed | 38 | 2 | 8412 / 1656 | $0.0017 |
| Жоқ мақала 12345 (missing article) | failed, readable reason, 0 LLM calls | – | – | 0 / 0 | $0 |

After the fixes below:

| Article | Status | Entities | For review | Tokens in / out | Cost |
|---|---|---|---|---|---|
| Алматы | completed | 50 | 0 | 8081 / 2024 | $0.0018 |
| Желтоқсан көтерілісі | completed | 43 | 1 (two overlapping spans from the model) | 8668 / 1940 | $0.0018 |

Total cost of all real runs: about $0.012. Every LLM call succeeded on the first attempt;
calls took 8–18 s per batch of 10 sentences. No agent was over 40% of the calls in any run.

## What the runs showed

- **The model miscounts word indices.** It sometimes included a closing quote `”` in a span.
  The BoundaryNormalizer found the span by its text and moved it (6 spans in one sentence).
- **The model writes the dictionary form of a word** (`Қазақстан` for `Қазақстанның`). The
  normaliser dropped such spans as "text not found". Fixed: a text that differs from the words
  only by a suffix on the last word (at most 12 letters) is now accepted.
- **The first version of that fix was too loose.** It also accepted a closing quote or bracket
  as an "ending", so spans that were one word too long (`“Желтоқсан құрбандарын жоқтау ”`)
  passed unrepaired. Found while preparing the report; now the ending must be letters inside
  the last word. Re-normalising the stored model answers of all six runs offline (no new API
  calls) gives: second Желтоқсан run 8 spans moved + 1 overlap dropped, 1 sentence for review;
  Алматы 0 sentences for review; the other runs unchanged.
- **Dots inside quotes and brackets** (`“Желтоқсан. 1986. Алматы.”`, `(1991, реж. Т.Теменов)`)
  split sentences. Fixed: the splitter does not end a sentence inside a matching pair of quotes
  or brackets, and `реж.` is a known abbreviation.
- **Labels that look odd but follow KazNERD:** ages (`10 жасқа`) are DATE (735 such I-DATE tokens
  in the training split), and brackets can be inside an entity.
- **Overlapping spans** from the model are dropped and the sentence goes to the review list.

## Assignment 4: mBERT + LLM + Verifier (2026-11-06)

Fine-tuned mBERT (the pilot's `mbert_100pct`) and `gpt-6-luna` (reasoning effort `low`) annotate
independently; the Verifier settles their disagreements; the Evaluator scores against gold.

### KazNERD test sample, 200 sentences (gold)

`kna run --source kaznerd_test --max-sentences 200 --batch-size 20 --seed 13 --llm openai`
(run `20261106-103314-de2d`, 10 batches, 20 LLM calls, 62 907 / 18 385 tokens, $0.0155, 3 min 18 s).

| Labels | Precision | Recall | F1 |
|---|---|---|---|
| mBERT | 0.941 | 0.949 | **0.945** |
| LLM (few-shot) | 0.810 | 0.738 | 0.773 |
| Final, after the Verifier | 0.900 | 0.916 | 0.908 |

mBERT's 0.945 equals the pilot's F1 on the full test split, so the sample is representative.
The Verifier agreed on 169 entities and adjudicated 75 regions (no escalations). Checked against
gold: mBERT was right in 62 of the 75 regions, the LLM in 10; the LLM adjudicator kept the right
version in 51 of 75 (68%) and chose the LLM's version 34 times, 24 of them wrongly. Its errors
are systematic: it reasons about meaning, not about KazNERD's conventions (an age such as
«жетпіске» and «күнделікті» "daily" are DATE in KazNERD; the adjudicator calls them "not a
date"). **On KazNERD's own domain the LLM adjudication lowers F1 from 0.945 (mBERT alone) to
0.908.**

### Kazakh Wikipedia, 20 sentences each (no gold: agreement)

| Article | Entity agreement (F1) | Token agreement | Adjudicated: kept mBERT / LLM / neither | Cost |
|---|---|---|---|---|
| KazNERD test sample, for comparison | 0.743 | 0.955 | 41 / 34 / 0 | |
| Абай Құнанбайұлы | 0.625 | 0.909 | 8 / 16 / 0 | $0.0032 |
| Алматы | 0.827 | 0.945 | 5 / 6 / 0 | $0.0027 |
| Желтоқсан көтерілісі | 0.452 | 0.847 | 6 / 19 / 1 | $0.0035 |

Outside KazNERD's domain the two annotators agree less on the biography and the historical
article (many names, titles and quoted works) and more on the city article. The adjudicator
sides with the LLM more often there; without gold it is not yet known who is right — that needs
a manually checked Wikipedia sample (exam stage).

Every run stayed under the 40% load limit (max 22.7% on the 200-sentence run, 23.1% on the
Wikipedia runs). Total cost of these four runs: about $0.025.

### What this means for the design

The Verifier's policy matters more than expected. Options to compare at the exam stage:
keep the mBERT version on KazNERD-domain text and adjudicate only out of domain; give the
adjudicator KazNERD-specific few-shot examples of the conventions it gets wrong; or escalate
every disagreement where the LLM would overrule mBERT.
