# Real runs (Assignment 2)

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
