# Datasets

This project uses two data sources. Neither is redistributed in this repository, except a
small few-shot subset of KazNERD (see below).

## 1. KazNERD — Kazakh Named Entity Recognition Dataset

- **Repository:** <https://github.com/IS2AI/KazNERD>
- **Data files (pinned version, commit `bd4333d`):**
  <https://github.com/IS2AI/KazNERD/tree/bd4333d0f5952b9fafb2ef2ac2fefa0ad3c0333f/KazNERD>
- **Hugging Face mirror (gated):** <https://huggingface.co/datasets/issai/KazNERD>
- **Project page:** <https://issai.nu.edu.kz/kaznerd-eng/>
- **Paper:** R. Yeshpanov, Y. Khassanov, H. A. Varol. *KazNERD: Kazakh Named Entity Recognition
  Dataset.* LREC 2022. <https://aclanthology.org/2022.lrec-1.44/>
- **Licence:** CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>), © ISSAI / IS2AI.

| Split | File | Sentences | Tokens | SHA-256 (as published, LF) |
|---|---|---:|---:|---|
| train | `IOB2_train.txt` | 90,228 | 1,043,305 | `e5a80f6d4bbff499a4bf7cbf96a169feb761411151fb45abebeaae06995ff647` |
| valid | `IOB2_valid.txt` | 11,167 | 129,223 | `29d584d1a0d9aca6d58b39eb8a93e154791d303ecff56ef93f12b3c889239bf1` |
| test | `IOB2_test.txt` | 11,307 | 129,824 | `359a1f61f6a21280c6c47b4fde9a5adcd07411d62015aa6e32d940c51f163b7a` |

Format: one `word LABEL` pair per line, a blank line between sentences, IOB2 labels over
25 entity types (ADAGE, ART, CARDINAL, CONTACT, DATE, DISEASE, EVENT, FACILITY, GPE, LANGUAGE,
LAW, LOCATION, MISCELLANEOUS, MONEY, NON_HUMAN, NORP, ORDINAL, ORGANISATION, PERCENTAGE,
PERSON, POSITION, PRODUCT, PROJECT, QUANTITY, TIME).

**How this project uses it**

| Use | Stage | Where |
|---|---|---|
| The label set (25 types) | A2 | `src/kazner_agents/labels.py`; a test checks it against the test split |
| Few-shot examples for the LLM prompt: 23 training sentences covering all 25 types | A2 | `src/kazner_agents/data/fewshot_kaznerd.jsonl`, selected by `scripts/select_fewshot.py` |
| Gold labels for measuring LLM quality (precision, recall, F1) | A4 | `kaznerd_test` source of the Collector |

The 23 few-shot sentences are the only KazNERD content stored in this repository; CC BY 4.0
allows this with attribution. Sentences with phone-number-like tokens were excluded.

**Getting the full dataset** (needed only to regenerate the few-shot file, and in A4): download
the three files from the pinned link above into `../ner-project/data/`, or use the `kazner`
toolkit (`python scripts/download_kaznerd.py`), which downloads the same commit and verifies the
checksums. The copy used for this project matches these checksums (after converting Windows
line endings to LF).

## 2. Kazakh Wikipedia

- **Site:** <https://kk.wikipedia.org>
- **API used:** MediaWiki Action API with TextExtracts, <https://kk.wikipedia.org/w/api.php>
  (`action=query&prop=extracts&explaintext=1`)
- **Licence of the text:** CC BY-SA 4.0 (<https://creativecommons.org/licenses/by-sa/4.0/>)

Articles are fetched at run time (`kna run --source wikipedia --title "<title>"`); nothing is
stored in the repository. Every exported sentence keeps the article URL and the licence in
`outputs/<run_id>/sentences.jsonl`, and files derived from Wikipedia text carry CC BY-SA 4.0.
Articles used in the documented runs: Абай Құнанбайұлы, Алматы, Назарбаев Университеті,
Желтоқсан көтерілісі (see `docs/real-runs.md`).
