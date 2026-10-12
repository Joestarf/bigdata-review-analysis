# Portable review scoring bundle

`src/portable.py` provides `PortablePredictor` and a CSV command line for scoring pairs with the frozen A–E methods. It loads only the three checked joblib artifacts listed in `models/portable_manifest.json`; it does not open the original review corpus, embedding cache, annotations, or training/evaluation locks. The packaged source modules supply the established text normalization, pair features, pair IDs, and pinned MiniLM encoder implementation.

## Bundle contents

The bundle root must contain:

- `src/` with the project Python modules, including `portable.py`
- `models/portable_manifest.json`
- `models/lexical.joblib`, `models/C.joblib`, and `models/E.joblib`
- `requirements-inference.txt` for inference deployment (preferred), or `requirements.txt` for the original full environment

The manifest must provide `model_revision` (the fixed MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`), `dataset_version`, `model_lock_sha256`, `feature_order`, `methods`, and `files_sha256`. The required feature order is `char_cosine`, `word_jaccard`, `length_ratio`, `length_difference`, `exact_match`, `semantic_cosine`. `methods` has A–E entries with `review_threshold` and boolean `qualified`; these fields document the frozen review-threshold configuration and qualification state. The output scores remain uncalibrated and all decisions are `review`, regardless of threshold or qualification.

The SHA-256 mapping must cover exactly `models/lexical.joblib`, `models/C.joblib`, and `models/E.joblib`. The predictor checks these hashes before loading the files. `model_lock_sha256` records the provenance of the frozen source lock; it does not cause any original lock or dataset to be loaded.

## Dependencies and model download

For deployment, install the minimal, observed-version dependencies in `requirements-inference.txt`; the packaged full `requirements.txt` is for recreating the original environment. Import and run from the bundle root so the `src` package resolves. The first scoring call downloads or loads the pinned `sentence-transformers/all-MiniLM-L6-v2` revision through the normal Hugging Face cache. Network access is needed on first use unless that exact revision is already cached. Encoder initialization is lazy; constructing `PortablePredictor` does not download the encoder.

Joblib artifacts are pickle files and can execute code while loading. Only use a bundle obtained from a trusted source. The manifest hashes detect changes relative to the supplied manifest but do not establish who created that manifest.

## Python use

```python
import pandas as pd
from src.portable import PortablePredictor

predictor = PortablePredictor(".")
pairs = predictor.predict_pairs([
    ("Battery lasts all day.", "The battery lasts a full day."),
])

reviews = pd.DataFrame([
    {"asin": "EXAMPLE", "review_id": "r1", "text": "Battery lasts all day."},
    {"asin": "EXAMPLE", "review_id": "r2", "text": "The battery lasts a full day."},
])
product_scores = predictor.score_product(reviews)
```

`predict_pairs` accepts at most 1,000 input pairs per call. `score_product` accepts at most 1,000 comments for one product and requires nonempty string ASINs, unique nonempty string review IDs, and nonempty text. Each method returns one row per pair; five method rows are emitted per pair. Generated inputs need no membership in the original 90,000-row dataset and no original review IDs. Reasons flag comments with fewer than five whitespace-separated words and encoder inputs longer than 256 tokens.

## CSV use

Prepare a UTF-8 CSV with columns `asin,review_id,text`, then run from the bundle root:

```sh
python -m src.portable --bundle . --input reviews.csv --output pair_scores.csv
```

The command checks required columns, nonempty fields, and globally unique `review_id` values across products. It emits every within-ASIN pair and enforces the 1,000-comment limit per product. An input file containing only the required header is accepted and produces a header-only output CSV. Keep products reasonably sized because pair count grows quadratically.

The output contains `pair_id,asin,review_id_a,review_id_b,method,score,decision,reason`. Scores use the frozen A exact-match feature, B character TF-IDF cosine, C lexical classifier, D MiniLM cosine, and E combined classifier. Feature and model scoring semantics are unchanged.

## Interpretation boundary

These scores are uncalibrated outputs for triage and independent review. They are not validated population accuracy, confirmed redundancy labels, or deployment approval. Every decision stays `review`, and the portable code does not retrain models. A downstream study may use the scores in an explicitly declared experimental weighting policy, with its assumptions and evaluation documented; this package does not enable validated automatic deployment or production reweighting.

## Dependencies and limitations

The module imports `src.common` for pair IDs, `src.features` for frozen feature construction, and `src.embedding` for the fixed-revision encoder and embedding helper. Those modules are code dependencies only: portable inference does not call study loading, lexical fitting, embedding-cache loading, inference validation, or model-lock validation. The vectorizer and classifiers are loaded only after their manifest checks pass. The portable manifest itself is not signed, so trusted distribution remains essential.


## Local handoff verification

The portable archive was extracted into a directory without the original review dataset. Six input pairs and a ten-review product produced the expected 30 and 225 method rows. Scores matched the original frozen interface within an absolute difference of 3.58e-7. All 6,000 product/scenario rows were independently recalculated from the included review-weight data with matching ratings and ranks. This checks implementation and packaging consistency, not detection accuracy on unseen natural data.

The separate full research archive also loaded the original locked interface and its 90,750 reviews in the recorded environment. That original interface checks exact dependency/platform versions; use the portable route on other compatible machines.
