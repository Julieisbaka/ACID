# ACID questions

Add benchmark items as `.json` files in this folder. The harness loads every JSON file and expects each item to contain:

- `id`: unique item identifier
- `domain`: one of `ACID-Math`, `ACID-Chem`, `ACID-Bio`, or `ACID-Code`
- `question`: the only item field sent to the benchmarked model
- `answer`: reference answer sent only to the scoring model

The scoring model receives the canonical `question`, `domain`, and `answer` plus the benchmark model's response, and never receives the noise block.
Difficulty is intentionally not a question field yet. It is reserved for a future version that may analyze degradation by difficulty, so current runs cannot accidentally use it.
