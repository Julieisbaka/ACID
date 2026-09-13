# ACID questions

Add benchmark items as `.json` files in this folder. The harness loads every JSON file and expects each item to contain:

- `id`: unique item identifier
- `domain`: one of `ACID-Math`, `ACID-Chem`, `ACID-Bio`, or `ACID-Code`
- `question`: the only item field sent to the benchmarked model
- `answer`: reference answer sent only to the scoring model

Optional `metadata` is retained in the source file but is not sent to either model. The scoring model receives the complete canonical item (`question`, `domain`, `answer`) plus the benchmark model's response, and never receives the noise block.
