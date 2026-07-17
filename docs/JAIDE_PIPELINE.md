# JAIDE production checkpoint, evaluation, export, inference and benchmarking pipeline

## Checkpoint format

The current trainer checkpoint format is the implementation in `DistributedTrainerFuthark.saveCheckpoint`; the current version is `9` and the magic is `JAIDECKP`.

The binary layout is little-endian:

1. 8-byte magic `JAIDECKP`.
2. `u32` checkpoint version, currently `9`.
3. `u64` global training step.
4. `u64` full model dimension.
5. `u64` RSF layer count.
6. `u64` tokenizer vocabulary size.
7. `u64` stored local batch size.
8. `f32` learning rate stored as raw IEEE-754 bits.
9. `f32` momentum stored as raw IEEE-754 bits.
10. For each RSF layer, four tensors, each encoded as `u64 element_count` followed by `element_count` finite `f32` values that are checked as representable `f16` when loaded: `weights_s`, `weights_t`, `velocity_s`, `velocity_t`. Each tensor has shape `[model_dim / 2, model_dim / 2 + 1]` and is resident in the accelerator as `f16`.
11. `f32` clip minimum and `f32` clip maximum.
12. `u8` embedding-present flag. Only `0` and `1` are valid.
13. If embedding is present: `u64 embedding_vocab_size`, `u64 embedding_dim`, `u64 embedding_weight_len`, `embedding_weight_len` finite `f32` values, `u64 embedding_velocity_len`, and `embedding_velocity_len` finite `f32` values. Weight, gradient, and velocity tensors have shape `[vocabulary_size, model_dim]`; gradients are not serialized in checkpoints and are reset on load.
14. `u32` NSIR node count. Each node stores `u32 id_len`, id bytes, `u32 data_len`, data bytes, five finite `f64` values: qubit `a.re`, `a.im`, `b.re`, `b.im`, and `phase`.
15. `u32` NSIR edge-key count. Each edge-key group stores source and target string lengths and bytes, then `u32 edge_count`. Each edge stores finite `f64 weight`, `u8 EdgeQuality`, finite `f64 quantum_correlation.re`, finite `f64 quantum_correlation.im`, and finite `f64 fractal_dimension`. Valid `EdgeQuality` values are `0..4` for superposition, entangled, coherent, collapsed, and fractal.
16. `u64` tokenizer payload length and the exact byte payload produced by `MGT.saveVocab`.
17. `u32` trailer `0xDEADBEEF`.
18. End of file. Trailing data is invalid.

`loadCheckpoint` checks magic, version, model dimension, layer count, batch size nonzero, learning rate, momentum, every RSF tensor length and finite value, clip range, embedding flag, embedding dimensions and finite values, NSIR string lengths, NSIR edge count limits, edge-quality enum values, finite graph numeric values, tokenizer length, trailer, and trailing bytes. It restores RSF weights and velocities, embedding weights and velocities, tokenizer state, global step, learning rate, momentum, local batch size, NSIR graph, and graph-dependent signal-engine binding.

## Training objective and masking

Training tokenizes each JSONL `text` value with the checkpoint tokenizer, truncates each sample to at most `max_sequence_length + 1` tokens, forms input tokens `tokens[0..n-1]`, and predicts the next-token embedding vectors from `tokens[1..n]`. The Futhark loss is mean squared L2 embedding error over the allocated `[active_batch, max_prediction_length, model_dim]` tensor. The upgraded evaluator reports the same objective as `embedding_l2_loss`; it is not perplexity. Conventional perplexity requires a normalized probability distribution and cross-entropy objective, which this implementation does not use.

Evaluation excludes empty samples, samples with fewer than two tokens, padding tokens, and positions beyond truncation from all denominators. Aggregate metrics are weighted by the actual count of valid next-token targets.

## Evaluation

`jaide-eval` loads every checkpoint independently, uses the tokenizer embedded in that checkpoint, runs a forward-only RSF/Futhark path, and never calls backward, optimizer updates, embedding-gradient accumulation, relational training passes, learning-rate updates, global-step increments, tokenizer mutation, or NSIR mutation.

Example:

```bash
jaide-eval \
  --checkpoint /checkpoints/epoch_001/model.ckpt \
  --checkpoint /checkpoints/epoch_002/model.ckpt \
  --dataset /data/dataset/validation.jsonl \
  --batch-size 32 \
  --max-sequence-length 512 \
  --reproducibility-runs 2 \
  --output /reports/checkpoint-comparison.json
```

The report is JSON schema version `1` and includes checkpoint metadata, SHA-256 digest, dataset digest, sample counts, valid target-token counts, `embedding_l2_loss`, exact top-1/top-5/top-10 retrieval accuracy, mean reciprocal rank, mean rank, collapse diagnostics, performance timing, memory accounting, reproducibility hashes, warnings, errors, and selected best checkpoint. Selection defaults to the lowest finite validation `embedding_l2_loss`; other accepted criteria are top-1, top-5, top-10, and MRR. Ties are resolved by metric, then global step, then path.

Retrieval ranking uses cosine similarity between the predicted vector and the trained checkpoint embedding table. Zero-norm vectors receive negative-infinity similarity. Ties are deterministic by ascending token ID. Ranking is chunked by `--ranking-chunk-size` and never allocates a full predictions-by-vocabulary matrix.

## Collapse diagnostics

Evaluation reports mean, standard deviation, minimum and maximum embedding-vector norm, zero or near-zero vector count, per-dimension variance summary, deterministic sampled-pair cosine mean and standard deviation, effective rank from variance entropy, near-duplicate pair rate, predicted-token entropy, most-frequent predicted-token percentage, and unique predicted-token count. Sampling is deterministic from `--seed` and bounded.

## Dataset splitting

```bash
jaide-dataset-split \
  --input /data/dataset/all.jsonl \
  --train-output /data/dataset/train.jsonl \
  --validation-output /data/dataset/validation.jsonl \
  --validation-ratio 0.05 \
  --seed 42
```

The splitter strictly parses JSONL, requires a nonempty `text` string, preserves full records, groups by `--group-field` when supplied, otherwise by common document identifier fields or deterministic content hash fallback, shuffles groups deterministically from the seed, writes atomically, refuses to overwrite without `--overwrite`, and fails on duplicate text across train and validation by default. Use `--duplicate-policy deduplicate` to skip duplicated cross-split records explicitly.

## Resume training

```bash
jaide-distributed-futhark \
  --resume /checkpoints/epoch_003/model.ckpt \
  --dataset /data/dataset/train.jsonl
```

Resume mode reads checkpoint metadata before trainer construction, uses checkpoint-compatible structural dimensions, extracts the embedded tokenizer, initializes compatible components, calls `loadCheckpoint`, restores training state, skips pre-training knowledge-graph mutation, and starts saving at the next available `/checkpoints/epoch_xxx/model.ckpt` directory so existing epoch snapshots are not overwritten.

Epoch checkpoint files are complete independent snapshots, not shards. Hugging Face-style shard files are pieces of one checkpoint; JAIDE `epoch_xxx/model.ckpt` files are separate full training-state snapshots.

## Export and inference serving

```bash
jaide-export \
  --checkpoint /checkpoints/epoch_003/model.ckpt \
  --output /models/jaide.model
```

The exported inference model contains RSF weights, trained embedding weights, tokenizer payload, model dimensions, clip parameters, format version, and checksum. It excludes RSF velocities, embedding velocities, gradients, optimizer buffers, and training counters. The supported export precision is lossless `f32` for trained `f32` embeddings. Export validates the model by immediately loading it.

```bash
jaide-inference-server --model /models/jaide.model
```

`JAIDE_MODEL_PATH` remains supported. The inference server accepts a model only after tokenizer, RSF weights, trained embeddings, dimensions, vocabulary size, and checksum are valid. It no longer initializes a random `LearnedEmbedding` when trained embeddings are available. Inference uses a stateless neural model plus fresh request-local relational structures; exported NSIR graph state is not required by the neural generation mode.

## Checkpoint inspection

```bash
jaide-checkpoint-inspect --checkpoint /checkpoints/epoch_003/model.ckpt
```

Inspection parses and validates headers, dimensions, lengths, enum values, trailer, and trailing data and returns metadata without constructing a trainer.

## Benchmarking and memory

`jaide-eval` separates model load time, tokenizer time, forward time, ranking time, complete batch latency, tokens per second, and samples per second. Accelerator synchronization is performed around forward timing. Reports include median, mean, minimum, maximum, standard deviation, p50, p90, p95, and p99 batch latency when samples are available.

`jaide-benchmark` runs dedicated trained-weight benchmarks from a real checkpoint and dataset:

```bash
jaide-benchmark \
  --checkpoint /checkpoints/epoch_003/model.ckpt \
  --dataset /data/dataset/validation.jsonl \
  --prompts /data/evaluation/prompts.jsonl \
  --batch-sizes 1,8,32 \
  --sequence-lengths 32,128,512 \
  --generation-lengths 16,64 \
  --warmup-iterations 3 \
  --measured-iterations 10 \
  --output /reports/epoch_003.benchmark.json
```

It loads the checkpoint tokenizer, RSF weights, and trained embeddings, measures model load time, tokenizer time, synchronized Futhark forward-pass time, exact chunked retrieval-ranking time, complete batch latency, samples per second, valid tokens per second, first-token latency, generated-token latency, generated-output loop and unknown-token diagnostics, and memory accounting for serialized checkpoint bytes, host model state, activations, and ranking buffers. It never uses random replacement weights or generated synthetic samples.

Memory metrics distinguish serialized checkpoint size, checkpoint payload bytes, host-resident model state, RSF weight bytes, RSF velocity bytes, embedding weight bytes, embedding velocity bytes, embedding gradient bytes, temporary activation bytes, temporary ranking-buffer bytes, and tracked device allocation fields when available. External GPU process memory should be measured separately with tools such as `nvidia-smi` on NVIDIA systems; unavailable device data is reported as `null` rather than fabricated.

## Overfitting

Compare training loss and validation `embedding_l2_loss` across epoch checkpoints. Overfitting is indicated when training loss continues decreasing while validation embedding loss worsens or retrieval metrics degrade on unseen validation data.
