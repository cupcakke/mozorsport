---
name: JAIDE naming constraint
description: Hard ban on standard deep-learning jargon anywhere in the JAIDE codebase
---

The user has stated an absolute, permanent rule for this project: the following terms (and obvious variants) must never appear anywhere in the codebase — not in code, identifiers, comments, docs, log strings, or generated artifacts:

perceptron, attention / multihead / q_proj, softmax, convolution / CNN, RNN / LSTM / GRU, Transformer, torch / pytorch / tensorflow / jax / onnx, numpy, sigmoid / tanh / relu / gelu / swish, cross_entropy / kl_div / nll_loss, layer_norm / batch_norm / dropout, beam_search / top_k / greedy_decode, positional_encoding, feedforward / FFN, backprop.

**Why:** JAIDE is deliberately built as a custom Zig/Futhark architecture (RSF/OFTB-based) and the user does not want it to read or resemble a standard ML-framework/transformer codebase, even in naming.

**How to apply:** Before writing or editing any file in this project (code, comments, docs, scripts, log message strings), scan new content against this list. If a concept overlaps with one of these (e.g. an activation function, a normalization step, a decoding strategy), name it with project-specific/non-jargon terminology instead. Also grep the diff before finalizing any large generated file (e.g. Futhark kernels, bench scripts) since these terms can slip in via boilerplate or generated code.
