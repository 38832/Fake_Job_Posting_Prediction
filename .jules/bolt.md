## 2024-05-22 - [Text Truncation Optimization]
**Learning:** For models using `pad_sequences` with default `truncating='pre'`, processing the entire input text is wasteful if the input is significantly larger than `MAX_SEQUENCE_LENGTH`. Truncating the input text *before* tokenization (with a safety buffer) yields significant performance gains for large inputs without affecting model input.
**Action:** When working with NLP models, always check how input sequence length is handled. If only a part of the text is used, discard the rest as early as possible in the pipeline.
