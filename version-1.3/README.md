# Version 1.3 — Inference Optimization and ASR Evaluation

Version 1.3 extends the Hindi CSM-1B LoRA experiment with inference
parameter analysis, automatic Hindi intelligibility scoring, and
word-level failure analysis.

## Main additions

- Inference sweep across checkpoints 2500, 2750, and 3000
- Greedy, focused, and moderate decoding configurations
- Three random seeds per configuration
- 135 generated audio samples
- Hindi ASR transcription using Whisper Large V3 Turbo
- Word Error Rate and Character Error Rate evaluation
- Word-level substitution and deletion analysis
- Direct audio comparison between three model-selection methods

## Final selected model

- Checkpoint: 2750
- Decoding: greedy
- Seed: 5107

Manual listening selected this configuration as the best overall system.
Across the five evaluation sentences, the speech was almost completely
understandable, with only one or two mispronounced words in most samples.

## Comparison of selection methods

| Selection method | Selected system |
|---|---|
| Lowest validation loss | Checkpoint 3000, greedy, seed 3407 |
| Lowest ASR WER/CER | Checkpoint 2500, focused, seed 4107 |
| Manual perceptual evaluation | Checkpoint 2750, greedy, seed 5107 |

The three methods selected different systems. This demonstrates that
validation loss, ASR-derived intelligibility, and human perceptual quality
measure different aspects of TTS performance.

## ASR evaluation

The automatic ASR system selected checkpoint 2500 with focused decoding
and seed 4107. However, the ASR metrics did not perfectly match human
listening.

Whisper made transcription errors on some speech that remained
understandable to human listeners. Therefore, WER and CER are presented as
supplementary proxy metrics rather than absolute measures of TTS quality.

## Audio folders

- `audio/best_model/` — final five selected outputs
- `audio/selection_comparison/lowest_validation_loss/`
- `audio/selection_comparison/lowest_asr_error/`
- `audio/selection_comparison/manual_winner/`

## Metrics

The `metrics/` folder contains:

- ASR transcriptions
- WER and CER scores
- Ranked inference configurations
- Word-level alignments
- Mispronunciation summaries
- Manual versus automatic selection comparison

## Key conclusion

Checkpoint 3000 achieved the lowest validation loss, checkpoint 2500
achieved the lowest ASR-derived error, and checkpoint 2750 produced the
best perceptual speech quality.

The final system was therefore selected using manual listening supported
by automated diagnostic metrics.