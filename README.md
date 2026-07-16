

---

## Version 1.2.0 — Female specialization through 3,000 steps

Version 1.2 extends the female-only specialization experiment from
1,500 to 3,000 steps.

### Checkpoints evaluated

- 1,500
- 1,750
- 2,000
- 2,250
- 2,500
- 2,750
- 3,000

### Main result

Recognizable complete Hindi target words began appearing at checkpoint
1,750. Pronunciation and target alignment improved progressively through
later checkpoints.

Checkpoint 3,000 performed best overall across the five evaluation
sentences. It generated multiple correct target words and occasionally
produced near-correct sentence fragments. Some short sections still
contained fumbling or mispronunciations.

Checkpoints 2,500 and 2,750 occasionally sounded smoother than checkpoint
3,000 for individual samples, showing that perceptual quality remained
sample-dependent.

### Validation-loss trend

Validation loss continued decreasing through checkpoint 3,000, although
the improvements became smaller near the end of training.

### Included in v1.2

- Reproducible final Kaggle notebook
- Base-model outputs
- Mixed-speaker 500-step outputs
- Female-only outputs from 1,500 through 3,000 steps
- Training metrics and validation-loss data
- Manual listening comparison
