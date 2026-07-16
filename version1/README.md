# Fine-Tuning Sesame CSM-1B for Hindi Text-to-Speech

This project fine-tunes **Sesame CSM-1B** for Hindi text-to-speech using **Unsloth**, **LoRA**, and the Hindi split of the **AI4Bharat IndicTTS** dataset.

The work focuses on building a reproducible speech fine-tuning pipeline, evaluating the base and adapted models on fixed Hindi prompts, and exploring whether speaker specialization improves intelligibility.

---

## 1. Project Objective

The project aims to:

- Download and preprocess Hindi speech data.
- Fine-tune `unsloth/csm-1b` using LoRA.
- Generate audio with the base and fine-tuned models.
- Compare pronunciation, fluency, naturalness, duration, and target-text alignment.
- Document implementation decisions, failures, fixes, and experimental results.
- Explore improvements beyond a basic fine-tuning run.

---

## 2. Model and Dataset

### Base model

- **Model:** [`unsloth/csm-1b`](https://huggingface.co/unsloth/csm-1b)
- **Architecture:** Sesame CSM-1B
- **Fine-tuning method:** LoRA
- **Training precision:** FP16
- **Target audio sampling rate:** 24 kHz

### Dataset

- **Dataset:** [`SPRINGLab/IndicTTS-Hindi`](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi)
- **Available fields:** `audio`, `text`, `gender`
- **Gender mapping:**
  - `0`: female
  - `1`: male
- **Original audio sampling rate:** typically 48 kHz
- **Training sampling rate:** 24 kHz

---

## 3. Evaluation Sentences

The same five sentences were used across the base model and fine-tuned checkpoints:

```text
नमस्ते, आपका ऑर्डर सफलतापूर्वक भेज दिया गया है।
आपका रिफंड अगले पाँच कार्य दिवसों में आपके खाते में जमा कर दिया जाएगा।
क्या मैं आपकी किसी और सहायता कर सकता हूँ?
धन्यवाद! आपका दिन शुभ हो।
कृपया अपना ऑर्डर नंबर बताइए।
```

Using fixed prompts and a consistent speaker-reference pair makes the comparisons more controlled.

---

## 4. Environment

The experiments were run on Kaggle with a Tesla T4 GPU.

### Tested versions

```text
Python: 3.12
PyTorch: 2.10.0+cu128
Transformers: 4.52.3
Datasets: 3.6.0
Unsloth: 2026.7.2
CUDA: available
GPU: Tesla T4
```

### Installation

```bash
pip install -U unsloth
pip install "transformers==4.52.3"
pip install --no-deps "trl==0.22.2"
pip install "datasets>=3.4.1,<4.0.0"
pip install torchcodec soundfile librosa jiwer evaluate tensorboard peft
```

Restart the runtime after installation.

### Hugging Face authentication

```python
from huggingface_hub import login

login(token="YOUR_HF_TOKEN")
```

Do not commit access tokens to the repository.

---

## 5. Repository Structure

```text
hindi-csm-tts/
├── adapters/
│   ├── smoke_test_adapter/
│   ├── hindi_csm_lora_100_steps/
│   ├── hindi_csm_lora_expanded_500_total_steps/
│   └── hindi_csm_lora_before_expanded_training/
├── checkpoints/
│   ├── pilot_300_total_steps/
│   └── expanded_400_steps/
├── female_specialization/
│   ├── adapters/
│   │   ├── female_smoke_20_steps/
│   │   └── female_specialized_1500/
│   ├── checkpoints/
│   │   └── female_1500_steps/
│   └── metadata.csv
├── fixed_csm_processor/
├── logs/
├── metrics/
├── outputs/
│   ├── base/
│   ├── finetuned/
│   ├── expanded/
│   ├── female_1500/
│   └── female_checkpoint_comparison/
├── notebooks/
├── inference.py
├── train.py
├── requirements.txt
└── README.md
```

Large processed datasets and some checkpoints may be excluded from GitHub because of storage limits. They can be reproduced from the notebook.

---

## 6. Data Preprocessing

### Text normalization

Hindi transcripts were normalized conservatively:

- Unicode NFC normalization
- Removal of invisible formatting characters
- Replacement of non-breaking spaces
- Collapsing repeated whitespace
- Collapsing repeated punctuation
- No transliteration
- No removal of Hindi matras
- No conversion to Romanized Hindi

Example:

```text
Before:
"  नमस्ते,   आपका ऑर्डर सफलतापूर्वक भेज दिया गया है।।  "

After:
"नमस्ते, आपका ऑर्डर सफलतापूर्वक भेज दिया गया है।"
```

### Audio filtering

The pipeline:

- Converts audio to mono
- Rejects empty arrays
- Rejects non-finite values
- Filters by duration
- Filters near-silent clips
- Resamples to 24 kHz
- Clips amplitudes to `[-1.0, 1.0]`
- Saves PCM-16 WAV files

### Audio-quality thresholds

```text
Minimum duration: 1.5 seconds
Maximum duration: 8.0 seconds
Minimum RMS: 0.001
Minimum peak amplitude: 0.005
Target sampling rate: 24,000 Hz
```

### Processed CSM fields

The CSM processor creates:

```text
input_ids
attention_mask
labels
input_values
input_values_cutoffs
```

Typical shapes:

```text
input_ids:            (256,)
attention_mask:       (256,)
labels:               (256,)
input_values:         (1, 192001)
input_values_cutoffs: (1,)
```

Label positions excluded from the loss are set to `-100`.

---

## 7. CSM Padding-Token Compatibility Fix

The installed Unsloth release did not automatically accept the CSM tokenizer padding configuration.

The CSM vocabulary already contains a dedicated padding token:

```text
Token: <|finetune_right_pad_id|>
Token ID: 128004
```

The EOS token remains:

```text
EOS token ID: 128001
```

A compatibility patch was applied before model loading so that Unsloth uses token ID `128004` without adding a new token or resizing the vocabulary.

A corrected processor was saved under:

```text
fixed_csm_processor/
```

This fix was necessary for reliable model loading in the tested Kaggle environment.

---

## 8. LoRA Configuration

```text
Rank: 32
Alpha: 32
Dropout: 0
Bias: none
Gradient checkpointing: enabled
Random seed: 3407
Trainable parameters: 29,032,448
Trainable percentage: 1.748%
```

### Target modules

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

The LoRA adapter was trained in FP16 using 8-bit AdamW.

---

# 9. Experiments

## Experiment 0: Base-model baseline

The base CSM model was evaluated before any Hindi fine-tuning.

### Control test

An English control sentence was fully understandable:

```text
Hello from Sesame. This is a test of the speech generation model.
```

This confirmed that:

- The model loaded correctly
- The processor worked
- The audio decoder worked
- The inference pipeline was valid

### Hindi baseline result

The five Hindi outputs were largely unintelligible.

Observed behavior:

- Speech-like output was generated
- Some samples sounded like an unknown foreign language
- The output did not match the target Hindi sentences
- Some generated files contained short speech followed by silence
- The base model showed weak Hindi pronunciation and text alignment

This formed the baseline for later comparison.

---

## Experiment 1: Pilot LoRA run

### Dataset

```text
Training samples: 216
Validation samples: 24
Gender balance:
- 108 female training clips
- 108 male training clips
- 12 female validation clips
- 12 male validation clips
```

### Configuration

```text
Smoke-test steps: 10
Additional steps: 90
Total reported steps: 100
Batch size: 1
Gradient accumulation: 4
Effective batch size: 4
Learning rate: 2e-4
Optimizer: adamw_8bit
Precision: FP16
```

### Smoke-test losses

```text
Step 1:  6.9068
Step 2:  7.1528
Step 3:  7.4191
Step 4:  7.1752
Step 5:  6.8518
Step 6:  6.6779
Step 7:  6.8849
Step 8:  6.6810
Step 9:  6.6755
Step 10: 6.5016
```

The smoke test confirmed that:

- LoRA parameters were trainable
- Loss remained finite
- The data collator was correct
- The model fit within T4 memory

### Pilot training results

```text
Training loss: 6.2187
Runtime: 2.93 minutes
Peak GPU memory: 4.84 GB
Best checkpoint: checkpoint-90
Best validation loss: 6.1417
```

Validation loss:

```text
Step 30: 6.2964
Step 60: 6.1760
Step 90: 6.1417
```

### Perceptual result

Compared with the base model:

- Speech became more continuous
- Some Hindi-like phonetics appeared
- Outputs were still mostly unintelligible
- The complete target sentences were not reliably spoken

---

## Experiment 2: Expanded mixed-speaker run

### Dataset

```text
Total selected samples: 1,200
Training samples: 1,080
Validation samples: 120
Female samples: 600
Male samples: 600
Training balance:
- 540 female
- 540 male
Validation balance:
- 60 female
- 60 male
```

### Configuration

```text
Previous pilot steps: 100
Additional steps: 400
Reported total steps: 500
Batch size: 1
Gradient accumulation: 4
Effective batch size: 4
Learning rate: 1e-4
Warmup steps: 20
Scheduler: cosine
Optimizer: adamw_8bit
Precision: FP16
```

### Results

```text
Training loss: 6.0928
Runtime: 11.5 minutes
Peak GPU memory: 4.72 GB
Best checkpoint: checkpoint-400
Best validation loss: 6.0912
```

Validation loss:

```text
Step 100: 6.2419
Step 200: 6.1455
Step 300: 6.0991
Step 400: 6.0912
```

### Perceptual result

The model improved numerically, but the difference between the 100-step and mixed-speaker 500-step outputs was difficult to identify by ear.

Observed behavior:

- Hindi-like rhythm remained present
- Speech was more continuous than the base model
- No clear perceptual improvement over the 100-step model
- Target sentences remained largely unintelligible

### Conclusion

Lower validation loss alone was not sufficient to guarantee an obvious improvement in human-perceived intelligibility.

---

## Experiment 3: Female-only specialization

The main hypothesis was that mixed-speaker training made the adaptation task harder. The inference reference used a female speaker, so a female-only continuation experiment was performed.

### Dataset

The filtering pipeline found:

```text
Valid female samples: 2,831
Training samples: 2,689
Validation samples: 142
```

All clips:

- Passed duration filtering
- Passed silence filtering
- Were resampled to 24 kHz
- Used speaker role `0`
- Used the same conservative transcript normalization

### Continuation smoke test

```text
Steps: 20
Training loss: 5.8522
Runtime: 0.59 minutes
Peak GPU memory: 8.71 GB
```

Loss remained finite, confirming that the restored adapter could continue training.

### Full female-specialization configuration

```text
Starting point: mixed-speaker 500-step adapter
Additional female-only steps: 1,500
Batch size: 1
Gradient accumulation: 4
Effective batch size: 4
Learning rate: 5e-5
Warmup steps: 50
Scheduler: cosine
Optimizer: adamw_8bit
Precision: FP16
Evaluation interval: 500 steps
```

### Female-specialization results

```text
Training loss: 5.6297
Runtime: 43.52 minutes
Peak GPU memory: 7.68 GB
Best checkpoint: checkpoint-1500
Best validation loss: 5.2246
```

Validation loss:

```text
Step 500:  5.7628
Step 1000: 5.3291
Step 1500: 5.2246
```

### Perceptual result

The female-only 1,500-step model showed a clear improvement over the mixed-speaker 500-step model.

Observed improvements:

- Several target Hindi words became recognizable
- Pronunciation was closer to the target text
- Speech continuity improved
- Target alignment improved
- Sample 3 sounded close to the intended sentence
- The voice remained more consistent
- Some samples were still only partially intelligible

### Final conclusion

> The female-only 1,500-step model showed a clear perceptual improvement over the mixed-speaker 500-step model. Several target Hindi words became recognizable, and Sample 3 sounded close to the intended sentence. Although speech was not fully intelligible across all samples, pronunciation, speech continuity, and alignment with the target text improved noticeably. This improvement was also reflected in the validation loss, which decreased from approximately 6.09 for the mixed-speaker model to 5.22 for the female-specialized model.

---

## 10. Results Summary

| Model | Training data | Total adaptation stage | Validation loss | Manual result |
|---|---:|---:|---:|---|
| Base CSM-1B | None | 0 | N/A | Hindi largely unintelligible |
| Pilot LoRA | 216 mixed clips | 100 steps | 6.1417 | Slightly more Hindi-like |
| Expanded mixed model | 1,080 mixed clips | 500 reported steps | 6.0912 | No clear perceptual gain over pilot |
| Female-specialized model | 2,689 female clips | +1,500 steps | 5.2246 | Clear improvement; recognizable Hindi words |

---

## 11. Inference Method

CSM is conditioned on a conversation containing:

1. A reference speaker turn with text and audio
2. A target speaker turn containing the requested text

Example:

```python
conversation = [
    {
        "role": "0",
        "content": [
            {
                "type": "text",
                "text": reference_text,
            },
            {
                "type": "audio",
                "path": reference_audio,
            },
        ],
    },
    {
        "role": "0",
        "content": [
            {
                "type": "text",
                "text": target_text,
            },
        ],
    },
]
```

Generation used deterministic decoding:

```python
generated = model.generate(
    **inputs,
    output_audio=True,
    min_new_tokens=min_new_tokens,
    max_new_tokens=max_new_tokens,
    do_sample=False,
    use_cache=True,
    pad_token_id=128004,
)
```

The same generation settings were used when comparing model stages.

---

## 12. Reproducibility

### Random seed

```text
3407
```

The same seed was used for:

- Dataset shuffling
- LoRA initialization
- Training
- Generation

### Saved artifacts

The project stores:

- Adapter weights
- Adapter configuration
- Trainer state
- Training logs
- Validation losses
- Audio manifests
- Manual evaluation notes
- Reference audio metadata
- Generated WAV files

### Recovery

The project was backed up as a recovery archive containing:

```text
adapters/
checkpoints/
fixed_csm_processor/
logs/
metrics/
outputs/
female_specialization/
```

---



## 13. Running Inference

A typical inference workflow is:

1. Install dependencies
2. Apply the CSM padding-token compatibility patch
3. Load the base model
4. Attach the LoRA adapter
5. Load the corrected processor
6. Load a verified text-audio reference pair
7. Generate and save WAV output

Example command:

```bash
python inference.py   --adapter female_specialization/adapters/female_specialized_1500   --text "क्या मैं आपकी किसी और सहायता कर सकता हूँ?"   --reference-audio outputs/female_checkpoint_comparison/comparison_reference_female.wav   --reference-text-file outputs/female_checkpoint_comparison/comparison_reference.json   --output generated.wav
```

---

## 14. Generated Audio

Generated files are organized by experiment:

```text
outputs/base/
outputs/finetuned/
outputs/expanded/
outputs/female_1500/
outputs/female_checkpoint_comparison/
```

Each folder should include:

- WAV files
- A manifest CSV
- Target text
- Model/checkpoint identifier
- Duration
- Generation settings
- Status

---


## 15. References

- [Sesame CSM-1B](https://huggingface.co/unsloth/csm-1b)
- [SPRINGLab IndicTTS Hindi](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi)
- [Official Unsloth CSM Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Sesame_CSM_%281B%29-TTS.ipynb)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/)
- [Hugging Face Transformers CSM Documentation](https://huggingface.co/docs/transformers/model_doc/csm)

---

## License

This project uses open-source model and dataset resources. Refer to the original model and dataset repositories for their respective licenses and usage conditions.
