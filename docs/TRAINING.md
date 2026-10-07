# Training Cloudy on Google Colab (Tesla T4)

This guide documents the exact steps to launch the **Cloudy Teacher–Student Distillation Pilot** on a Google Colab Tesla T4 GPU without manual code copying.

---

## 📋 Architecture & Experiment Summary

- **Model**: Cloudy Decoder-Only Transformer (Fresh random student weights).
- **Parameters**: 8,585,856 (~8.59M).
- **Vocabulary**: 32,000 Byte-Level BPE tokens.
- **Context Length**: 128 tokens (safe continuation chunking prevents any truncation of teacher answers).
- **Objective**: Supervised teacher–student demonstration prediction with `-100` prompt masking.
- **Dataset**: `experiments/cloudy_distill_v1/teacher_pilot_3.jsonl` (3 teacher demonstrations from `openai/gpt-oss-20b`).
- **Target Output**: `/content/drive/MyDrive/scratch_llm_1b/experiments/cloudy_distill_v1/pilot_run/`
- **Asset Preservation**: Existing v1 assets and checkpoints are never modified or overwritten.

---

## 🚀 One-Command Launch Steps in Colab

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Pull Latest Repository & Install Dependencies
```bash
!git clone https://github.com/revanth-kumar-02/Cloudy.git /content/Cloudy || (cd /content/Cloudy && git pull)
%cd /content/Cloudy
!pip install -r requirements.txt -q
```

### Cell 3: Launch Training with Preflight
```bash
!python scripts/train_colab.py --config configs/cloudy_distill_v1.json
```

---

## 🔍 Verification Checklist

During execution, verify:
1. **GPU Detected**: PyTorch detects NVIDIA Tesla T4 (~14.7 GiB VRAM).
2. **Preflight Passes**: Confirms `teacher_pilot_3.jsonl` and `tokenizer/v2/tokenizer.json` exist on Drive.
3. **Loss Decreases**: Initial cross-entropy loss begins around ~10.4 and steadily decreases with each epoch.
4. **Finite Gradients**: Every step logs finite loss and finite gradient norms (clipped at 1.0).
5. **Checkpoint Saved**:
   - `checkpoints/cloudy_student_distill_final.pt` is saved atomically in Google Drive.
6. **Generation Tested**: Sample text generation runs automatically on test prompts post-training.
7. **Metrics Logged**: Full step history and summary metrics are written to `logs/distill_run_metrics.json`.
