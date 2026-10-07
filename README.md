# Cloudy ☁️

**Cloudy** is an independent Small Language Model (SLM) built and trained from scratch.

---

## 🎯 Architecture

Cloudy is a modern decoder-only Transformer tailored for efficiency and future edge deployment:
- **Embedding**: Token embedding (32,000 vocab).
- **Attention**: Causal Multi-Head Self-Attention with merged QKV linear projections.
- **Positional Encoding**: Rotary Position Embeddings (RoPE).
- **Normalization**: Pre-layer RMSNorm.
- **Feed-Forward**: Gated MLP / GeLU projections (`d_model` 128 → `d_ff` 512 → `d_model` 128).
- **Output Head**: Unbiased linear projection to 32,000 vocabulary.
- **Total Parameters (v1/Student)**: 8,585,856 (~8.59M parameters).

---

## 📁 Repository Structure

```text
Cloudy/
├── configs/
│   ├── cloudy_distill_pilot.json # Teacher distillation pilot config
│   ├── cloudy_pretrain_v1.json   # Baseline v1 configuration
│   └── cloudy_pretrain_v2.json   # Baseline v2 configuration
├── src/cloudy/
│   ├── __init__.py
│   ├── model.py                  # Decoder-only Transformer with RoPE & RMSNorm
│   ├── dataset.py                # Teacher distillation dataset & non-truncating chunking
│   └── train_distill.py          # Supervised distillation training loop & metrics logger
├── tests/
│   ├── test_model.py             # Shapes, backward, clipping, masking & save/load tests
│   ├── test_dataset.py           # Special tokens & non-truncating chunking tests
│   └── test_train_distill.py     # End-to-end training pilot verification
├── cloudy_colab_runner.ipynb     # Clean 3-cell Colab runner (No copy-pasting!)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧪 Testing Locally

Run all unit tests:
```bash
PYTHONPATH=src pytest tests/
```

All 13 unit tests cover:
1. Model parameter initialization and tensor shapes
2. Finite gradients and backpropagation
3. Gradient clipping enforcement
4. Weight updates under AdamW optimizer
5. Prompt token masking (`-100` label)
6. Checkpoint saving and loading verification
7. Autoregressive generation smoke test
8. Parity with saved checkpoint parameter shapes (8,585,856 parameters)
9. Special token IDs: `<s>` (0), `</s>` (1), `<pad>` (2), `<unk>` (3), `[INST]` (4), `[/INST]` (5), `[RESP]` (6), `[/RESP]` (7)
10. Teacher response non-truncating window chunking
11. Full end-to-end distillation training pilot with loss reduction and checkpoint generation

---

## 🚀 Running on Colab Tesla T4 (No Copy-Pasting!)

Open `cloudy_colab_runner.ipynb` in Colab, or run these 3 cells in your notebook:

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

```bash
# Cell 2: Clone or pull the latest code
!git clone https://github.com/revanth-kumar-02/Cloudy.git /content/Cloudy || (cd /content/Cloudy && git pull)
%cd /content/Cloudy
!pip install -r requirements.txt -q
```

```bash
# Cell 3: Launch Training with One Command
!PYTHONPATH=src python -m cloudy.train_distill --config configs/cloudy_distill_pilot.json
```

All new checkpoints and metrics are saved directly to:
`/content/drive/MyDrive/scratch_llm_1b/experiments/cloudy_distill_v1/pilot_run/`
Existing v1 assets are never overwritten or touched.
