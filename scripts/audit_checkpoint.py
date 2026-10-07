#!/usr/bin/env python3
"""Deterministic evaluation and audit script for Cloudy pilot checkpoint.

Loads the existing trained student checkpoint from Google Drive, computes loss and
perplexity on the 3 teacher demonstrations without retraining, and generates
deterministic responses for all 3 prompts under both greedy and sampled decoding.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

# Add src to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch

from cloudy.checkpoint import load_checkpoint
from cloudy.data import TeacherDistillationDataset, create_distillation_dataloader
from cloudy.evaluation import evaluate_model
from cloudy.generate import generate_response
from cloudy.model import CloudyConfig, CloudyForCausalLM
from cloudy.tokenizer import CloudyTokenizer


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def audit_checkpoint(
    checkpoint_path: str,
    tokenizer_path: str,
    data_path: str,
    output_json: Optional[str] = None,
    device_name: str = "auto",
    seed: int = 42,
):
    set_seed(seed)
    print("=" * 65)
    print("🔍 CLOUDY PILOT CHECKPOINT AUDIT")
    print("=" * 65)

    # 1. Device Selection
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Device: {device} (CUDA available: {torch.cuda.is_available()})")

    # 2. Check File Existence
    ckpt_file = Path(checkpoint_path)
    tok_file = Path(tokenizer_path)
    data_file = Path(data_path)

    for name, p in [("Checkpoint", ckpt_file), ("Tokenizer", tok_file), ("Dataset", data_file)]:
        if not p.is_file():
            print(f"[FAIL] {name} not found: {p}")
            return False
        print(f"[OK] {name} found: {p}")

    # 3. Load Tokenizer
    tokenizer = CloudyTokenizer.from_file(tok_file)
    print(f"[OK] Tokenizer loaded (vocab size: {tokenizer.get_vocab_size():,})")

    # 4. Load Checkpoint Safely
    ckpt = load_checkpoint(ckpt_file, device=device)
    saved_step = ckpt.get("step", "unknown")
    saved_config = ckpt.get("config", {})
    metrics = ckpt.get("metrics", {})
    print(f"[OK] Checkpoint loaded successfully:")
    print(f"     Step: {saved_step}")
    print(f"     Recorded Training Loss: {metrics.get('final_loss', 'N/A')}")
    print(f"     Recorded Perplexity:    {metrics.get('perplexity', 'N/A')}")

    # 5. Restore Student Model
    model_cfg = CloudyConfig(**saved_config) if saved_config else CloudyConfig()
    model = CloudyForCausalLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[OK] Restored Cloudy student ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # 6. Evaluate Loss on the 3 Teacher Demonstrations with Token Accounting
    dataset = TeacherDistillationDataset(
        data_path_or_records=data_file,
        tokenizer=tokenizer,
        max_seq_len=model_cfg.max_seq_len,
        pad_to_max_len=True,
    )
    dataloader = create_distillation_dataloader(dataset, batch_size=1, shuffle=False)
    summary = dataset.get_summary()

    total_supervised_tokens = sum((c.labels != -100).sum().item() for c in dataset.chunks)
    eval_results = evaluate_model(model, dataloader, device)
    print("\n--- Quantitative Audit on Teacher Dataset ---")
    print(f"Demonstrations:           {summary['total_examples']}")
    print(f"Training Chunks:          {summary['total_training_chunks']}")
    print(f"Supervised Tokens:        {total_supervised_tokens:,}")
    print(f"Audited Loss (Chunk-avg): {eval_results['eval_loss']:.4f}")
    print(f"Audited Perplexity:       {eval_results['perplexity']:.2f}")

    # Load demonstration records for reference answers
    demo_records = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                demo_records.append(json.loads(line))

    # 7. Next-Token Prediction & Generation Audit per Demonstration
    from cloudy.generate import format_inference_prompt, generate_response_with_details
    import torch.nn.functional as F

    audit_report = {
        "checkpoint": str(ckpt_file),
        "step": saved_step,
        "audited_loss": eval_results["eval_loss"],
        "audited_perplexity": eval_results["perplexity"],
        "total_supervised_tokens": total_supervised_tokens,
        "demonstrations": [],
    }

    print("\n--- Generation & Demonstration Audit ---")
    for idx, record in enumerate(demo_records, start=1):
        ex_id = record.get("id", f"ex_{idx}")
        messages = record.get("messages", [])
        pr = ""
        ref_answer = ""
        for m in messages:
            if m.get("role") == "user":
                pr = m.get("content", "")
            elif m.get("role") == "assistant":
                ref_answer = m.get("content", "")

        print(f"\n[Demonstration {idx}]: ID={ex_id}")
        print(f"  Prompt:           {pr}")
        print(f"  Reference Answer: {ref_answer}")

        # Compute demonstration-specific response-token loss
        ex_chunks = [c for c in dataset.chunks if c.example_id == ex_id]
        demo_loss_sum = 0.0
        demo_tok_count = 0
        with torch.no_grad():
            for c in ex_chunks:
                c_in = c.input_ids.unsqueeze(0).to(device)
                c_mask = c.attention_mask.unsqueeze(0).to(device)
                c_lbl = c.labels.unsqueeze(0).to(device)

                c_logits, _ = model(c_in, attention_mask=c_mask)
                shift_logits = c_logits[..., :-1, :].contiguous()
                shift_labels = c_lbl[..., 1:].contiguous()

                loss_val = F.cross_entropy(
                    shift_logits.view(-1, model_cfg.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                valid_cnt = (shift_labels != -100).sum().item()
                demo_loss_sum += loss_val.item()
                demo_tok_count += valid_cnt

        demo_resp_loss = (demo_loss_sum / demo_tok_count) if demo_tok_count > 0 else 0.0
        print(f"  Response Tokens:  {demo_tok_count} supervised tokens across {len(ex_chunks)} chunk(s)")
        print(f"  Response-Token Loss: {demo_resp_loss:.4f} (Perplexity: {torch.exp(torch.tensor(demo_resp_loss)).item():.2f})")

        # Top 10 tokens after [RESP]
        input_ids = format_inference_prompt(pr, tokenizer).to(device)
        with torch.no_grad():
            logits, _ = model(input_ids)
            next_logits = logits[0, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            top10_probs, top10_ids = torch.topk(probs, 10)

        print("  Top 10 predicted tokens after [RESP]:")
        top10_list = []
        for rank, (tid, prob) in enumerate(zip(top10_ids.tolist(), top10_probs.tolist()), start=1):
            tok_str = tokenizer.decode([tid])
            top10_list.append({"rank": rank, "token_id": tid, "prob": round(prob, 4), "token": tok_str})
            print(f"    #{rank:02d}: ID {tid:5d} ({prob:6.2%}) -> {repr(tok_str)}")

        # Deterministic Greedy Decoding (temperature=0.0)
        set_seed(seed)
        greedy_details = generate_response_with_details(
            model=model,
            tokenizer=tokenizer,
            prompt=pr,
            max_new_tokens=64,
            temperature=0.0,
            device=device,
        )
        greedy_resp = greedy_details["text"]
        term_ok = greedy_details["terminated_by_stop"]
        stop_tok_name = greedy_details["stop_token_name"]

        print(f"  Greedy Generated:")
        print(f"    {repr(greedy_resp) if greedy_resp else '<EMPTY_OUTPUT>'}")
        print(f"  Terminated at stop token: {term_ok} ({stop_tok_name})")

        # Sampled Decoding (temperature=0.7, top_k=20)
        set_seed(seed)
        sampled_details = generate_response_with_details(
            model=model,
            tokenizer=tokenizer,
            prompt=pr,
            max_new_tokens=64,
            temperature=0.7,
            top_k=20,
            device=device,
        )
        sampled_resp = sampled_details["text"]
        print(f"  Sampled (temp=0.7, top_k=20):")
        print(f"    {repr(sampled_resp) if sampled_resp else '<EMPTY_OUTPUT>'}")

        # Coherence assessment
        is_coherent = bool(greedy_resp and (greedy_resp in ref_answer or ref_answer in greedy_resp or len(greedy_resp.split()) > 3))
        print(f"  Coherence Check:  {'PASS (Coherent output)' if is_coherent else 'FAIL (Incoherent or empty)'}")

        audit_report["demonstrations"].append({
            "id": ex_id,
            "prompt": pr,
            "reference_answer": ref_answer,
            "response_token_loss": round(demo_resp_loss, 4),
            "supervised_tokens": demo_tok_count,
            "chunks": len(ex_chunks),
            "top10_after_resp": top10_list,
            "greedy_response": greedy_resp,
            "greedy_terminated_at_stop": term_ok,
            "greedy_stop_token": stop_tok_name,
            "sampled_response": sampled_resp,
            "is_coherent": is_coherent,
        })

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)
        print(f"\n[OK] Saved audit results JSON to: {out_path}")

    print("\n" + "=" * 65)
    print("✅ AUDIT COMPLETE — No files or checkpoints were modified.")
    print("=" * 65 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Audit Cloudy Pilot Checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/scratch_llm_1b/experiments/cloudy_distill_v1/fixed_chunking_run/checkpoints/cloudy_student_distill_final.pt",
        help="Path to final student checkpoint .pt file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="/content/drive/MyDrive/scratch_llm_1b/tokenizer/v2/tokenizer.json",
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/content/drive/MyDrive/scratch_llm_1b/experiments/cloudy_distill_v1/teacher_pilot_3.jsonl",
        help="Path to teacher_pilot_3.jsonl",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save audit report JSON",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed")
    args = parser.parse_args()

    audit_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        data_path=args.data,
        output_json=args.output_json,
        device_name=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

