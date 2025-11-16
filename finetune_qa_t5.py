# -*- coding: utf-8 -*-
import os, json, csv, argparse, random, hashlib
from typing import List, Dict, Tuple
from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# ---------- Utils ----------
def read_jsonl(path: str) -> List[Dict]:
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                r = json.loads(line)
                q = (r.get("question") or "").strip()
                a = (r.get("answer") or "").strip()
                if q and a:
                    rows.append({"question": q, "answer": a, "source": r.get("source","")})
            except Exception:
                pass
    return rows

def read_csv(path: str) -> List[Dict]:
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            if q and a:
                rows.append({"question": q, "answer": a, "source": r.get("source","")})
    return rows

def dedup_pairs(pairs: List[Dict]) -> List[Dict]:
    seen = set(); out = []
    for r in pairs:
        key = hashlib.sha1((r["question"].lower().strip()+"|"+r["answer"][:300]).encode()).hexdigest()
        if key in seen: 
            continue
        seen.add(key); out.append(r)
    return out

def train_val_split(pairs: List[Dict], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_val = max(1, int(n * val_ratio)) if n > 10 else max(1, n//6 or 1)
    return pairs[n_val:], pairs[:n_val]

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_jsonl", default="corpus/qa/qa_pairs.jsonl")
    ap.add_argument("--qa_csv",   default="corpus/qa/qa_pairs.csv")
    ap.add_argument("--base_model_id", default="google/flan-t5-base")
    ap.add_argument("--out_dir", default="models/flan_t5_brain_qa")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--max_src_len", type=int, default=384)
    ap.add_argument("--max_tgt_len", type=int, default=256)
    ap.add_argument("--use_lora", action="store_true")
    args = ap.parse_args()

    # 1) Load + dedup
    pairs = read_jsonl(args.qa_jsonl) + read_csv(args.qa_csv)
    pairs = dedup_pairs(pairs)
    if len(pairs) < 50:
        print(f"[FT] WARNING: very few pairs after dedup: {len(pairs)} — training may be unstable. "
              f"Check that your qa_pairs.* aren’t empty or filtered too hard.")
    train_rows, val_rows = train_val_split(pairs, val_ratio=args.val_split)

    print(f"[FT] Q/A examples -> train: {len(train_rows)} | val: {len(val_rows)}")

    # 2) HF datasets
    train_ds = Dataset.from_list(train_rows) if train_rows else Dataset.from_list([])
    val_ds   = Dataset.from_list(val_rows)   if val_rows   else Dataset.from_list([])
    dsd = DatasetDict({"train": train_ds, "validation": val_ds})

    # 3) Tokenizer + Model
    tok = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True)
    tok.padding_side = "right"

    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_id)

    # 3b) Optional: LoRA
    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05,
                bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
                target_modules=["q","k","v","o","wi","wo"]  # works for T5-family
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
            print("[FT] Using LoRA adapters")
        except Exception as e:
            print(f"[FT] LoRA unavailable, continuing without. ({type(e).__name__}: {e})")

    # 4) Preprocessing → must return dicts (NOT tokenizers.Encoding)
    def preprocess(batch):
        # input text
        inputs = [f"Question: {q.strip()}\nAnswer concisely:" for q in batch["question"]]
        model_inputs = tok(
            inputs,
            max_length=args.max_src_len,
            truncation=True,
            padding="max_length",
            return_tensors=None,   # <-- return plain dict (lists), not torch tensors
        )
        # targets
        with tok.as_target_tokenizer():
            labels = tok(
                batch["answer"],
                max_length=args.max_tgt_len,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )
        # replace pad token ids in labels with -100 so they’re ignored
        pad = tok.pad_token_id
        labels_ids = []
        for seq in labels["input_ids"]:
            labels_ids.append([(-100 if (pad is not None and t == pad) else t) for t in seq])
        model_inputs["labels"] = labels_ids
        return model_inputs

    dsd = dsd.map(preprocess, batched=True, remove_columns=dsd["train"].column_names)

    # 5) Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

    # 6) Training args
    targs = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=max(1, args.bsz // 2),
        learning_rate=args.lr,
        evaluation_strategy="epoch",   # use eval_strategy (works on 4.45+)
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=False,
        bf16=False,
        report_to=[],
    )

    # 7) Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tok,
        data_collator=data_collator,
    )

    # 8) Train
    trainer.train()
    os.makedirs(args.out_dir, exist_ok=True)
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)

    # 9) Tiny marker file
    with open(os.path.join(args.out_dir, "FINETUNED.OK"), "w") as f:
        f.write("ok\n")

    print(f"[FT] Done. Saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
