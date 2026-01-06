#!/usr/bin/env python
# evaluate_kgrag_hardcoded.py
import json, re, string
from collections import Counter
from pathlib import Path

# ---------- 하드코딩된 파일 경로 ----------
PRED_PATH = Path("/workspace/PathCoRAG/PathCoRAG/hyper_multihoprag_result.json")  # 필요 시 수정
GOLD_PATH = Path("/workspace/PathCoRAG/PathCoRAG/multihoprag_qa.json")

# ---------- text normalization ----------
def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())

# ---------- metrics ----------
def compute_metrics(pred: str, gold: str):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    if not pred_tokens or not gold_tokens:
        em = int(pred_tokens == gold_tokens)
        return em, 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0, 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    em = int(pred_tokens == gold_tokens)

    return em, f1, precision, recall

# ---------- robust loader ----------
def load_pairs(path: Path, answer_key: str):
    """Return {question_text: answer} dict.

    허용:
      - 항목에서 질문 키가 'query' 또는 'question'
      - None / 비어있는 문자열 / answer_key 누락 → 스킵
      - 중복 질문: 마지막 항목이 overwrite (로그 출력)
    """
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    pairs = {}
    dup = 0
    if isinstance(data, dict):  # 혹시 dict 형태라면 values 순회
        data_iter = data.values()
    else:
        data_iter = data
    for d in data_iter:
        if not d or not isinstance(d, dict):
            continue
        q = d.get("query") or d.get("question")
        if not q or not isinstance(q, str) or not q.strip():
            continue
        if answer_key not in d:
            continue
        ans = d[answer_key]
        if ans is None:
            continue
        if q in pairs:
            dup += 1
        pairs[q] = ans
    if dup:
        print(f"[Info] {dup} duplicate questions overwritten in {path.name}")
    print(f"[Load] {path.name}: {len(pairs)} usable QA pairs")
    return pairs

# ---------- driver ----------
def main():
    if not GOLD_PATH.exists():
        raise SystemExit(f"Gold file not found: {GOLD_PATH}")
    if not PRED_PATH.exists():
        raise SystemExit(f"Pred file not found: {PRED_PATH}")

    gold = load_pairs(GOLD_PATH, "answer")
    pred = load_pairs(PRED_PATH, "result")

    em_sum = f1_sum = precision_sum = recall_sum = 0
    contain_correct = 0  # Accuracy: gold answer substring
    missing = 0
    error_pred = 0

    for q, gold_ans in gold.items():
        p = pred.get(q)
        if p is None:
            missing += 1
            continue
        if isinstance(p, str) and p.startswith('[Error]'):
            error_pred += 1
            continue
        em, f1_val, prec, rec = compute_metrics(str(p), str(gold_ans))
        em_sum += em
        f1_sum += f1_val
        precision_sum += prec
        recall_sum += rec
        if normalize(str(gold_ans)) in normalize(str(p)):
            contain_correct += 1

    compared = len(gold) - missing - error_pred
    if compared <= 0:
        print(f"No comparable items (gold={len(gold)} missing={missing} error_pred={error_pred})")
        return

    em        = em_sum        / compared
    f1        = f1_sum        / compared
    precision = precision_sum / compared
    recall    = recall_sum    / compared
    accuracy  = contain_correct / compared

    print(f"#gold items      : {len(gold)}")
    print(f"#pred items      : {len(pred)}")
    print(f"missing (no pred): {missing}")
    print(f"error preds      : {error_pred}")
    print(f"compared         : {compared}")
    print(f"Exact‑Match      : {em:.3f}")
    print(f"F1               : {f1:.3f}")
    print(f"Precision        : {precision:.3f}")
    print(f"Recall           : {recall:.3f}")
    print(f"Accuracy(subset) : {accuracy:.3f}")

if __name__ == "__main__":
    main()
