"""Run both pipelines on the 10 questions and write results/ outputs.

Usage: python src/evaluate.py [--embedder tfidf|minilm] [--k 3]
"""
import argparse, csv, json, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunking import naive_chunks, structure_chunks
from embeddings import get_embedder
from generate import generate_answer
from questions import QUESTIONS
from retrieval import VectorStore


def judge(item, retrieved_text: str) -> str:
    """Rubric-based relevance judgement: yes / partial / no (n/a if unanswerable)."""
    if not item["answerable"]:
        return "n/a"
    text = retrieved_text.lower()
    found = sum(any(alt in text for alt in grp) for grp in item["facts"])
    if found >= item["min_needed"]:
        return "yes"
    return "partial" if found > 0 else "no"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="tfidf", choices=["tfidf", "minilm"])
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    text = (ROOT / "data" / "diabetes_reference_document.md").read_text(encoding="utf-8")
    strategies = {"A_naive": naive_chunks(text), "B_structure": structure_chunks(text)}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    # dump chunks for manual inspection
    for name, chunks in strategies.items():
        with open(out / f"chunks_{name}.txt", "w", encoding="utf-8") as f:
            for i, c in enumerate(chunks):
                f.write(f"===== chunk {i} ({len(c)} chars) =====\n{c}\n\n")

    stores = {n: VectorStore(c, get_embedder(args.embedder)) for n, c in strategies.items()}
    log, rows = [], []
    for qi, item in enumerate(QUESTIONS, 1):
        row = {"#": qi, "question": item["q"]}
        for name, store in stores.items():
            hits = store.retrieve(item["q"], args.k)
            joined = "\n".join(h[2] for h in hits)
            verdict = judge(item, joined)
            top1 = judge(item, hits[0][2])
            answer = generate_answer(item["q"], [h[2] for h in hits])
            row[f"{name}_chunks"] = ",".join(str(h[0]) for h in hits)
            row[f"{name}_topk"] = verdict
            row[f"{name}_top1"] = top1
            log.append({"q": qi, "strategy": name, "verdict_topk": verdict, "verdict_top1": top1,
                        "generated_answer": answer,
                        "retrieved": [{"chunk_id": h[0], "score": round(h[1], 3), "text": h[2]} for h in hits]})
        rows.append(row)

    # summary statistics
    summary = {"embedder": args.embedder, "k": args.k}
    for name, chunks in strategies.items():
        sizes = [len(c) for c in chunks]
        ans = [r for r, it in zip(rows, QUESTIONS) if it["answerable"]]
        yes = sum(r[f"{name}_topk"] == "yes" for r in ans)
        part = sum(r[f"{name}_topk"] == "partial" for r in ans)
        top1_yes = sum(r[f"{name}_top1"] == "yes" for r in ans)
        summary[name] = {
            "num_chunks": len(chunks), "avg_chunk_chars": round(st.mean(sizes), 1),
            "min_chunk_chars": min(sizes), "max_chunk_chars": max(sizes),
            "answerable_questions": len(ans),
            "accuracy_strict_topk_pct": round(100 * yes / len(ans), 1),
            "accuracy_lenient_topk_pct": round(100 * (yes + part) / len(ans), 1),
            "accuracy_strict_top1_pct": round(100 * top1_yes / len(ans), 1),
            "accuracy_strict_over_all10_pct": round(100 * yes / len(QUESTIONS), 1),
        }

    (out / "retrieval_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out / "results_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    md = ["| # | Question | A: chunks | A: top-k | A: top-1 | B: chunks | B: top-k | B: top-1 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['#']} | {r['question']} | {r['A_naive_chunks']} | {r['A_naive_topk']} | {r['A_naive_top1']} "
                  f"| {r['B_structure_chunks']} | {r['B_structure_topk']} | {r['B_structure_top1']} |")
    (out / "results_table.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md)); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
