# tools/analyse_debug.py

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

from dotenv import load_dotenv


# ----------------------------- Parsing helpers -----------------------------

WEIGHT_RE = re.compile(
    r"""^weight\((?P<field>[^:()]+):(?P<term>[^ )]+)\s+in\s+(?P<docid>\d+)\)\s+\[PerFieldSimilarity\],\s+result\s+of:$"""
)

# Examples we want to catch:
# "score(freq=1.0), computed as boost * idf * tf from:"
SCORE_RE = re.compile(r"score\(freq=(?P<freq>[0-9.]+)\), computed as boost \* idf \* tf from:")

# "idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:"
IDF_DESC_RE = re.compile(r"^idf, computed as log\(.+\) from:$")

# "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:"
TF_DESC_RE = re.compile(r"^tf, computed as freq / \(.+\) from:$")

# Hybrid/pipeline-ish tag
WITHIN_TOP_RE = re.compile(r"^within top\s+(?P<k>\d+)\s+docs$")


def _get(obj: Any, path: str) -> Any:
    """
    Safe nested get via dot-path. Returns None if anything is missing.
    """
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _load_repo_env() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    load_dotenv(dotenv_path=env_path, override=False)


def _find_first_debug_root(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Accepts:
      - file is the full API response (contains `_debug`)
      - file is just the `_debug` object
      - file is some wrapper object with `_debug` nested
    """
    if isinstance(payload, dict) and "_debug" in payload and isinstance(payload["_debug"], dict):
        return payload["_debug"]
    if isinstance(payload, dict) and "explain_top3" in payload:
        return payload  # file is directly the _debug dict
    # Try a few common nestings
    for candidate in ("detail", "data", "response"):
        maybe = _get(payload, f"{candidate}._debug")
        if isinstance(maybe, dict):
            return maybe
    return None


@dataclass
class BM25Contribution:
    hit_id: str
    parent_id: Optional[str]
    field: str
    term: str
    doc_internal_id: Optional[str]
    value: float

    # Optional BM25 components
    boost: Optional[float] = None
    idf: Optional[float] = None
    n: Optional[float] = None
    N: Optional[float] = None
    tf: Optional[float] = None
    freq: Optional[float] = None
    k1: Optional[float] = None
    b: Optional[float] = None
    dl: Optional[float] = None
    avgdl: Optional[float] = None


@dataclass
class HybridSignal:
    hit_id: str
    parent_id: Optional[str]
    kind: str  # e.g. "within_top"
    value: float
    meta: Dict[str, Any]


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _extract_named_numbers(details: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    From lists like:
      [{"value": 1.2, "description": "k1, ..."}, ...]
    return {"k1":1.2, ...} for keys we recognise.
    """
    out: Dict[str, float] = {}
    for d in details or []:
        desc = str(d.get("description", ""))
        val = _as_float(d.get("value"))
        if val is None:
            continue

        # We only need the leading token name before the comma:
        # "k1, term saturation parameter" -> "k1"
        key = desc.split(",", 1)[0].strip()
        if key in {"boost", "idf", "tf", "freq", "k1", "b", "dl", "avgdl", "n", "N"}:
            out[key] = val
    return out


def _walk_explanation(
    node: Dict[str, Any],
    *,
    hit_id: str,
    parent_id: Optional[str],
    doc_internal_id: Optional[str],
    bm25_rows: List[BM25Contribution],
    hybrid_rows: List[HybridSignal],
) -> None:
    """
    Depth-first traversal of Lucene explanation tree.
    """
    if not isinstance(node, dict):
        return

    desc = str(node.get("description", ""))
    val = _as_float(node.get("value")) or 0.0
    details = node.get("details") or []

    # 1) Capture hybrid-ish markers
    m_top = WITHIN_TOP_RE.match(desc)
    if m_top:
        hybrid_rows.append(
            HybridSignal(
                hit_id=hit_id,
                parent_id=parent_id,
                kind="within_top",
                value=val,
                meta={"k": int(m_top.group("k"))},
            )
        )

    # 2) Capture BM25 weight(field:term ...) nodes
    m_weight = WEIGHT_RE.match(desc)
    if m_weight:
        field = m_weight.group("field")
        term = m_weight.group("term")
        docid = m_weight.group("docid")

        contrib = BM25Contribution(
            hit_id=hit_id,
            parent_id=parent_id,
            field=field,
            term=term,
            doc_internal_id=docid,
            value=val,
        )

        # Look for the "score(freq=...)" child and its grandchildren
        for child in details:
            c_desc = str(child.get("description", ""))
            c_details = child.get("details") or []

            m_score = SCORE_RE.search(c_desc)
            if m_score:
                contrib.freq = _as_float(m_score.group("freq"))

                # Typically c_details contains 3 nodes: boost/idf/tf
                # Each of those has its own `details` where n/N or dl/avgdl are provided.
                for part in c_details:
                    p_desc = str(part.get("description", ""))
                    p_val = _as_float(part.get("value"))
                    p_details = part.get("details") or []

                    if p_desc.strip() == "boost":
                        contrib.boost = p_val

                    elif IDF_DESC_RE.match(p_desc):
                        contrib.idf = p_val
                        nums = _extract_named_numbers(p_details)
                        # In idf subtree, we see "n, ..." and "N, ..."
                        contrib.n = nums.get("n")
                        contrib.N = nums.get("N")

                    elif TF_DESC_RE.match(p_desc):
                        contrib.tf = p_val
                        nums = _extract_named_numbers(p_details)
                        # In tf subtree: freq/k1/b/dl/avgdl
                        contrib.k1 = nums.get("k1")
                        contrib.b = nums.get("b")
                        contrib.dl = nums.get("dl")
                        contrib.avgdl = nums.get("avgdl")
                        # Sometimes tf subtree repeats freq
                        contrib.freq = contrib.freq or nums.get("freq")

        bm25_rows.append(contrib)

    # Recurse
    for child in details:
        _walk_explanation(
            child,
            hit_id=hit_id,
            parent_id=parent_id,
            doc_internal_id=doc_internal_id,
            bm25_rows=bm25_rows,
            hybrid_rows=hybrid_rows,
        )


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_hits(debug: Dict[str, Any], hits_path: str) -> List[Dict[str, Any]]:
    """
    Default hits_path = "explain_top3"
    """
    hits = _get(debug, hits_path) if hits_path else debug.get("explain_top3")
    if not isinstance(hits, list):
        raise ValueError(f"Could not find a list at _debug.{hits_path!s}")
    return hits


# ----------------------------- Reporting -----------------------------

def _ranked(items: Iterable[Tuple[str, float]], top: int) -> List[Tuple[str, float]]:
    return sorted(items, key=lambda x: x[1], reverse=True)[:top]


def _print_hit_summary(
    *,
    hit: Dict[str, Any],
    bm25_rows: List[BM25Contribution],
    hybrid_rows: List[HybridSignal],
    top: int,
) -> None:
    hit_id = str(hit.get("_id", ""))
    parent_id = hit.get("parent_id")
    score = hit.get("_score")
    expl = hit.get("_explanation") or {}
    expl_value = expl.get("value")

    print("=" * 90)
    print(f"Hit: {hit_id}")
    print(f"Parent: {parent_id}")
    print(f"Returned _score: {score}")
    print(f"Explain root value: {expl_value}")
    print("-" * 90)

    # Hybrid-ish signals
    hrows = [r for r in hybrid_rows if r.hit_id == hit_id]
    if hrows:
        print("Hybrid/Pipeline signals:")
        for r in hrows:
            if r.kind == "within_top":
                print(f"  - within top {r.meta.get('k')} docs: {r.value}")
            else:
                print(f"  - {r.kind}: {r.value} meta={r.meta}")
        print("-" * 90)

    # BM25 contributions, grouped
    rows = [r for r in bm25_rows if r.hit_id == hit_id]
    if not rows:
        print("No BM25 weight(field:term ...) nodes found in explanation tree.")
        return

    # Top contributions overall
    overall = [(f"{r.field}:{r.term}", r.value) for r in rows]
    print(f"Top {top} BM25 term contributions (value = contribution of that field/term):")
    for k, v in _ranked(overall, top):
        print(f"  {v:>10.6f}  {k}")
    print("-" * 90)

    # Field totals
    field_totals: Dict[str, float] = {}
    for r in rows:
        field_totals[r.field] = field_totals.get(r.field, 0.0) + r.value

    print("Field totals (sum of term contributions per field):")
    for field, v in _ranked(field_totals.items(), top=50):
        print(f"  {v:>10.6f}  {field}")
    print("-" * 90)

    # Term totals
    term_totals: Dict[str, float] = {}
    for r in rows:
        term_totals[r.term] = term_totals.get(r.term, 0.0) + r.value

    print("Term totals (sum across fields):")
    for term, v in _ranked(term_totals.items(), top=50):
        print(f"  {v:>10.6f}  {term}")

    # Show detailed breakdown for the top few rows
    print("-" * 90)
    print("Detail for top contributions:")
    for key, _ in _ranked(overall, min(top, 8)):
        field, term = key.split(":", 1)
        r_best = max((r for r in rows if r.field == field and r.term == term), key=lambda r: r.value)
        print(f"  {r_best.value:>10.6f}  {field}:{term}")
        print(f"    boost={r_best.boost} idf={r_best.idf} (n={r_best.n}, N={r_best.N})")
        print(f"    tf={r_best.tf} freq={r_best.freq} dl={r_best.dl} avgdl={r_best.avgdl} k1={r_best.k1} b={r_best.b}")


def _write_csv(path: Path, bm25_rows: List[BM25Contribution], hybrid_rows: List[HybridSignal]) -> None:
    # BM25 rows
    bm25_path = path
    hyb_path = path.with_name(path.stem + "_hybrid_signals.csv")

    with bm25_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(BM25Contribution(
            hit_id="", parent_id=None, field="", term="", doc_internal_id=None, value=0.0
        )).keys()))
        w.writeheader()
        for r in bm25_rows:
            w.writerow(asdict(r))

    with hyb_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["hit_id", "parent_id", "kind", "value", "meta"])
        w.writeheader()
        for r in hybrid_rows:
            w.writerow({
                "hit_id": r.hit_id,
                "parent_id": r.parent_id,
                "kind": r.kind,
                "value": r.value,
                "meta": json.dumps(r.meta, ensure_ascii=False),
            })


# ----------------------------- Optional LLM explainer -----------------------------

def _llm_explain(
    *,
    llm_url: str,
    llm_model: str,
    llm_key: Optional[str],
    extracted_summary: str,
    timeout_s: int = 60,
) -> str:
    if requests is None:
        raise RuntimeError("requests is not installed. Install it with: pip install requests")

    headers = {"Content-Type": "application/json"}
    if llm_key:
        headers["Authorization"] = f"Bearer {llm_key}"

    payload = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a search relevance engineer. Explain the scoring/debug summary "
                    "in plain English, focusing on which fields/terms dominate and why."
                ),
            },
            {"role": "user", "content": extracted_summary},
        ],
        "temperature": 0.2,
    }

    base = llm_url.rstrip("/")
    endpoint = base
    if not base.endswith("/v1/chat/completions"):
        endpoint = f"{base}/v1/chat/completions"

    r = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    # OpenAI style: choices[0].message.content
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        # Fallback: return raw JSON
        return json.dumps(data, indent=2, ensure_ascii=False)


# ----------------------------- Main -----------------------------

def main() -> int:
    _load_repo_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, help="Path to JSON file containing _debug output")
    ap.add_argument("--hits-path", default="explain_top3", help="Dot-path inside _debug to the hits list")
    ap.add_argument("--top", type=int, default=12, help="How many top contributions to print")
    ap.add_argument("--csv", default=None,
                    help="Write BM25 contributions to this CSV path (also writes *_hybrid_signals.csv)")
    ap.add_argument("--llm-url", default=os.getenv("LLM_URL"),
                    help="Chat-completions base URL (OpenAI-compatible).")
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL"), help="Model name.")
    ap.add_argument("--llm-key", default=os.getenv("LLM_KEY", None), help="Bearer token, if required.")

    args = ap.parse_args()

    payload = _load_json(Path(args.infile))
    debug = _find_first_debug_root(payload)
    if debug is None:
        print("Could not locate a _debug object in the JSON file.", file=sys.stderr)
        return 2

    hits = _iter_hits(debug, args.hits_path)

    bm25_rows: List[BM25Contribution] = []
    hybrid_rows: List[HybridSignal] = []

    # Parse each hit
    for hit in hits:
        hit_id = str(hit.get("_id", ""))
        parent_id = hit.get("parent_id")
        expl = hit.get("_explanation") or {}
        _walk_explanation(
            expl,
            hit_id=hit_id,
            parent_id=parent_id,
            doc_internal_id=None,
            bm25_rows=bm25_rows,
            hybrid_rows=hybrid_rows,
        )

    # Print summaries per hit
    for hit in hits:
        _print_hit_summary(
            hit=hit,
            bm25_rows=bm25_rows,
            hybrid_rows=hybrid_rows,
            top=args.top,
        )
        print()

    # CSV export
    if args.csv:
        _write_csv(Path(args.csv), bm25_rows, hybrid_rows)
        print(f"Wrote CSV: {args.csv}")
        print(f"Wrote CSV: {Path(args.csv).with_name(Path(args.csv).stem + '_hybrid_signals.csv')}")

    # Optional LLM explanation of the extracted summary
    if args.llm_url and args.llm_url.strip():
        # Create a compact summary for the LLM: top contributions across all hits
        lines: List[str] = []
        lines.append("Extracted debug summary (BM25 term contributions):")

        # group by hit
        by_hit: Dict[str, List[BM25Contribution]] = {}
        for r in bm25_rows:
            by_hit.setdefault(r.hit_id, []).append(r)

        for hit in hits:
            hit_id = str(hit.get("_id", ""))
            lines.append("")
            lines.append(f"Hit {hit_id} parent={hit.get('parent_id')} _score={hit.get('_score')}")
            rows = by_hit.get(hit_id, [])
            rows_sorted = sorted(rows, key=lambda r: r.value, reverse=True)[: min(args.top, 10)]
            for r in rows_sorted:
                lines.append(
                    f"- {r.value:.4f} {r.field}:{r.term} boost={r.boost} idf={r.idf} tf={r.tf} freq={r.freq}"
                )
            for s in [h for h in hybrid_rows if h.hit_id == hit_id]:
                lines.append(f"- HYBRID {s.kind} value={s.value} meta={s.meta}")

        extracted = "\n".join(lines)
        print("\n" + "#" * 90)
        print("Explaining:\n")
        try:
            out = _llm_explain(
                llm_url=args.llm_url,
                llm_model=args.llm_model,
                llm_key=args.llm_key,
                extracted_summary=extracted,
            )
            print(out.strip())
        except Exception as e:
            print(f"LLM call failed: {e}", file=sys.stderr)
            print("If your endpoint is not OpenAI-compatible, tweak _llm_explain() payload parsing.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
