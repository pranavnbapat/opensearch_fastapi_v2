# tools/debug_llm_summary.py

from __future__ import annotations

from typing import Any, Dict, List


def build_llm_summary_from_explain_top3(explain_top3: List[Dict[str, Any]]) -> str:
    """
    Makes a short, LLM-friendly summary from _debug.explain_top3.
    We avoid dumping the entire explanation tree; we only send per-hit metadata.
    """
    lines: List[str] = []
    lines.append("Search debug summary for top results:")

    for h in explain_top3:
        hit_id = h.get("_id")
        parent_id = h.get("parent_id")
        score = h.get("_score")
        expl = h.get("_explanation") or {}
        expl_value = expl.get("value")

        lines.append("")
        lines.append(f"Hit: {hit_id}")
        lines.append(f"Parent: {parent_id}")
        lines.append(f"Final _score: {score}")
        lines.append(f"Explain root value: {expl_value}")
        lines.append("Explain tree present: yes")

    return "\n".join(lines).strip()
