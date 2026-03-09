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


def build_llm_summary_from_advanced_matches(
    *,
    query: str,
    parsed_query: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> str:
    """
    Build a compact LLM-friendly summary for advanced endpoint clause matching.
    Only includes the top few results and their matched clause/evidence summary.
    """
    lines: List[str] = []
    lines.append(f"Advanced search debug summary for query: {query}")
    lines.append(f"Parsed query AST: {parsed_query.get('ast')}")
    if parsed_query.get("mode"):
        lines.append(f"Mode: {parsed_query.get('mode')}")
    if parsed_query.get("project_filters"):
        lines.append(f"Project filters: {parsed_query.get('project_filters')}")

    for item in results[:3]:
        dbg = ((item.get("_debug") or {}).get("matched_clauses") or {})
        lines.append("")
        lines.append(f"Result title: {item.get('title')}")
        lines.append(f"Result id: {item.get('_id')}")
        lines.append(f"Score: {item.get('_score')}")
        lines.append(f"Matched clauses: {dbg.get('matched_clause_tree')}")

    return "\n".join(lines).strip()
