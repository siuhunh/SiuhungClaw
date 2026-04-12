"""Periodic LLM extraction of durable facts into long-term memory (JSON + Milvus `long`)."""

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.core.model_factory import get_chat_model

logger = logging.getLogger("ace_claw")

_SYSTEM = """You are a memory curator for a coding assistant. Given recent dialogue and the latest user–assistant turn, output 0–6 short, standalone facts worth remembering later (preferences, goals, tech stack, project names, constraints). Skip trivial or duplicate information.

Reply with ONLY a JSON array of strings. No markdown fences, no commentary.
Example: ["User prefers TypeScript", "Project AceClaw uses FastAPI backend"]"""


def _stringify_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _parse_bullet_array(raw: str) -> list[str]:
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("long-term extraction JSON parse failed: %s", raw[:200])
        return []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for item in data:
        if isinstance(item, str):
            s = item.strip()
            if s and len(s) <= 512:
                out.append(s)
    return out


def _norm_key(s: str) -> str:
    return " ".join(s.lower().split())


async def extract_long_term_bullets(
    *,
    recent_dialogue: str,
    user_message: str,
    assistant_message: str,
) -> list[str]:
    model = get_chat_model()
    human = HumanMessage(
        content=(
            f"Recent dialogue (most recent last):\n{recent_dialogue}\n\n"
            f"Latest turn:\nuser: {user_message}\nassistant: {assistant_message}\n\n"
            "JSON array only:"
        )
    )
    out = await model.ainvoke([SystemMessage(content=_SYSTEM), human])
    content: Any = getattr(out, "content", out)
    return _parse_bullet_array(_stringify_content(content))


def merge_bullets(
    existing: list[str],
    new_items: list[str],
    *,
    max_bullets: int,
) -> tuple[list[str], list[str]]:
    """Return (merged_list, newly_added_for_vector)."""
    seen = {_norm_key(b) for b in existing}
    merged = list(existing)
    added: list[str] = []
    for item in new_items:
        k = _norm_key(item)
        if not k or k in seen:
            continue
        seen.add(k)
        merged.append(item)
        added.append(item)
    overflow = len(merged) - max_bullets
    if overflow > 0:
        merged = merged[overflow:]
    return merged, added
