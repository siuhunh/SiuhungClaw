"""
3.9 System Prompt：每次 Agent 调用时从 backend/workspace/*.md 重新读取并组装，
顺序与 dev.md 一致；技能列表可通过 {{AUTO_SKILLS}} 注入；
会话记忆分为短期（最近消息窗口）与长期（抽取要点 + 可选 _longterm.md），全文 transcript 仅落盘不整段注入。
"""

import json
from pathlib import Path

from backend.app.core.config import MEMORY_DIR, SYSTEM_PROMPT_WORKSPACE, get_settings


def _read_text(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _load_session_memory_bundle(session_id: str) -> tuple[list[dict], list[str]]:
    """Return (messages, long_term_bullets) from session json if present."""
    path = MEMORY_DIR / f"{session_id}.json"
    if not path.exists():
        return [], []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], []
    messages = data.get("messages")
    if not isinstance(messages, list):
        messages = []
    lt = data.get("long_term")
    bullets: list[str] = []
    if isinstance(lt, dict):
        raw = lt.get("bullets")
        if isinstance(raw, list):
            bullets = [str(b) for b in raw if isinstance(b, str) and b.strip()]
    return messages, bullets


def _auto_skills_block() -> str:
    from backend.app.api.routes.skills import skill_manager

    lines = ["<available_skills>"]
    for s in skill_manager.list():
        lines.append(f"- **{s.name}** (`{s.path}`): {s.description}")
    if len(lines) == 1:
        lines.append("- （当前无已扫描技能，可在 storage/skill/ 添加 *.md）")
    lines.append("</available_skills>")
    return "\n".join(lines)


def build_system_prompt(session_id: str) -> str:
    ws = SYSTEM_PROMPT_WORKSPACE
    blocks: list[str] = []
    mem_cfg = get_settings().memory

    # 1) Skills Snapshot
    snap_path = ws / "SKILLS_SNAPSHOT.md"
    snap = _read_text(snap_path)
    if not snap:
        snap = "<!-- Skills Snapshot -->\n\n{{AUTO_SKILLS}}"
    if "{{AUTO_SKILLS}}" in snap:
        snap = snap.replace("{{AUTO_SKILLS}}", _auto_skills_block())
    else:
        snap = snap + "\n\n" + _auto_skills_block()
    blocks.append(snap)

    # 2) Soul
    soul = _read_text(ws / "SOUL.md")
    blocks.append(soul if soul else "<!-- Soul -->\n\n（未找到 SOUL.md，请在 backend/workspace/ 补充核心设定。）")

    # 3) Identity
    ident = _read_text(ws / "IDENTITY.md")
    blocks.append(
        ident if ident else "<!-- Identity -->\n\n（未找到 IDENTITY.md，请补充自我认知。）"
    )

    # 4) User Profile
    user = _read_text(ws / "USER.md")
    blocks.append(
        user if user else "<!-- User Profile -->\n\n（未找到 USER.md，请补充用户画像。）"
    )

    # 5) Agents Guide
    agents = _read_text(ws / "AGENTS.md")
    blocks.append(
        agents
        if agents
        else "<!-- Agents Guide -->\n\n（未找到 AGENTS.md，请补充行为准则与记忆操作说明。）"
    )

    messages, long_bullets = _load_session_memory_bundle(session_id)
    if not long_bullets:
        lt_md = _read_text(MEMORY_DIR / f"{session_id}_longterm.md")
        if lt_md:
            for line in lt_md.splitlines():
                s = line.strip()
                if s.startswith("- "):
                    long_bullets.append(s[2:].strip())

    if long_bullets:
        lt_body = "\n".join(f"- {b}" for b in long_bullets)
    else:
        lt_body = (
            "（尚未抽取长期要点；多轮对话后写入 "
            f"`storage/memory/{session_id}.json` 的 `long_term.bullets` 与 `{session_id}_longterm.md`。）"
        )
    blocks.append(f"<!-- Long-term Memory (extracted) -->\n\n{lt_body}")

    n = mem_cfg.short_term_messages
    tail = messages[-n:] if n else messages
    if tail:
        st_chunks: list[str] = []
        for m in tail:
            role = m.get("role", "?")
            content = m.get("content", "")
            st_chunks.append(f"**{role}**: {content}")
        st_body = "\n\n".join(st_chunks)
    else:
        st_body = (
            f"（本会话尚无结构化消息缓存；完整对话日志见 `storage/memory/{session_id}.md`。"
            " RAG 检索请用 search_knowledge_base，勿与对话记忆混淆。）"
        )
    blocks.append(f"<!-- Short-term Memory (recent window) -->\n\n{st_body}")

    archival = (
        f"<!-- Session archival note -->\n\n"
        f"Full transcript (not injected): `storage/memory/{session_id}.md`."
    )
    blocks.append(archival)

    return "\n\n".join(blocks)
