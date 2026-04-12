import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.app.core.config import MEMORY_DIR, get_settings
from backend.app.services.memory_extraction import extract_long_term_bullets, merge_bullets
from backend.app.services.vector_memory import get_vector_memory

logger = logging.getLogger("ace_claw")


def _ensure_long_term(data: dict) -> dict:
    lt = data.get("long_term")
    if not isinstance(lt, dict):
        lt = {}
        data["long_term"] = lt
    bullets = lt.get("bullets")
    if not isinstance(bullets, list):
        lt["bullets"] = []
    else:
        lt["bullets"] = bullets
    return lt


class MemoryStore:
    def md_path(self, session_id: str) -> Path:
        return MEMORY_DIR / f"{session_id}.md"

    def json_path(self, session_id: str) -> Path:
        return MEMORY_DIR / f"{session_id}.json"

    def longterm_md_path(self, session_id: str) -> Path:
        return MEMORY_DIR / f"{session_id}_longterm.md"

    def memory_path(self, session_id: str) -> str:
        return str(self.md_path(session_id))

    def _write_longterm_md(self, session_id: str, bullets: list[str], updated_at: str) -> None:
        path = self.longterm_md_path(session_id)
        lines = [
            f"# Long-term memory — `{session_id}`",
            "",
            f"_Updated: {updated_at}_",
            "",
        ]
        for b in bullets:
            lines.append(f"- {b}")
        if not bullets:
            lines.append("_(No extracted bullets yet.)_")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    async def append_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        md_file = self.md_path(session_id)
        block = (
            f"## {ts}\n"
            f"**user**: {user_message}\n\n"
            f"**assistant**: {assistant_message}\n\n"
            "---\n"
        )
        with md_file.open("a", encoding="utf-8") as f:
            f.write(block)

        jf = self.json_path(session_id)
        data: dict = {"session_id": session_id, "updated_at": ts, "messages": []}
        if jf.exists():
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        if "messages" not in data or not isinstance(data["messages"], list):
            data["messages"] = []
        _ensure_long_term(data)
        data["session_id"] = session_id
        data["updated_at"] = ts
        data["messages"].append({"role": "user", "content": user_message, "ts": ts})
        data["messages"].append({"role": "assistant", "content": assistant_message, "ts": ts})

        mem_cfg = get_settings().memory
        user_turns = sum(1 for m in data["messages"] if m.get("role") == "user")
        if (
            mem_cfg.long_term_enabled
            and mem_cfg.long_term_every_n_user_turns > 0
            and user_turns % mem_cfg.long_term_every_n_user_turns == 0
        ):
            ctx_n = mem_cfg.extraction_context_messages
            slice_msgs = data["messages"][-ctx_n:] if ctx_n else data["messages"]
            recent_dialogue = "\n".join(
                f'{m.get("role", "?")}: {m.get("content", "")}' for m in slice_msgs
            )
            try:
                new_bullets = await extract_long_term_bullets(
                    recent_dialogue=recent_dialogue,
                    user_message=user_message,
                    assistant_message=assistant_message,
                )
                lt = _ensure_long_term(data)
                merged, added = merge_bullets(
                    lt["bullets"],
                    new_bullets,
                    max_bullets=mem_cfg.max_long_term_bullets,
                )
                lt["bullets"] = merged
                lt["last_extracted_at"] = ts
                vstore = get_vector_memory()
                if vstore is not None:
                    for fact in added:
                        try:
                            vstore.remember_long_fact(session_id, fact)
                        except Exception as e:
                            logger.warning("vector long-term write failed: %s", e)
            except Exception as e:
                logger.warning("long-term extraction failed: %s", e)

        jf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        lt_final = _ensure_long_term(data)
        self._write_longterm_md(session_id, lt_final.get("bullets", []), data["updated_at"])

        vstore = get_vector_memory()
        if vstore is not None:
            try:
                vstore.remember_turn(session_id, user_message, assistant_message)
            except Exception as e:
                logger.warning("vector memory write failed: %s", e)

    async def read(self, session_id: str) -> str:
        path = self.md_path(session_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def list_memory_files(self) -> list[dict[str, str]]:
        """Entries with API-style paths `memory/{session_id}.md|json` (one row per session json)."""
        if not MEMORY_DIR.exists():
            return []
        out: list[dict[str, str]] = []
        for js_p in sorted(MEMORY_DIR.glob("*.json")):
            if js_p.stem.endswith("_longterm"):
                continue
            sid = js_p.stem
            md_p = self.md_path(sid)
            mtime = js_p.stat().st_mtime
            if md_p.exists():
                mtime = max(mtime, md_p.stat().st_mtime)
            lt_p = self.longterm_md_path(sid)
            if lt_p.exists():
                mtime = max(mtime, lt_p.stat().st_mtime)
            out.append(
                {
                    "session_id": sid,
                    "path_md": f"memory/{sid}.md",
                    "path_json": f"memory/{sid}.json",
                    "path_longterm_md": f"memory/{sid}_longterm.md",
                    "updated_at": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                }
            )
        out.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return out
