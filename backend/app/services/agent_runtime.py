import logging
from collections.abc import AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from backend.app.core.config import get_settings
from backend.app.core.model_factory import get_chat_model
from backend.app.services.system_prompt import build_system_prompt
from backend.app.services.vector_memory import get_vector_memory
from backend.tools.registry import get_core_tools

logger = logging.getLogger(__name__)


def _stringify_message_content(content: object) -> str:
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


def _final_text_from_agent_result(out: object) -> str:
    if isinstance(out, dict):
        msgs = out.get("messages")
        if isinstance(msgs, list) and msgs:
            return _stringify_message_content(getattr(msgs[-1], "content", ""))
    if hasattr(out, "content"):
        return _stringify_message_content(getattr(out, "content", ""))
    return str(out)


class AgentRuntime:
    def __init__(self) -> None:
        self._model = get_chat_model()
        self._tools = get_core_tools()
        self._tool_map = {t.name: t for t in self._tools}

    def _build_prompt_with_vector_memory(self, session_id: str, user_message: str) -> str:
        base = build_system_prompt(session_id)
        store = get_vector_memory()
        if store is None or not store.enabled:
            return base
        mem = get_settings().memory
        sections: list[str] = []
        if store.supports_memory_kinds:
            if mem.vector_recall_long_k > 0:
                long_hits = store.search(
                    session_id,
                    user_message,
                    memory_kinds=["long"],
                    limit=mem.vector_recall_long_k,
                )
                if long_hits:
                    sections.append(
                        "**Long-term (vector):**\n" + "\n".join(f"- {x}" for x in long_hits)
                    )
            if mem.vector_recall_short_k > 0:
                short_hits = store.search(
                    session_id,
                    user_message,
                    memory_kinds=["short"],
                    limit=mem.vector_recall_short_k,
                )
                if short_hits:
                    sections.append(
                        "**Short-term turns (vector):**\n" + "\n".join(f"- {x}" for x in short_hits)
                    )
        else:
            lim = max(1, mem.vector_recall_long_k + mem.vector_recall_short_k)
            recalled = store.search(session_id=session_id, query=user_message, limit=lim)
            if recalled:
                sections.append("\n".join(f"- {x}" for x in recalled))
        if not sections:
            return base
        block = "<!-- Vector Memory Matches -->\n\n" + "\n\n".join(sections)
        return f"{base}\n\n{block}"

    async def _manual_agent(self, user_message: str, session_id: str) -> str:
        """Tool-calling loop（create_agent 不可用时的回退）。"""
        system_text = self._build_prompt_with_vector_memory(session_id, user_message)
        model = self._model.bind_tools(self._tools)
        messages: list = [
            SystemMessage(content=system_text),
            HumanMessage(content=user_message),
        ]

        for _ in range(24):
            ai = await model.ainvoke(messages)
            messages.append(ai)
            if not isinstance(ai, AIMessage):
                continue
            tcalls = getattr(ai, "tool_calls", None) or []
            if not tcalls:
                return _stringify_message_content(ai.content)

            for tc in tcalls:
                name = tc.get("name")
                tid = tc.get("id") or ""
                args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
                tool = self._tool_map.get(name or "")
                if tool is None:
                    out = f"Unknown tool: {name}"
                else:
                    try:
                        out = tool.invoke(args)
                    except Exception as e:
                        out = f"Tool error: {e}"
                text = str(out)
                if len(text) > 12000:
                    text = text[:12000] + "\n...[truncated]"
                messages.append(ToolMessage(content=text, tool_call_id=tid))

        return "Stopped: maximum agent steps reached."

    async def run(self, user_message: str, session_id: str = "main_session") -> str:
        system_text = self._build_prompt_with_vector_memory(session_id, user_message)
        try:
            from langchain.agents import create_agent

            agent = create_agent(
                model=self._model,
                tools=self._tools,
                system_prompt=system_text,
            )
            out = await agent.ainvoke({"messages": [HumanMessage(content=user_message)]})
            return _final_text_from_agent_result(out)
        except Exception as e:
            logger.debug("create_agent failed (%s); using manual tool loop.", e)
            return await self._manual_agent(user_message, session_id)

    async def stream(
        self, user_message: str, session_id: str = "main_session"
    ) -> AsyncGenerator[str, None]:
        """Stream final answer in small chunks (tool loop runs to completion first)."""
        text = await self.run(user_message, session_id=session_id)
        for chunk in text.split(" "):
            if chunk:
                yield chunk + " "
