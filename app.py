"""
🤖 AI Agent with Tools: Codomax Internship, Module 05
Understand Task → Decide Action → Use Tool → Process Result → Respond

Run locally:  streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from agent import Agent, AgentError, GeminiLLM, get_api_key
from tools import TOOL_DECLARATIONS, ToolBox

st.set_page_config(page_title="AI Agent · Codomax", page_icon="🤖", layout="wide")

TOOL_ICONS = {"calculator": "🧮", "get_weather": "🌦️", "search_wikipedia": "📖", "get_current_datetime": "🕒",
              "query_database": "🗄️", "add_task": "➕", "remember": "🧠", "recall_memories": "🧠"}
EXAMPLES = [
    "What's the weather in Pune for the next 3 days? Should I carry an umbrella?",
    "Who created Python, and how many years ago was it first released?",
    "Which team member has the most unfinished high-priority tasks?",
    "Add a task for Vishva to finish Module 05 by next Friday, high priority",
    "Remember that my favourite programming language is Python",
    "Delete all completed tasks from the database",
]
MAX_MESSAGES = 30  # per session, protects the free quota

ss = st.session_state
ss.setdefault("messages", [])      # display history: {"role", "content", "steps", "meta"}
ss.setdefault("toolbox", None)


def show_error(e: Exception):
    msg = getattr(e, "user_message", str(e))
    st.error(msg)
    if getattr(e, "detail", ""):
        with st.expander("Technical details"):
            st.code(e.detail)


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str) -> GeminiLLM:
    return GeminiLLM(api_key)


def render_step(step: dict):
    icon = TOOL_ICONS.get(step["name"], "🔧")
    if step["kind"] == "tool":
        failed = isinstance(step["result"], dict) and "error" in step["result"]
        with st.expander(f"{icon} **{step['name']}**({', '.join(f'{k}={v!r}' for k, v in step['args'].items())})"
                         f" · {step['seconds']}s {'⚠️' if failed else '✅'}"):
            st.markdown("**Arguments the agent chose:**")
            st.json(step["args"])
            st.markdown("**Tool result sent back to the agent:**")
            st.json(step["result"])
    elif step["kind"] == "info":
        st.warning(f"🛑 {step['result']}")


# ============================================================================ sidebar
with st.sidebar:
    st.header("⚙️ Agent settings")
    api_key = get_api_key() or st.text_input("Gemini API key", type="password",
                                             help="Free key from aistudio.google.com")
    llm = None
    if api_key:
        try:
            llm = get_llm(api_key)
        except Exception as e:
            show_error(e)

    if llm:
        prefer = st.selectbox("Preferred model", llm.models,
                              help="If its free daily limit is used up, the agent automatically switches "
                                   "to the next model.")
        if llm.exhausted:
            st.caption("⛔ Daily limit reached today: " + ", ".join(sorted(llm.exhausted)))
    else:
        prefer = None
    max_steps = st.slider("Max tool steps per question", 1, 10, 6,
                          help="Safety limit: stops the agent from looping forever.")
    safe_mode = st.toggle("🛡️ Safe mode (block data changes)", value=True,
                          help="When ON, the add_task tool is blocked. Try asking the agent to add a task "
                               "with it ON and OFF.")
    if ss.toolbox is None:
        ss.toolbox = ToolBox(safe_mode=safe_mode)
    ss.toolbox.safe_mode = safe_mode

    st.subheader("🧰 Tools available")
    for d in TOOL_DECLARATIONS:
        st.markdown(f"{TOOL_ICONS.get(d['name'], '🔧')} `{d['name']}`", help=d["description"])

    st.subheader("🧠 Long-term memory")
    mem = ss.toolbox.recall_memories()["memories"]
    st.caption("\n".join(f"• {m}" for m in mem) if mem else "Nothing saved yet.")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Clear chat", width="stretch", help="Clears short-term memory (conversation)."):
        ss.messages = []
        ss.pop("agent_history", None)
        st.rerun()
    if c2.button("♻️ Reset all", width="stretch", help="Also resets the database and long-term memory."):
        ss.messages, ss.toolbox = [], None
        ss.pop("agent_history", None)
        st.rerun()

# ============================================================================ main
st.title("🤖 AI Agent with Tools")
st.caption("Gemini function calling · Weather, Wikipedia, calculator, SQL database, memory · "
           "Codomax GenAI Internship · Module 05")
st.markdown("**Agent loop:** 🧩 Understand task → 🤔 Decide action → 🔧 Use tool → 📥 Process result → 💬 Respond")

tab_chat, tab_db, tab_about = st.tabs(["💬 Agent", "🗄️ Task database", "ℹ️ Chatbot vs Agent"])

with tab_db:
    st.caption("The agent can read this database with SQL (read-only) and add tasks (only when Safe mode is OFF).")
    tasks = ss.toolbox.query_database("SELECT * FROM tasks ORDER BY due_date")
    st.dataframe(pd.DataFrame(tasks.get("rows", [])), hide_index=True, width="stretch")
    team = ss.toolbox.query_database("SELECT * FROM team")
    st.dataframe(pd.DataFrame(team.get("rows", [])), hide_index=True, width="stretch")

with tab_about:
    st.markdown("""
| | Chatbot (Module 03) | Agent (this app) |
|---|---|---|
| What it does | Generates a reply from training data | **Decides actions**, uses tools, then replies |
| Live data | ❌ | ✅ weather API, Wikipedia, database |
| Exact maths | Often guesses | ✅ calculator tool |
| Multi-step tasks | One reply | ✅ loop: tool → result → next tool → answer |
| State | Chat history | Chat history **+ tool results + database + long-term memory** |
| Risks | Wrong text | Wrong **actions**, so it needs guardrails |

**Guardrails in this app:** read-only SQL (checked in code *and* locked at database level) · Safe mode for
data-changing tools · safe calculator (no `eval`) · step limit · tool results treated as data, not instructions ·
errors returned to the agent instead of crashing · automatic model switching when a free daily limit is hit.
""")

with tab_chat:
    agent = None
    if llm:
        agent = Agent(llm, ss.toolbox, max_steps=max_steps, prefer_model=prefer)
        agent.history = ss.get("agent_history", [])

    for m in ss.messages:
        with st.chat_message(m["role"]):
            for s in m.get("steps", []):
                render_step(s)
            st.markdown(m["content"])
            if m.get("meta"):
                st.caption(m["meta"])

    if agent and not ss.messages:
        st.markdown("##### Try one of these:")
        cols = st.columns(3)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 3].button(ex, width="stretch", key=f"ex{i}"):
                ss.pending = ex
                st.rerun()
    elif not llm:
        st.info("👈 Add your free Gemini API key to start.")

    prompt = st.chat_input("Give the agent a task...", disabled=agent is None) or ss.pop("pending", None)
    if prompt and agent:
        if sum(m["role"] == "user" for m in ss.messages) >= MAX_MESSAGES:
            st.warning("Message limit reached for this session. Click **Clear chat**.")
        else:
            ss.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                live_steps = []

                def on_step(step):
                    d = {"kind": step.kind, "name": step.name, "args": step.args,
                         "result": step.result, "seconds": step.seconds}
                    if step.kind != "answer":
                        live_steps.append(d)
                        render_step(d)

                try:
                    with st.spinner("🤔 Agent is working..."):
                        result = agent.run(prompt, on_step=on_step)
                    ss.agent_history = agent.history
                    tools_used = [s["name"] for s in live_steps if s["kind"] == "tool"]
                    meta = (f"🔧 {len(tools_used)} tool call(s) · 🤖 {result.model_calls} model call(s) · "
                            f"🔢 {result.input_tokens} in / {result.output_tokens} out tokens · "
                            f"model: {', '.join(dict.fromkeys(result.models_used))}")
                    ss.messages.append({"role": "assistant", "content": result.answer or "_(no answer)_",
                                        "steps": live_steps, "meta": meta})
                    st.rerun()
                except Exception as e:
                    ss.messages.pop()
                    show_error(e)

    if ss.messages:
        st.download_button("⬇️ Download agent trace (JSON)", file_name="agent_trace.json", mime="application/json",
                           data=json.dumps(ss.messages, indent=2, default=str))
