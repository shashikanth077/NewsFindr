"""NewsFindr web app (Streamlit). Run with:  streamlit run app.py

A thin UI over newsfindr.core: the same SQL agent -> query expansion -> DuckDuckGo -> filtering ->
LLM summarisation workflow as the notebook, with each stage made visible.
"""
import os

import pandas as pd
import streamlit as st

from newsfindr import core

st.set_page_config(page_title="NewsFindr", page_icon="📰", layout="wide")

DEMO_CUSTOMERS = {
    "John": ("john@example.com", "What are the latest developments in Generative AI?"),
    "Sarah": ("sarah@example.com", "What are the latest AI applications in healthcare?"),
    "David": ("david@example.com", "Find recent breakthroughs in Large Language Models."),
}


def secret_key() -> str:
    """Groq key from the environment or Streamlit secrets; never hard-coded."""
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return ""


def pick_demo(name: str) -> None:
    st.session_state["email"], st.session_state["request"] = DEMO_CUSTOMERS[name]


def fmt_date(article) -> str:
    return article.published_at.strftime("%d %b %Y, %H:%M UTC") if article.published_at else "date unavailable"


def articles_df(articles, extra=()) -> pd.DataFrame:
    rows = [{"title": a.title, "source": a.source, "published": fmt_date(a), "url": a.url,
             **{k: getattr(a, k) for k in extra}} for a in articles]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📰 NewsFindr")
    st.caption("Personalized news with an SQL agent, Groq LLM and live DuckDuckGo search.")

    api_key = st.text_input("Groq API key", value=secret_key(), type="password",
                            help="Kept in this session only. Or set the GROQ_API_KEY environment variable.")
    model = st.text_input("Model override (optional)", value=os.environ.get("GROQ_MODEL", ""),
                          placeholder="llama-3.3-70b-versatile")

    st.divider()
    st.subheader("Try a demo customer")
    for name in DEMO_CUSTOMERS:
        st.button(f"{name} · {DEMO_CUSTOMERS[name][0]}", key=f"demo_{name}", on_click=pick_demo, args=(name,),
                  width="stretch")

    st.divider()
    st.subheader("Filtering settings")
    st.write(f"- Trusted-source allowlist: **{len(core.TRUSTED_SOURCES)}** domains")
    st.write(f"- Max article age: **{core.MAX_ARTICLE_AGE_DAYS} days** (widens to 30 if needed)")
    st.write(f"- Minimum keyword relevance: **{core.MIN_RELEVANCE}**")
    st.caption("The allowlist reflects source reputation. It does **not** mean an article is fact-checked.")

# ----------------------------------------------------------------------------- main form
st.title("Find the news that matters to you")
st.write("Enter a registered customer email and, optionally, what you want to know. "
         "Everything below is retrieved live; nothing is generated from the model's memory.")

with st.form("search"):
    c1, c2 = st.columns([1, 2])
    email = c1.text_input("Customer email", key="email", placeholder="john@example.com")
    request = c2.text_input("What are you looking for? (optional)", key="request",
                            placeholder="What are the latest developments in Generative AI?")
    submitted = st.form_submit_button("Find my news", type="primary")

if submitted:
    try:
        core.EmailRequest(email=email)  # reject malformed input before touching the LLM or database
        email_ok = True
    except Exception:
        email_ok = False
    if not email.strip():
        st.error("Enter a customer email.")
    elif not email_ok:
        st.session_state["run"] = None
        st.warning("That doesn't look like a valid email address, so the database was not queried.")
    elif not api_key.strip():
        st.error("Enter your Groq API key in the sidebar (or set GROQ_API_KEY) to run the agent.")
    else:
        try:
            with st.spinner("Starting the Groq model and opening the customer database…"):
                model_used = core.init_runtime(api_key=api_key, model=model.strip() or None)
        except Exception as exc:
            st.error(f"Could not initialise Groq ({type(exc).__name__}): {str(exc)[:300]}")
        else:
            with st.status(f"Running the NewsFindr workflow with {model_used}…", expanded=True) as status:
                def on_stage(stage: str, detail: str) -> None:
                    st.write(f"**{stage}** — {detail}")

                run = core.run_newsfindr(email, request or None, show_agent_trace=False, on_stage=on_stage)
                final = {"ok": "complete", "error": "error", "search_error": "error"}.get(run.status, "complete")
                status.update(label=f"Finished: {run.status}", state=final, expanded=False)
            st.session_state["run"] = run

# ----------------------------------------------------------------------------- results
run = st.session_state.get("run")
if run is not None:
    if run.status == "ok":
        st.success(run.message)
    elif run.status in ("not_found", "invalid_email"):
        st.warning(run.message)
    else:
        st.error(run.message or f"Run ended with status: {run.status}")

    tab_recs, tab_how, tab_trace, tab_rejected = st.tabs(
        ["📰 Recommendations", "🔎 How it worked", "🤖 SQL agent trace", "🗑️ Filtered out"])

    with tab_recs:
        if not run.recommendations:
            st.info("No recommendations to show for this run.")
        for r in run.recommendations:
            a = r.article
            with st.container(border=True):
                st.markdown(f"### {r.rank}. [{a.title}]({a.url})")
                st.caption(f"**{a.source}** ({a.domain}) · {fmt_date(a)} · LLM relevance {r.llm_relevance}/10")
                st.markdown(f"**Summary.** {r.summary}")
                st.markdown(f"**Why this matches you.** {r.why_relevant}")
                st.caption(r.content_basis)
                if r.unsupported_numbers:
                    st.warning(f"Numbers not found in the source text: {', '.join(r.unsupported_numbers)}")
                st.link_button("Open original article", a.url)

    with tab_how:
        if run.profile:
            st.subheader("Verified customer profile")
            st.dataframe(pd.DataFrame([run.profile.model_dump()]), hide_index=True, width="stretch")
            for note in run.profile_notes:
                st.caption(f"• {note}")
        if run.queries:
            st.subheader(f"Expanded search queries ({run.query_method})")
            st.dataframe(pd.DataFrame([q.model_dump() for q in run.queries]), hide_index=True,
                         width="stretch")
        if run.funnel:
            st.subheader("Filtering funnel")
            funnel_df = pd.DataFrame({"articles": run.funnel.values()}, index=list(run.funnel.keys()))
            left, right = st.columns([2, 1])
            left.bar_chart(funnel_df)
            right.dataframe(funnel_df, width="stretch")
        if run.ranked:
            st.subheader("Ranked candidates")
            st.dataframe(articles_df(run.ranked, ("relevance_score", "recency_score", "credibility_score",
                                                  "rank_score")), hide_index=True, width="stretch")
        if run.review:
            st.subheader("LLM relevance review")
            st.dataframe(pd.DataFrame(run.review), hide_index=True, width="stretch")
        st.subheader("Stage log")
        st.dataframe(pd.DataFrame(run.stage_log), hide_index=True, width="stretch")

    with tab_trace:
        if run.agent_trace:
            st.caption("Each `calls …` line is a decision the SQL agent (an LLM) made on its own.")
            st.code("\n".join(run.agent_trace), language="text")
        else:
            st.info("No agent trace for this run (the agent did not run or did not complete).")

    with tab_rejected:
        if run.rejected:
            st.dataframe(articles_df(run.rejected, ("rejection_reason",)), hide_index=True,
                         width="stretch")
        else:
            st.info("No articles were filtered out.")
