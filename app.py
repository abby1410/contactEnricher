import streamlit as st
import pandas as pd
import json
import os
import io
import csv
import re
import contextlib
import ollama
from duckduckgo_search import DDGS

# --- Page Configuration ---
st.set_page_config(
    page_title="Resilient Contact Miner & Lead Enricher",
    page_icon="🕵️‍♂️",
    layout="wide"
)

st.title("🕵️‍♂️ Resilient Multi-Agent Contact Miner & Lead Enricher")
st.caption("Powered locally by Ollama (`qwen2.5-coder:3b`) with resilient web harvesting, domain pattern synthesis, and deterministic CSV serialization.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Settings & Parameters")
    selected_model = st.selectbox("Select Local LLM:", ["qwen2.5-coder:3b"], index=0)
    ollama_url = st.text_input("Ollama Host URL:", "http://localhost:11434")
    target_count = st.number_input("Maximum Contacts:", min_value=1, max_value=250, value=50, step=1)

os.environ["OPENAI_API_BASE"] = ollama_url
os.environ["OPENAI_API_KEY"] = "ollama"

# --- Resilient Web Harvester ---
def harvest_live_data(query_text: str, max_results: int = 20):
    """Search the web using the criteria supplied in the UI."""
    variations = [
        f"{query_text} LinkedIn profile",
        f"{query_text} team staff directory",
        f"{query_text} professional profile"
    ]
    
    harvested_snippets = []
    with DDGS() as ddgs:
        for q in variations:
            try:
                results = list(ddgs.text(q, max_results=5))
                for r in results:
                    snippet = f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}\n"
                    if snippet not in harvested_snippets:
                        harvested_snippets.append(snippet)
            except Exception:
                continue

    return "\n---\n".join(harvested_snippets)

# --- User Input UI ---
st.subheader("📝 Choose Your Search Criteria")
with st.form("lead_search_form"):
    role = st.text_input("Role or type of contact", placeholder="e.g. Recruiter, product manager, dentist")
    company = st.text_input("Company or organization", placeholder="e.g. Google")
    location = st.text_input("Location", placeholder="e.g. Hyderabad, India")
    additional_terms = st.text_input(
        "Additional search terms",
        placeholder="e.g. LinkedIn profile, hiring team, public contact details"
    )
    submitted = st.form_submit_button(
        "🚀 Run Lead Extraction Pipeline",
        type="primary",
        use_container_width=True
    )

if submitted:
    search_terms = [role, company, location, additional_terms]
    user_query = " ".join(term.strip() for term in search_terms if term.strip())
    if not user_query:
        st.warning("Enter at least one search criterion.")
        st.stop()

    meta = {
        "role": role.strip() or "Not specified",
        "company": company.strip() or "Not specified",
        "location": location.strip() or "Not specified",
    }

    log_container = st.expander("🔍 Live Agent Execution Logs", expanded=True)

    # --- STEP 1: Intent Extraction ---
    with log_container:
        st.write(f"🔌 Connected to Ollama using model: `{selected_model}`")
        st.write("🎯 **Agent 1 (Intent Specialist):** Search criteria:")
        st.json(meta)
        st.write(f"Search query: `{user_query}`")

    # --- STEP 2: Live Web Harvesting ---
    with st.status("🌐 **Agent 2 (Web Harvester):** Gathering live search web snippets...", expanded=True) as status2:
        web_snippets = harvest_live_data(user_query)
        if web_snippets:
            status2.update(label=f"✅ **Agent 2 Done:** Successfully retrieved live web snippets!", state="complete", expanded=False)
        else:
            status2.update(label="⚠️ **Agent 2 Notice:** No public web snippets were retrieved; only search-supported contacts will be included.", state="complete", expanded=False)

    # --- STEP 3: Contact Extraction ---
    with st.status(f"⛏️ **Agent 3 (Enrichment Engine):** Extracting up to {target_count} matching contacts...", expanded=True) as status3:

        prompt = f"""
        Target Parameters:
        - Company: {meta['company']}
        - Role: {meta['role']}
        - Location: {meta['location']}
        - Full search query: {user_query}

        Web Data Context:
        ---
        {web_snippets[:6000]}
        ---

        Tasks:
        Extract up to {target_count} matching contacts only when their names are explicitly present in the web data context.
        Include only information supported by that context. Do not invent names, companies, locations, email addresses, or phone numbers.
        Use "N/A" for email addresses or phone numbers not present in the context. Return fewer contacts if fewer are supported.

        OUTPUT REQUIREMENTS:
        Output strictly a raw JSON list of objects with keys:
        "Name", "Title", "Company", "Location", "Email", "Phone", "Status"
        Do not output any prose, commentary, or markdown text outside the JSON array.
        """

        try:
            llm_response = ollama.chat(
                model=selected_model,
                messages=[{'role': 'user', 'content': prompt}]
            )['message']['content']
        except Exception as e:
            st.error(f"Could not generate contacts with Ollama: {e}")
            st.stop()

        status3.update(label="✅ **Agent 3 Done:** Lead generation complete!", state="complete", expanded=False)

    # --- STEP 4: QA Validation & Native Python CSV Export ---
    st.subheader("📊 Final Enriched Contact Database")

    fieldnames = ["Name", "Title", "Company", "Location", "Email", "Phone", "Status"]
    parsed_records = []
    try:
        match = re.search(r'\[.*\]', llm_response, re.DOTALL)
        if match:
            raw_records = json.loads(match.group(0))
            if not isinstance(raw_records, list):
                raise ValueError("The model response must be a JSON list.")
            for rec in raw_records[:int(target_count)]:
                if not isinstance(rec, dict):
                    continue
                name = str(rec.get("Name", ""))
                if name.strip().lower() in ["none", "null", ""]:
                    continue
                normalized = {key: rec.get(key, "N/A") for key in fieldnames}
                if company.strip():
                    normalized["Company"] = company.strip()
                if location.strip():
                    normalized["Location"] = location.strip()
                normalized["Status"] = "EXTRACTED FROM SEARCH RESULTS"
                parsed_records.append(normalized)
    except Exception as e:
        st.error(f"QA Parsing Error: {e}")

    # Python Native CSV Serialization
    csv_filename = "contacts_output.csv"

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in parsed_records:
        writer.writerow({k: row.get(k, "N/A") for k in fieldnames})

    csv_data = csv_buffer.getvalue()

    with open(csv_filename, "w", encoding="utf-8") as f:
        f.write(csv_data)

    st.success(f"🎉 Found **{len(parsed_records)} matching contacts** and saved them to `{os.path.abspath(csv_filename)}`.")
    if not parsed_records:
        st.info("No matching named contacts were found in the search results. Try broader criteria or different additional terms.")

    df = pd.read_csv(io.StringIO(csv_data))
    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="📥 Download contacts_output.csv",
        data=csv_data,
        file_name="contacts_output.csv",
        mime="text/csv"
    )