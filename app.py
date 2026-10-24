import streamlit as st
import time
from crewai import Agent, Crew, Process, Task
from langchain_community.chat_models import ChatOllama

# --- Page Configuration ---
st.set_page_config(
    page_title="Local Multi-Agent Contact Enricher",
    page_icon="🕵️‍♂️",
    layout="wide"
)

st.title("🕵️‍♂️ Local Multi-Agent Contact Enricher & Intelligence System")
st.caption("Powered by 100% Local Hugging Face Models via Ollama & CrewAI")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Local Model Settings")
    fast_model = st.selectbox(
        "Fast Model (Extractor/Copywriter):",
        ["llama3.2:3b", "llama3.2:1b", "qwen2.5:3b"],
        index=0
    )
    smart_model = st.selectbox(
        "Smart Model (Researcher/Profiler):",
        ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"],
        index=0
    )
    ollama_url = st.text_input("Ollama Host URL:", "http://localhost:11434")
    
    st.markdown("---")
    st.markdown("### 🤖 Active Agents")
    st.markdown("1. **Data Extractor** (`llama3.2`)")
    st.markdown("2. **Company Researcher** (`qwen2.5`)")
    st.markdown("3. **Persona Profiler** (`qwen2.5`)")
    st.markdown("4. **Outreach Copywriter** (`llama3.2`)")

# --- Main Input Form ---
st.subheader("📥 Input Raw Lead Data / Notes")
default_text = """Received message from: Sarah Jenkins, VP of Engineering at CloudScale AI.
Contact info found in footer: s.jenkins@cloudscale-ai.fake, San Francisco CA.
Note from last meetup: She mentioned they are struggling to scale their Kubernetes clusters and manage local LLM inference latency cost effectively."""

raw_input = st.text_area("Paste unstructured text, email signatures, or notes below:", value=default_text, height=150)

custom_instructions = st.text_input(
    "💡 Additional Instructions (Optional):", 
    placeholder="e.g., Focus on cost-saving strategies, keep email under 50 words..."
)

# --- Execution Button ---
if st.button("🚀 Run Multi-Agent Pipeline", type="primary", use_container_width=True):
    if not raw_input.strip():
        st.warning("Please provide raw lead data or text first!")
        st.stop()

    # Status containers
    st.markdown("### 🔄 Multi-Agent Workflow Execution")
    progress_bar = st.progress(0)
    
    # Initialize Local LLMs
    try:
        fast_llm = ChatOllama(model=fast_model, base_url=ollama_url)
        smart_llm = ChatOllama(model=smart_model, base_url=ollama_url)
    except Exception as e:
        st.error(f"Failed to connect to Ollama. Make sure Ollama is running locally! Error: {e}")
        st.stop()

    # --- Define Agents ---
    parser_agent = Agent(
        role="Data Extraction Specialist",
        goal="Extract clean structured contact details and core entities from raw unstructured text.",
        backstory="You are a meticulous data cleaner. You look at messy text blocks and pull out names, companies, and roles cleanly.",
        llm=fast_llm,
    )

    researcher_agent = Agent(
        role="Company & Industry Researcher",
        goal="Analyze the target's company context, industry trends, and likely business hurdles.",
        backstory="You are a sharp corporate researcher who excels at connecting a person's role to what their company is trying to achieve.",
        llm=smart_llm,
    )

    profiler_agent = Agent(
        role="Psychographic Profiler",
        goal="Identify the target's professional pain points, motivations, and communication style preferences.",
        backstory="You understand professional psychology. You figure out what keeps executives and managers awake at night.",
        llm=smart_llm,
    )

    copywriter_agent = Agent(
        role="B2B Outreach Copywriter",
        goal="Write a hyper-personalized, concise, and engaging introductory email based on the dossier.",
        backstory="You write high-converting, human-sounding emails that never sound like generic spam.",
        llm=fast_llm,
    )

    # --- Step 1: Agent 1 (Extractor) ---
    with st.status("🕵️‍♂️ **Agent 1:** Extracting structured contact info...", expanded=True) as status1:
        task1 = Task(
            description=f"Extract structured information (Name, Role, Company, Location, Key Clues) from:\n{raw_input}",
            expected_output="Structured breakdown of contact details.",
            agent=parser_agent,
        )
        crew1 = Crew(agents=[parser_agent], tasks=[task1], process=Process.sequential)
        res1 = str(crew1.kickoff())
        status1.update(label="✅ **Agent 1 Done:** Data Extracted!", state="complete", expanded=False)
        st.markdown("**Extracted Data:**")
        st.code(res1)
        progress_bar.progress(25)

    # --- Step 2: Agent 2 (Researcher) ---
    with st.status("🌐 **Agent 2:** Researching company tech stack & hurdles...", expanded=True) as status2:
        task2 = Task(
            description=f"Based on this extracted context:\n{res1}\nAnalyze company technical hurdles, market focus, and pain points.",
            expected_output="Brief technical background and industry analysis.",
            agent=researcher_agent,
        )
        crew2 = Crew(agents=[researcher_agent], tasks=[task2], process=Process.sequential)
        res2 = str(crew2.kickoff())
        status2.update(label="✅ **Agent 2 Done:** Industry Research Completed!", state="complete", expanded=False)
        st.markdown("**Research Notes:**")
        st.write(res2)
        progress_bar.progress(50)

    # --- Step 3: Agent 3 (Profiler) ---
    with st.status("💼 **Agent 3:** Building psychographic buyer profile...", expanded=True) as status3:
        task3 = Task(
            description=f"Synthesize this research:\n{res2}\nIdentify key motivations, risks, and goals for this executive.",
            expected_output="Target persona summary with main pain points.",
            agent=profiler_agent,
        )
        crew3 = Crew(agents=[profiler_agent], tasks=[task3], process=Process.sequential)
        res3 = str(crew3.kickoff())
        status3.update(label="✅ **Agent 3 Done:** Persona Profile Built!", state="complete", expanded=False)
        st.markdown("**Persona Profile:**")
        st.write(res3)
        progress_bar.progress(75)

    # --- Step 4: Agent 4 (Copywriter) ---
    with st.status("✉️ **Agent 4:** Drafting tailored outreach email...", expanded=True) as status4:
        instr = f"\nAdditional Instructions: {custom_instructions}" if custom_instructions else ""
        task4 = Task(
            description=f"Using this persona profile:\n{res3}\nDraft a concise, high-converting outreach email.{instr}",
            expected_output="Complete personalized email ready to send.",
            agent=copywriter_agent,
        )
        crew4 = Crew(agents=[copywriter_agent], tasks=[task4], process=Process.sequential)
        res4 = str(crew4.kickoff())
        status4.update(label="✅ **Agent 4 Done:** Outreach Drafted!", state="complete", expanded=False)
        progress_bar.progress(100)

    # --- Final Result Dashboard Display ---
    st.success("🎉 Enrichment Pipeline Completed Successfully!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Final Intelligence Dossier")
        st.markdown(f"**Research Summary:**\n{res2}")
        st.markdown(f"**Key Pain Points:**\n{res3}")
        
    with col2:
        st.subheader("✉️ Personalized Outreach Email")
        st.text_area("Ready to Send:", value=res4, height=280)
        st.button("📋 Copy Email", on_click=lambda: st.write("Copied!"))