# Local Multi-Agent Contact Enricher 🕵️

A Streamlit app that turns unstructured lead notes, email signatures, and pasted contact context into a contact brief and a tailored B2B outreach email.

It uses **CrewAI** to coordinate four specialist agents and **Ollama** to run language models locally.

## How it works

1. **Data extraction** — identifies details such as name, role, company, location, and useful context.
2. **Company and industry analysis** — generates an analysis from the extracted information.
3. **Persona profiling** — summarizes likely professional goals, motivations, and pain points.
4. **Outreach writing** — drafts an introductory email using the preceding outputs.

The Streamlit interface displays intermediate outputs alongside the final intelligence brief and email. The sidebar lets you select separate models for faster extraction/copywriting and more capable analysis/profiling.

## Stack

Python · Streamlit · CrewAI · LangChain community Ollama integration · Ollama

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- The models you select downloaded into your local Ollama installation

## Run locally

```bash
git clone https://github.com/abby1410/contactEnricher.git
cd contactEnricher

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install streamlit crewai langchain-community

ollama pull llama3.2:3b
ollama pull qwen2.5:7b

streamlit run app.py
```

If Ollama is not already running as a background service, start it in a separate terminal with `ollama serve`. On Windows, create the environment with `py -m venv .venv` and activate it with `.venv\\Scripts\\Activate.ps1`.

The app's model dropdown includes other options; download whichever model you select before running the pipeline.

## Privacy and research limitations

- Inference uses the Ollama host configured in the sidebar (localhost by default).
- The current app has no web-search or CRM integration. Its company/industry analysis is generated from the pasted context and the model's existing knowledge. Verify important claims independently.
- Only submit personal or business information you are authorised to process.

## Current limitations

- The four CrewAI steps currently run as separate sequential crews.
- A sample lead is prefilled in the input box; replace it with your own test data.
- The “Copy Email” button currently displays a confirmation message; it does not yet copy text to the system clipboard.
- There is no pinned dependency file or automated test suite yet.

## Roadmap

- [ ] Add structured JSON output and editable contact fields
- [ ] Add CSV upload and batch enrichment
- [ ] Add evidence/source tracking for company research
- [ ] Implement real clipboard support and export to Markdown/CSV
- [ ] Add configurable prompts and model settings
- [ ] Add tests and a pinned dependency file

## License

No license file is currently included. All rights remain with the repository owner unless a license is added.
