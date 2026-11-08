# Public Contact Search & Enrichment

A Streamlit app that interprets a one-sentence search request, searches the web for matching public professional profiles, and exports only records that pass an independent QA review and source-evidence checks.

## How it works

1. **Intent extraction** — Ollama derives the requested role, organization, location, count, and search query from one sentence.
2. **Web search** — the app runs up to 44 topical and regional metasearch queries across engines supported by `ddgs`, using 12 search-region settings, with an explicit multi-provider fallback for empty queries. It collects up to 100 unique result URLs. Search engines do not index the entire web, so exhaustive coverage is not possible.
3. **Public-page retrieval** — the app fetches at most 24 result pages concurrently, checks each site's `robots.txt`, follows at most three redirects, and caps each page at 1 MB.
4. **Extraction and QA** — all collected result snippets and the full text of fetched pages are passed to Ollama in size-bounded batches, rather than truncating each page to its first 900 characters. A separate QA pass reviews candidates in small batches with evidence excerpts centered on each person's name. Deterministic checks require the name, title, and cited URL to match collected evidence; unsupported fields are removed.
5. **Export** — approved contacts are displayed and can be downloaded as CSV, including a source URL for verification.

The app does not crawl entire websites or bypass access controls. Search engines and website coverage are incomplete, so results can be fewer than requested and should be checked against their linked sources. Public availability does not itself establish permission to use personal data; process only information you are authorised to use.

## Stack and requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- `qwen2.5-coder:3b` downloaded in Ollama
- Streamlit, pandas, the Ollama Python client, DuckDuckGo Search, Requests, and Beautiful Soup

## Run locally

```bash
git clone https://github.com/abby1410/contactEnricher.git
cd contactEnricher

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

ollama pull qwen2.5-coder:3b
streamlit run app.py
```

If Ollama is not already running as a background service, start it in a separate terminal with `ollama serve`. On Windows, create the environment with `py -m venv .venv` and activate it with `.venv\\Scripts\\Activate.ps1`.

## Deploy to Streamlit Community Cloud

1. Connect the public `abby1410/contactEnricher` repository at [share.streamlit.io](https://share.streamlit.io/) and select `main` / `app.py`.
2. In the app's **Settings → Secrets**, set `OLLAMA_HOST` to the URL of an Ollama service reachable from the cloud app, for example:

   ```toml
   OLLAMA_HOST = "https://your-ollama-host.example"
   ```

3. Ensure that service has `qwen2.5-coder:3b` available and permits requests from the deployed app. Do not expose an unauthenticated Ollama server to the public internet.

The app reads `OLLAMA_HOST` from Streamlit secrets (or the environment) as the sidebar default. The default `localhost` address only works when Ollama runs on the same machine as the app.

## Data handling

- Language-model inference uses the Ollama host configured in the sidebar (localhost by default). For hosted deployment, configure a reachable Ollama service; the deployment container's localhost is not the developer's machine.
- Search queries are sent to DuckDuckGo Search, and selected public result pages are requested directly from their hosts.
- Page retrieval is limited and obeys the site's `robots.txt`; pages that block the app are not scraped.
- The app does not fabricate contact details to fill the requested count. Verify results using the included source URLs.
