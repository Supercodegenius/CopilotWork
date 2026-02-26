# Name Matching Streamlit App

Minimal Streamlit app for fuzzy and phonetic name matching.

Run locally:

```bash
python -m pip install -r name_matching_app/requirements.txt
streamlit run name_matching_app/app.py
```

### Usage
1. Choose mode (single→list or list→list) in the sidebar.
2. Pick a matching algorithm and optional phonetic method.
3. Adjust Top K results and minimum score threshold.
4. Upload one or two CSV files (or use the built‑in sample).
5. Enter a name to query (single mode) or click **Run pairwise**.
6. Preview results and download as CSV if desired.

### Chat Assistant
The sidebar includes an AI chat assistant powered by OpenAI. To enable it, place your OpenAI API key in a file named `openai_key.txt` at the project root (same folder as `app.py`). Once the key is present the assistant will appear and can answer questions about name matching techniques, algorithms, or general tips.

> Tip: If no key is detected the sidebar will show a warning explaining how to add one.

Features:
- Sample names included
- Fuzzy and phonetic scoring options
- Top‑K and threshold filtering
- Exportable CSV output
- Lightweight and easy to extend
