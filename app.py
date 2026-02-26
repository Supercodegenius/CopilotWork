import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import jellyfish
from io import StringIO
import os
from openai import OpenAI

# load OpenAI key if available
api_key = None
key_path = "openai_key.txt"
if os.path.exists(key_path):
    with open(key_path) as f:
        api_key = f.read().strip()
client = OpenAI(api_key=api_key) if api_key else None

st.set_page_config(page_title="Name Matching", layout="wide")

# small custom styling to mimic reference app
st.markdown(
    """
    <style>
    .stApp { background-color: #fafafa; }
    .title {font-size:2.5rem; font-weight:bold;}
    .sidebar .stButton>button {width:100%;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='title'>Name Matching</h1>", unsafe_allow_html=True)
st.markdown("**Fuzzy‑ and phonetic‑matching dashboard** — upload names or use the sample, then enter a query or run pairwise comparisons.")

with st.sidebar:
    st.header("Options")
    mode = st.radio("Mode", ["Single → List", "List → List"], index=0)
    algorithm = st.selectbox("Algorithm", [
        "Levenshtein ratio",
        "Partial ratio",
        "Token sort ratio",
        "Jaro-Winkler",
        "Phonetic"
    ])
    # show phonetic method only when selected
    phonetic_method = None
    if algorithm == "Phonetic":
        phonetic_method = st.selectbox("Phonetic method", ["soundex", "metaphone"], index=0)
    top_k = st.slider("Top K matches", 1, 50, 10)
    threshold = st.slider("Minimum score", 0, 100, 0)

uploaded = st.file_uploader("Upload CSV with a single column of names (header optional)")
use_sample = st.checkbox("Use sample names (provided)", value=not bool(uploaded))

@st.cache_data

def load_names(uploaded_file, use_sample):
    """Return unique list of names from uploaded file or sample CSV."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            names = df.iloc[:,0].astype(str).str.strip()
        except Exception:
            uploaded_file.seek(0)
            txt = StringIO(uploaded_file.getvalue().decode('utf-8'))
            df = pd.read_csv(txt, header=None)
            names = df.iloc[:,0].astype(str).str.strip()
        return names.dropna().unique().tolist()
    if use_sample:
        sample = pd.read_csv("name_matching_app/sample_names.csv", header=None).iloc[:,0].astype(str).str.strip()
        return sample.dropna().unique().tolist()
    return []

names = load_names(uploaded, use_sample)
if not names:
    st.warning("No names loaded — upload a CSV or enable the sample names checkbox.")
else:
    with st.expander("Preview loaded names", expanded=False):
        st.write(pd.DataFrame(names, columns=["name"]).head(20))
        st.caption("Showing first 20 of {} names".format(len(names)))

def phonetic_match(query, candidates, method='soundex'):
    qcode = jellyfish.soundex(query) if method=='soundex' else jellyfish.metaphone(query)
    rows = []
    for c in candidates:
        ccode = jellyfish.soundex(c) if method=='soundex' else jellyfish.metaphone(c)
        score = 100 if qcode and ccode and qcode==ccode else 0
        rows.append((c, score, ccode))
    return sorted(rows, key=lambda x: x[1], reverse=True)

def jaro_scores(query, candidates):
    rows = [(c, jellyfish.jaro_winkler_similarity(query, c)*100) for c in candidates]
    return sorted(rows, key=lambda x: x[1], reverse=True)

def run_single(query, candidates, algorithm, top_k, threshold, phonetic_method=None):
    if algorithm == 'Levenshtein ratio':
        matches = process.extract(query, candidates, scorer=fuzz.ratio, limit=top_k)
    elif algorithm == 'Partial ratio':
        matches = process.extract(query, candidates, scorer=fuzz.partial_ratio, limit=top_k)
    elif algorithm == 'Token sort ratio':
        matches = process.extract(query, candidates, scorer=fuzz.token_sort_ratio, limit=top_k)
    elif algorithm == 'Jaro-Winkler':
        matches = jaro_scores(query, candidates)[:top_k]
    elif algorithm == 'Phonetic':
        method = phonetic_method or 'soundex'
        matches = phonetic_match(query, candidates, method=method)[:top_k]
    else:
        matches = []

    # build dataframe; phonetic adds code column
    if algorithm == 'Jaro-Winkler' or algorithm == 'Phonetic':
        if matches and len(matches[0]) == 3:
            df = pd.DataFrame(matches, columns=['candidate', 'score', 'code'])
        else:
            df = pd.DataFrame(matches, columns=['candidate', 'score'])
    else:
        df = pd.DataFrame(matches, columns=['candidate', 'score', 'index'])
        df = df[['candidate', 'score']]

    df = df[df['score'] >= threshold]
    return df

if mode == 'Single → List':
    col1, col2 = st.columns([2,3])
    with col1:
        query = st.text_input("Name to match", placeholder="e.g. Jonathon Doe")
        if st.button("Search"):
            if not query:
                st.error("Please enter a name to match.")
            elif not names:
                st.error("No candidate names available.")
            else:
                with st.spinner("Finding matches..."):
                    df = run_single(query, names, algorithm, top_k, threshold, phonetic_method)
                st.subheader(f"Matches for: {query}")
                st.dataframe(df.style.highlight_max(axis=0))
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download results", data=csv, file_name="matches.csv")
    with col2:
        st.markdown("**Instructions**\n\n1. Upload a list of names or use the sample.\n2. Choose algorithm and parameters in sidebar.\n3. Enter a query and press **Search**.\n4. Download matches if needed.")

else:  # List → List
    st.markdown("### Pairwise matching\nUpload two CSVs (one for left, one for right) or use the same list for both sides. Results show top‑K matches per left item.")
    uploaded_b = st.file_uploader("Second list CSV (right side)", key="b")
    use_same = st.checkbox("Use same list for both sides", value=True)

    left_names = names
    if uploaded_b and not use_same:
        left_names = load_names(uploaded_b, False)
    right_names = names if use_same or not uploaded_b else load_names(uploaded_b, False)

    if st.button("Run pairwise"):
        if not left_names or not right_names:
            st.error("Both lists must be available.")
        else:
            results = []
            for q in left_names:
                df = run_single(q, right_names, algorithm, top_k, threshold, phonetic_method)
                for _, r in df.iterrows():
                    results.append({'left': q, 'right': r['candidate'], 'score': r['score']})
            out = pd.DataFrame(results)
            st.dataframe(out)
            st.download_button("Download pairwise results", data=out.to_csv(index=False).encode('utf-8'), file_name='pairwise_matches.csv')

st.markdown("---")
st.markdown("Built with Streamlit — supports `rapidfuzz` and `jellyfish` scorers.")

# --- Chat assistant ---------------------------------------------------------
if client is None:
    st.sidebar.warning("No OpenAI API key found. Place your key in openai_key.txt to enable chat assistant.")
else:
    st.sidebar.header("🗣️ Chat Assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.sidebar.text_input("Ask the assistant:", key="chat_input")
    if st.sidebar.button("Send", key="send_chat"):
        if user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            messages = [
                {"role": "system", "content": "You are a helpful assistant specialized in name-matching and fuzzy string comparisons."}
            ] + st.session_state.chat_history
            try:
                resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
                reply = resp.choices[0].message.content
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {e}"})
    # display history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.sidebar.markdown(f"**You:** {msg['content']}")
        else:
            st.sidebar.markdown(f"**Bot:** {msg['content']}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Tip: the assistant can help explain algorithms or suggest search strategies.*")
