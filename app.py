import os, datetime as dt
import joblib, pandas as pd, requests, streamlit as st

st.set_page_config(page_title="StarSight", page_icon="⭐")
LABELS = ["Low", "Medium", "High"]

@st.cache_resource
def load_model():
    return joblib.load("notebooks/models/best_model.joblib")

model = load_model()
languages = sorted(model.named_steps["prep"].named_transformers_["cat"].categories_[0])

DEFAULTS = {"language": languages[0], "size_kb": 1000, "forks_count": 10,
            "open_issues_count": 5, "age_days": 500, "days_since_push": 30,
            "topics_count": 3, "has_license": True,
            "description_length": 60, "has_wiki": True, "has_pages": False}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ---- optional: autofill from GitHub ----
def fetch_repo():
    name = st.session_state.get("repo_name", "").strip()
    headers = {}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    try:
        r = requests.get(f"https://api.github.com/repos/{name}", headers=headers, timeout=10)
        r.raise_for_status(); d = r.json()
    except Exception as e:
        st.session_state["fetch_msg"] = f"Could not fetch repo: {e}"
        return
    now = dt.datetime.now(dt.timezone.utc)
    p = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    lang = d.get("language") or "Other"
    st.session_state.update({
        "language": lang if lang in languages else ("Other" if "Other" in languages else languages[0]),
        "size_kb": d["size"], "forks_count": d["forks_count"],
        "open_issues_count": d["open_issues_count"],
        "age_days": (now - p(d["created_at"])).days,
        "days_since_push": (now - p(d["pushed_at"])).days,
        "topics_count": len(d.get("topics", [])),
        "has_license": d.get("license") is not None,
        "description_length": len(d.get("description") or ""),
        "has_wiki": d["has_wiki"], "has_pages": d["has_pages"],
        "fetch_msg": f"Loaded {name}",
    })

st.title("⭐ StarSight: GitHub Popularity Predictor")
st.caption("Predicts Low (10-99), Medium (100-999) or High (1000+) stars from repo details.")

st.text_input("Optional: owner/repo (e.g. pallets/flask)", key="repo_name")
st.button("Fetch from GitHub", on_click=fetch_repo)
if "fetch_msg" in st.session_state:
    st.info(st.session_state["fetch_msg"])

c1, c2 = st.columns(2)
with c1:
    st.selectbox("Language", languages, key="language")
    st.number_input("Size (KB)", min_value=0, key="size_kb")
    st.number_input("Forks", min_value=0, key="forks_count")
    st.number_input("Open issues", min_value=0, key="open_issues_count")
    st.number_input("Age (days)", min_value=0, key="age_days")
with c2:
    st.number_input("Days since last push", min_value=0, key="days_since_push")
    st.number_input("Topics count", min_value=0, key="topics_count")
    st.number_input("Description length (chars)", min_value=0, key="description_length")
    st.checkbox("Has license", key="has_license")
    st.checkbox("Has wiki", key="has_wiki")
    st.checkbox("Has GitHub Pages", key="has_pages")

if st.button("Predict", type="primary"):
    s = st.session_state
    row = pd.DataFrame([{
        "language": s["language"], "size_kb": s["size_kb"], "forks_count": s["forks_count"],
        "open_issues_count": s["open_issues_count"], "age_days": s["age_days"],
        "days_since_push": s["days_since_push"], "topics_count": s["topics_count"],
        "has_license": int(s["has_license"]), "description_length": s["description_length"],
        "has_wiki": int(s["has_wiki"]), "has_pages": int(s["has_pages"]),
    }])
    proba = model.predict_proba(row)[0]
    st.success(f"Predicted popularity: **{LABELS[proba.argmax()]}**")
    st.bar_chart(pd.Series(proba, index=LABELS, name="Probability"))
    for lab, pr in zip(LABELS, proba):
        st.write(f"{lab}: {pr:.1%}")