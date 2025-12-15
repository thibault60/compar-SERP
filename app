import pandas as pd
import streamlit as st
from urllib.parse import urlparse

from serpapi_client import serpapi_google_search

st.set_page_config(page_title="SERP → Top 10 (SerpApi)", layout="wide")

st.title("Extraction SERP (SerpApi) → Top 10 concurrents")
st.caption("Étape 1 : saisir une requête, récupérer les résultats organiques, exporter en CSV.")

with st.sidebar:
    st.header("Paramètres SERP")
    gl = st.selectbox("Pays (gl)", ["fr", "be", "ch", "ca", "us", "uk", "de", "es", "it"], index=0)
    hl = st.selectbox("Langue (hl)", ["fr", "en", "de", "es", "it"], index=0)
    google_domain = st.text_input("Google domain", value="google.fr")
    location = st.text_input("Location (optionnel)", value="France")
    num = st.number_input("Nb résultats (num)", min_value=1, max_value=100, value=10, step=1)
    no_cache = st.checkbox("Forcer no_cache", value=False)

q = st.text_input("Requête (mot-clé)", value="circuit all inclusive", placeholder="ex: circuit autotour")

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""

colA, colB = st.columns([1, 1])
with colA:
    run = st.button("Lancer la recherche", type="primary")
with colB:
    st.info("Astuce : garde gl=fr / hl=fr pour une SERP FR comparable.")

if run:
    if not q.strip():
        st.error("Merci de saisir une requête.")
        st.stop()

    with st.spinner("Appel SerpApi…"):
        data = serpapi_google_search(
            q=q.strip(),
            gl=gl,
            hl=hl,
            num=int(num),
            location=location.strip() if location.strip() else None,
            google_domain=google_domain.strip() if google_domain.strip() else "google.fr",
            no_cache=no_cache,
        )

    organic = data.get("organic_results", []) or []
    if not organic:
        st.warning("Aucun résultat organique retourné (ou champ organic_results absent).")
        st.json(data)
        st.stop()

    rows = []
    for r in organic:
        link = r.get("link") or ""
        rows.append(
            {
                "position": r.get("position"),
                "domain": domain_of(link),
                "title_serp": r.get("title"),
                "url": link,
                "snippet": r.get("snippet"),
            }
        )

    df = pd.DataFrame(rows).sort_values("position")
    st.subheader("Top résultats organiques")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Télécharger CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"serp_top_{gl}_{hl}_{q.strip().replace(' ', '_')}.csv",
        mime="text/csv",
    )

    # (optionnel) Une version “groupée par domaine” utile pour ta cellule Concurrents
    st.subheader("Groupé par domaine (1ère URL trouvée)")
    g = df.dropna(subset=["domain"]).groupby("domain", as_index=False).first()
    st.dataframe(g[["domain", "url", "title_serp"]], use_container_width=True)
