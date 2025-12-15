import os
import requests

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"


def get_api_key() -> str:
    # Priorité Streamlit secrets, sinon variable d'env
    try:
        import streamlit as st  # local import pour éviter dépendance circulaire
        key = st.secrets.get("SERPAPI_API_KEY", None)
        if key:
            return key
    except Exception:
        pass

    key = os.getenv("SERPAPI_API_KEY")
    if not key:
        raise RuntimeError(
            "Clé SerpApi manquante. Définis SERPAPI_API_KEY (env) ou .streamlit/secrets.toml"
        )
    return key


def serpapi_google_search(
    q: str,
    gl: str = "fr",
    hl: str = "fr",
    num: int = 10,
    location: str | None = None,
    google_domain: str = "google.fr",
    no_cache: bool = False,
    timeout: int = 30,
) -> dict:
    api_key = get_api_key()

    params = {
        "engine": "google",
        "q": q,
        "api_key": api_key,
        "gl": gl,
        "hl": hl,
        "num": num,
        "google_domain": google_domain,
    }
    if location:
        params["location"] = location
    if no_cache:
        params["no_cache"] = "true"

    r = requests.get(SERPAPI_ENDPOINT, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    # SerpApi renvoie parfois un champ "error" si problème
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"SerpApi error: {data['error']}")

    return data
