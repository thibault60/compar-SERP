from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


# =========================================================
# CONFIG
# =========================================================
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Critères "client-facing" — focus sur les MANQUANTS
VISIBLE_FEATURES = [
    ("H1", lambda fp: bool(fp.get("headings", {}).get("h1"))),
    ("Filtres visibles", lambda fp: bool(fp.get("modules", {}).get("filters_present"))),
    ("Avis", lambda fp: bool(fp.get("modules", {}).get("reviews_present"))),
    ("Q&A", lambda fp: bool(fp.get("modules", {}).get("faq_visible"))),
    ("Encart Contact", lambda fp: bool(fp.get("modules", {}).get("contact_present"))),
    ("Articles", lambda fp: bool(fp.get("modules", {}).get("articles_present"))),
]

META_FEATURES = [
    ("Schema ItemList", lambda fp: "ItemList" in (fp.get("structured_data", {}).get("jsonld_types") or [])),
    ("Schema FAQPage", lambda fp: "FAQPage" in (fp.get("structured_data", {}).get("jsonld_types") or [])),
    ("Schema TravelAgency", lambda fp: "TravelAgency" in (fp.get("structured_data", {}).get("jsonld_types") or [])),
    ("Schema ListItem", lambda fp: "ListItem" in (fp.get("structured_data", {}).get("jsonld_types") or [])),
    ("Schema AggregateRating", lambda fp: "AggregateRating" in (fp.get("structured_data", {}).get("jsonld_types") or [])),
]


# =========================================================
# UTILS
# =========================================================
def clean_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def score_features(fp: dict, features: list[tuple[str, callable]]) -> tuple[list[str], list[str]]:
    present, missing = [], []
    for label, fn in features:
        ok = False
        try:
            ok = bool(fn(fp))
        except Exception:
            ok = False
        (present if ok else missing).append(label)
    return present, missing


# =========================================================
# SERPAPI
# =========================================================
def get_serpapi_key() -> str:
    key = None
    try:
        key = st.secrets.get("SERPAPI_API_KEY", None)
    except Exception:
        pass
    if not key:
        key = os.getenv("SERPAPI_API_KEY")
    if not key:
        raise RuntimeError("Clé SerpApi manquante. Mets SERPAPI_API_KEY (env) ou .streamlit/secrets.toml")
    return key


@st.cache_data(ttl=3600, show_spinner=False)
def serpapi_google_search(q: str, gl: str, hl: str, num: int, location: Optional[str], google_domain: str) -> dict:
    params = {
        "engine": "google",
        "q": q,
        "gl": gl,
        "hl": hl,
        "num": num,
        "google_domain": google_domain,
        "api_key": get_serpapi_key(),
    }
    if location:
        params["location"] = location

    r = requests.get(SERPAPI_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"SerpApi error: {data['error']}")
    return data


# =========================================================
# FETCHERS
# =========================================================
@dataclass
class FetchResult:
    url: str
    final_url: Optional[str]
    status: Optional[int]
    html: Optional[str]
    headers: Dict[str, str]
    elapsed_ms: Optional[int]
    error: Optional[str]
    mode: str  # static|rendered


def fetch_static(url: str, timeout: int = 20) -> FetchResult:
    t0 = time.time()
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": DEFAULT_UA, "Accept-Language": "fr-FR,fr;q=0.9"},
            allow_redirects=True,
        )
        return FetchResult(
            url=url,
            final_url=r.url,
            status=r.status_code,
            html=r.text if r.text else None,
            headers={k.lower(): v for k, v in (r.headers or {}).items()},
            elapsed_ms=int((time.time() - t0) * 1000),
            error=None,
            mode="static",
        )
    except Exception as e:
        return FetchResult(
            url=url,
            final_url=None,
            status=None,
            html=None,
            headers={},
            elapsed_ms=int((time.time() - t0) * 1000),
            error=str(e),
            mode="static",
        )


def fetch_rendered_playwright(url: str, timeout_ms: int = 30000, headless: bool = True) -> FetchResult:
    t0 = time.time()
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(locale="fr-FR")
            page = context.new_page()

            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            # Best effort : stabilisation
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass

            # Best effort : cookies
            for selector in [
                "button#onetrust-accept-btn-handler",
                "button:has-text('Tout accepter')",
                "button:has-text('Accepter')",
                "button[aria-label*='Accepter']",
            ]:
                try:
                    if page.locator(selector).first.is_visible(timeout=800):
                        page.locator(selector).first.click(timeout=800)
                        break
                except Exception:
                    continue

            html = page.content()
            final_url = page.url

            context.close()
            browser.close()

        return FetchResult(
            url=url,
            final_url=final_url,
            status=200,
            html=html,
            headers={},
            elapsed_ms=int((time.time() - t0) * 1000),
            error=None,
            mode="rendered",
        )
    except Exception as e:
        return FetchResult(
            url=url,
            final_url=None,
            status=None,
            html=None,
            headers={},
            elapsed_ms=int((time.time() - t0) * 1000),
            error=str(e),
            mode="rendered",
        )


# =========================================================
# DETECTOR (need JS?)
# =========================================================
def needs_js(html: Optional[str]) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    if not html or len(html) < 2000:
        reasons.append("html_trop_court")
        return True, reasons

    soup = BeautifulSoup(html, "lxml")

    if soup.select_one("#__next") or soup.select_one("#__NUXT__"):
        reasons.append("spa_wrapper_detecte")

    h1 = soup.find("h1")
    if not h1 or not (h1.get_text(strip=True) or ""):
        reasons.append("h1_absent_en_statique")

    text = soup.get_text(" ", strip=True)
    wc = len(text.split())
    if wc < 200:
        reasons.append(f"word_count_faible:{wc}")

    return (len(reasons) >= 2), reasons


# =========================================================
# EXTRACTOR (fingerprint)
# =========================================================
def extract_fingerprint_from_html(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    # --- Meta
    title = clean_text(soup.title.get_text()) if soup.title else None

    meta_desc = None
    md = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if md and md.get("content"):
        meta_desc = clean_text(md.get("content"))

    canonical = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
    canonical_href = clean_text(canonical.get("href")) if canonical else None

    # --- Headings
    h1_tag = soup.find("h1")
    h1 = clean_text(h1_tag.get_text()) if h1_tag else None

    toc_present = bool(
        soup.select_one("[id*='toc' i], [class*='toc' i], [class*='sommaire' i]")
    )

    # --- Modules (visible)
    header = soup.find("header")
    header_present = bool(header)

    breadcrumb_present = bool(
        soup.select_one("[aria-label*='breadcrumb' i], nav.breadcrumb, .breadcrumb, [class*='breadcrumb']")
    )

    filters_present = bool(
        soup.select_one("[class*='filter' i], [id*='filter' i], [aria-label*='filtre' i], [data-testid*='filter' i]")
    )

    # Listing visible (heuristique) — count = estimé
    card_candidates = soup.select(
        "[class*='card' i], [class*='offer' i], [class*='deal' i], [class*='result' i], "
        "[class*='product' i], [data-testid*='card' i]"
    )
    listing_count = len(card_candidates)
    listing_present = listing_count >= 4

    faq_visible = bool(soup.select_one("[class*='faq' i], [id*='faq' i], details summary, [role='tablist']"))

    # --- Heuristiques supplémentaires demandées (Avis / Contact / Articles)
    page_text = soup.get_text(" ", strip=True)
    page_text_lower = page_text.lower()

    reviews_present = bool(
        soup.select_one(
            "[class*='review' i], [class*='rating' i], [class*='star' i], "
            "[id*='review' i], [id*='rating' i], [data-testid*='rating' i]"
        )
        or (" avis" in page_text_lower)
        or ("trustpilot" in page_text_lower)
        or ("★" in page_text)  # étoile brute
        or ("/5" in page_text_lower and "note" in page_text_lower)
    )

    contact_present = bool(
        soup.select_one("a[href^='tel:'], a[href*='contact' i], [class*='contact' i], [id*='contact' i]")
        or ("nous contacter" in page_text_lower)
        or ("service client" in page_text_lower)
        or ("contact" in page_text_lower)
        or ("chat" in page_text_lower)
    )

    articles_present = bool(
        soup.select_one(
            "[class*='article' i], [id*='article' i], [class*='blog' i], [class*='magazine' i], "
            "[class*='inspiration' i]"
        )
        or soup.select_one(
            "a[href*='/blog' i], a[href*='/magazine' i], a[href*='inspiration' i], a[href*='conseil' i]"
        )
        or ("articles" in page_text_lower)
        or ("inspiration" in page_text_lower)
        or ("nos conseils" in page_text_lower)
    )

    # --- Structured data (JSON-LD types)
    jsonld_types: List[str] = []
    for s in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        txt = s.get_text(strip=True)
        if txt:
            jsonld_types += re.findall(r'"@type"\s*:\s*"([^"]+)"', txt)
    jsonld_types = sorted(set([t.strip() for t in jsonld_types if t.strip()]))

    # --- Content stats (très approximatif)
    soup2 = BeautifulSoup(html, "lxml")
    for tag in soup2(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup2.select("nav, footer"):
        tag.decompose()
    word_count = len(soup2.get_text(" ", strip=True).split())

    return {
        "url": url,
        "domain": domain_of(url),
        "meta": {
            "title": title,
            "meta_description": meta_desc,
            "canonical": canonical_href,
        },
        "headings": {
            "h1": h1,
            "toc_present": toc_present,
        },
        "modules": {
            "header_present": header_present,
            "breadcrumb_present": breadcrumb_present,
            "filters_present": filters_present,
            "listing_present": listing_present,
            "listing_count_est": listing_count,
            "faq_visible": faq_visible,
            "reviews_present": reviews_present,
            "contact_present": contact_present,
            "articles_present": articles_present,
        },
        "structured_data": {"jsonld_types": jsonld_types},
        "content": {"word_count_est": word_count},
    }


def extract_fingerprint(url: str, mode: str = "hybrid") -> dict:
    static = fetch_static(url)
    fp_static = None
    js_needed = False
    js_reasons: List[str] = []

    if mode in ("static", "hybrid") and static.html:
        fp_static = extract_fingerprint_from_html(static.final_url or url, static.html)
        js_needed, js_reasons = needs_js(static.html)

    if mode == "rendered" or (mode == "hybrid" and (static.error or not static.html or js_needed)):
        rendered = fetch_rendered_playwright(url)
        if rendered.html:
            fp = extract_fingerprint_from_html(rendered.final_url or url, rendered.html)
            fp["_fetch"] = {
                "mode": rendered.mode,
                "status": rendered.status,
                "final_url": rendered.final_url,
                "elapsed_ms": rendered.elapsed_ms,
                "error": rendered.error,
                "js_decision": {"needs_js": js_needed, "reasons": js_reasons},
                "static_status": static.status,
                "static_error": static.error,
            }
            return fp

        # fallback si playwright échoue
        if fp_static is not None:
            fp_static["_fetch"] = {
                "mode": "static",
                "status": static.status,
                "final_url": static.final_url,
                "elapsed_ms": static.elapsed_ms,
                "error": static.error or f"Playwright error: {rendered.error}",
                "js_decision": {"needs_js": js_needed, "reasons": js_reasons},
            }
            return fp_static

        return {"url": url, "_fetch": {"mode": "none", "error": rendered.error}}

    if fp_static is None:
        fp_static = {"url": url}
    fp_static["_fetch"] = {
        "mode": "static",
        "status": static.status,
        "final_url": static.final_url,
        "elapsed_ms": static.elapsed_ms,
        "error": static.error,
        "js_decision": {"needs_js": js_needed, "reasons": js_reasons},
    }
    return fp_static


# =========================================================
# FORMATTERS (Excel cells)
# =========================================================
def format_cell_competitors(fps: List[dict]) -> str:
    seen = set()
    blocks = ["CONCURRENTS (Top SERP)", ""]
    i = 0

    for fp in fps:
        dom = fp.get("domain") or ""
        if not dom or dom in seen:
            continue
        seen.add(dom)
        i += 1

        url = fp.get("url")
        title = fp.get("meta", {}).get("title")
        h1 = fp.get("headings", {}).get("h1")
        m = fp.get("modules", {})
        schema = fp.get("structured_data", {}).get("jsonld_types") or []
        canonical = fp.get("meta", {}).get("canonical")

        blocks += [
            f"Domaine {i} : {dom}",
            f"URL : {url}",
            f"• Title : {title}",
            f"• H1 : {h1}",
            f"• Breadcrumb : {'Oui' if m.get('breadcrumb_present') else 'Non'}",
            f"• Listing : {'Oui' if m.get('listing_present') else 'Non'} | Nb encarts (est.) : {m.get('listing_count_est')}",
            f"• Filtres : {'Oui' if m.get('filters_present') else 'Non'}",
            f"• Q&A : {'Oui' if m.get('faq_visible') else 'Non'}",
            f"• Avis : {'Oui' if m.get('reviews_present') else 'Non'}",
            f"• Contact : {'Oui' if m.get('contact_present') else 'Non'}",
            f"• Articles : {'Oui' if m.get('articles_present') else 'Non'}",
            f"• Canonical : {'OK' if canonical else 'KO'}",
            f"• Données structurées (JSON-LD types) : {', '.join(schema) if schema else 'Non détecté'}",
            "",
            "---",
            "",
        ]

    return "\n".join(blocks).strip()


def format_cell_vp(fp: dict, brand: str = "Voyage Privé") -> str:
    url = fp.get("url")
    title = fp.get("meta", {}).get("title")
    meta_desc = fp.get("meta", {}).get("meta_description")
    canonical = fp.get("meta", {}).get("canonical")
    schema = fp.get("structured_data", {}).get("jsonld_types") or []
    m = fp.get("modules", {})
    h1 = fp.get("headings", {}).get("h1")

    return "\n".join([
        f"VP : {brand}",
        f"URL : {url}",
        f"• Title : {title}",
        f"• Meta description : {meta_desc or 'NC'}",
        f"• Canonical : {'OK' if canonical else 'KO'}",
        f"• H1 : {h1}",
        f"• Listing : {'Oui' if m.get('listing_present') else 'Non'} | Nb encarts (est.) : {m.get('listing_count_est')}",
        f"• Filtres : {'Oui' if m.get('filters_present') else 'Non'}",
        f"• Q&A : {'Oui' if m.get('faq_visible') else 'Non'}",
        f"• Avis : {'Oui' if m.get('reviews_present') else 'Non'}",
        f"• Contact : {'Oui' if m.get('contact_present') else 'Non'}",
        f"• Articles : {'Oui' if m.get('articles_present') else 'Non'}",
        f"• Données structurées (JSON-LD types) : {', '.join(schema) if schema else 'Non détecté'}",
    ])


def format_cell_vp_missing_only(fp: dict) -> str:
    _, v_missing = score_features(fp, VISIBLE_FEATURES)
    _, m_missing = score_features(fp, META_FEATURES)

    url = fp.get("url")
    notes = []
    if fp.get("modules", {}).get("listing_present") and not fp.get("modules", {}).get("filters_present"):
        notes.append("Listing détecté sans filtres → peut être injecté en JS, ou filtres non exposés en HTML.")

    return "\n".join([
        f"URL : {url}",
        "",
        "❌ Visible — Manquant / Non détecté",
        "• " + " | ".join(v_missing) if v_missing else "• RAS",
        "",
        "❌ Meta — Manquant / Non détecté",
        "• " + " | ".join(m_missing) if m_missing else "• RAS",
        "",
        "🔎 Notes",
        "• " + "\n• ".join(notes) if notes else "• RAS",
    ])


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="SERP → Empreintes SEO (1 fichier)", layout="wide")
st.title("SERP → Empreintes SEO (1 fichier) — SerpApi + Hybrid fetch")

with st.sidebar:
    st.header("Paramètres SERP")
    gl = st.selectbox("Pays (gl)", ["fr", "be", "ch", "ca", "us", "uk", "de", "es", "it"], index=0)
    hl = st.selectbox("Langue (hl)", ["fr", "en", "de", "es", "it"], index=0)
    google_domain = st.text_input("Google domain", value="google.fr")
    location = st.text_input("Location (optionnel)", value="France")
    topn = st.number_input("Top N", min_value=1, max_value=20, value=10, step=1)

    st.header("Mode de crawl")
    mode = st.radio(
        "Méthode",
        ["hybrid", "static", "rendered"],
        index=0,
        help="hybrid = statique + fallback Playwright si page 'vide'",
    )

st.subheader("1) Requête & URL VP")
q = st.text_input("Requête (mot-clé)", value="circuit autotour")
vp_url = st.text_input("URL VP", value="https://www.voyage-prive.com/offres/voyage-organise-et-circuit-all-inclusive")

run = st.button("Lancer (SERP + empreintes)", type="primary")

if run:
    if not q.strip():
        st.error("Merci de saisir une requête.")
        st.stop()

    with st.spinner("Récupération SERP via SerpApi…"):
        data = serpapi_google_search(
            q=q.strip(),
            gl=gl,
            hl=hl,
            num=int(topn),
            location=location.strip() if location.strip() else None,
            google_domain=google_domain.strip() if google_domain.strip() else "google.fr",
        )

    organic = data.get("organic_results", []) or []
    serp_urls = [r.get("link") for r in organic if r.get("link")]
    serp_urls = serp_urls[: int(topn)]

    if not serp_urls:
        st.warning("Aucune URL organique récupérée.")
        st.json(data)
        st.stop()

    st.write(f"URLs SERP récupérées : **{len(serp_urls)}**")

    # Fingerprints concurrents
    fps = []
    prog = st.progress(0)
    for i, url in enumerate(serp_urls, start=1):
        fp = extract_fingerprint(url, mode=mode)
        fps.append(fp)
        prog.progress(int(i / max(1, len(serp_urls)) * 100))

    # Fingerprint VP
    fp_vp = extract_fingerprint(vp_url, mode=mode)

    # Debug table (utile pour vérifier les heuristiques)
    rows = []
    for fp in fps:
        rows.append({
            "domain": fp.get("domain"),
            "url": fp.get("url"),
            "title": fp.get("meta", {}).get("title"),
            "h1": fp.get("headings", {}).get("h1"),
            "listing_present": fp.get("modules", {}).get("listing_present"),
            "listing_count_est": fp.get("modules", {}).get("listing_count_est"),
            "filters_present": fp.get("modules", {}).get("filters_present"),
            "faq_visible": fp.get("modules", {}).get("faq_visible"),
            "reviews_present": fp.get("modules", {}).get("reviews_present"),
            "contact_present": fp.get("modules", {}).get("contact_present"),
            "articles_present": fp.get("modules", {}).get("articles_present"),
            "schema_types": ", ".join(fp.get("structured_data", {}).get("jsonld_types") or []),
            "fetch_mode": fp.get("_fetch", {}).get("mode"),
            "js_reasons": ",".join(fp.get("_fetch", {}).get("js_decision", {}).get("reasons", [])),
            "fetch_error": fp.get("_fetch", {}).get("error"),
        })
    df = pd.DataFrame(rows)

    st.subheader("2) Debug (table)")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Télécharger CSV debug",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"debug_fingerprints_{gl}_{hl}_{q.strip().replace(' ', '_')}.csv",
        mime="text/csv",
    )

    # Sorties Excel
    st.subheader("3) Sorties Excel (copier-coller)")
    cell_comp = format_cell_competitors(fps)
    cell_vp = format_cell_vp(fp_vp)
    cell_vp_missing = format_cell_vp_missing_only(fp_vp)

    c1, c2 = st.columns(2)
    with c1:
        st.text_area("Cellule CONCURRENTS", value=cell_comp, height=520)
        st.text_area("Cellule VP", value=cell_vp, height=420)
    with c2:
        st.text_area("Cellule VP — Réca​p (Manquants)", value=cell_vp_missing, height=420)
