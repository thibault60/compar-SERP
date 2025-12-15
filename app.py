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


# -----------------------------
# CONFIG / CONSTANTS
# -----------------------------
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ✅ Visible sur la page (UX / contenu)
VISIBLE_FEATURES = [
    ("H1", lambda fp: bool(fp.get("headings", {}).get("h1"))),
    ("Header", lambda fp: bool(fp.get("modules", {}).get("header_present"))),
    ("Breadcrumb visible", lambda fp: bool(fp.get("modules", {}).get("breadcrumb_present"))),
    ("Listing visible", lambda fp: bool(fp.get("modules", {}).get("listing_present"))),
    ("Filtres visibles", lambda fp: bool(fp.get("modules", {}).get("filters_present"))),
    ("Sommaire/TOC visible", lambda fp: bool(fp.get("headings", {}).get("toc_present"))),
    ("Q&A visible", lambda fp: bool(fp.get("modules", {}).get("faq_visible"))),
    ("Contenu éditorial base", lambda fp: (fp.get("content", {}).get("word_count_est") or 0) >= 300),
]

# ✅ Meta / SEO (non “visible” au sens UX)
META_FEATURES = [
    ("Title", lambda fp: bool(fp.get("meta", {}).get("title"))),
    ("Meta description", lambda fp: bool(fp.get("meta", {}).get("meta_description"))),
    ("Canonical", lambda fp: bool(fp.get("meta", {}).get("canonical"))),
    ("Schema BreadcrumbList", lambda fp: "BreadcrumbList" in (fp.get("structured_data", {}).get("jsonld_types") or [])),
    ("Schema ItemList", lambda fp: "ItemList" in (fp.get("structured_data", {}).get("jsonld_types") or [])),
    ("Schema FAQPage", lambda fp: "FAQPage" in (fp.get("structured_data", {}).get("jsonld_types") or [])),
]


# -----------------------------
# UTILS
# -----------------------------
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


def score_features(fp: dict, features: list[tuple[str, callable]]) -> tuple[list[str], list[str], str]:
    present, missing = [], []
    for label, fn in features:
        ok = False
        try:
            ok = bool(fn(fp))
        except Exception:
            ok = False
        (present if ok else missing).append(label)
    score = f"{len(present)}/{len(present) + len(missing)}"
    return present, missing, score


def schema_recommendations_for_vp(fp: dict) -> list[str]:
    """
    Recos basées sur ce qui est visible :
    - Breadcrumb visible -> recommander BreadcrumbList si absent
    - Listing visible -> recommander ItemList si absent
    - FAQ visible -> recommander FAQPage si absent
    """
    recos = []
    types = fp.get("structured_data", {}).get("jsonld_types") or []
    m = fp.get("modules", {}) or {}

    if m.get("breadcrumb_present") and "BreadcrumbList" not in types:
        recos.append("Ajouter Schema BreadcrumbList (fil d’Ariane visible)")

    if m.get("listing_present") and "ItemList" not in types:
        recos.append("Ajouter Schema ItemList (listing d’offres visible)")

    if m.get("faq_visible") and "FAQPage" not in types:
        recos.append("Ajouter Schema FAQPage (bloc Q&A visible)")

    return recos


# -----------------------------
# SERPAPI
# -----------------------------
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


# -----------------------------
# FETCHERS
# -----------------------------
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

            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass

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


# -----------------------------
# DETECTOR (need JS?)
# -----------------------------
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


# -----------------------------
# EXTRACTOR (fingerprint)
# -----------------------------
def extract_fingerprint_from_html(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    title = clean_text(soup.title.get_text()) if soup.title else None

    meta_desc = None
    md = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if md and md.get("content"):
        meta_desc = clean_text(md.get("content"))

    h1_tag = soup.find("h1")
    h1 = clean_text(h1_tag.get_text()) if h1_tag else None

    h2s = [clean_text(h.get_text()) for h in soup.find_all("h2")]
    h2s = [x for x in h2s if x]

    header = soup.find("header")
    header_present = bool(header)
    header_links = len(header.find_all("a")) if header else 0

    breadcrumb_present = bool(
        soup.select_one("[aria-label*='breadcrumb' i], nav.breadcrumb, .breadcrumb, [class*='breadcrumb']")
    )

    toc_present = bool(
        soup.select_one("[id*='toc' i], [class*='toc' i], [class*='sommaire' i]")
    )

    canonical = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
    canonical_href = clean_text(canonical.get("href")) if canonical else None

    # JSON-LD types
    jsonld_types: List[str] = []
    for s in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        txt = s.get_text(strip=True)
        if txt:
            jsonld_types += re.findall(r'"@type"\s*:\s*"([^"]+)"', txt)
    jsonld_types = sorted(set([t.strip() for t in jsonld_types if t.strip()]))

    # FAQ visible (UX) + schema FAQPage (meta)
    faq_visible = bool(soup.select_one("[class*='faq' i], [id*='faq' i], details summary, [role='tablist']"))

    # Filtres visibles (heuristique)
    filters_present = bool(
        soup.select_one("[class*='filter' i], [id*='filter' i], [aria-label*='filtre' i], [data-testid*='filter' i]")
    )

    # Listing visible (heuristique) — on évite "article" (trop bruit)
    card_candidates = soup.select(
        "[class*='card' i], [class*='offer' i], [class*='deal' i], [class*='result' i], "
        "[class*='product' i], [data-testid*='card' i]"
    )
    listing_count = len(card_candidates)
    listing_present = listing_count >= 4  # seuil + “soft” (et count = estimé)

    # Word count (approx) – remove noisy sections
    soup2 = BeautifulSoup(html, "lxml")
    for tag in soup2(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup2.select("nav, footer"):
        tag.decompose()
    text = soup2.get_text(" ", strip=True)
    word_count = len(text.split())

    return {
        "url": url,
        "domain": domain_of(url),
        "meta": {
            "title": title,
            "meta_description": meta_desc,
            "canonical": canonical_href,
        },
        "headings": {"h1": h1, "h2_count": len(h2s), "h2_sample": h2s[:6], "toc_present": toc_present},
        "modules": {
            "header_present": header_present,
            "header_links": header_links,
            "breadcrumb_present": breadcrumb_present,
            "filters_present": filters_present,
            "listing_present": listing_present,
            "listing_count_est": listing_count,
            "faq_visible": faq_visible,
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


# -----------------------------
# FORMATTERS (Excel cells)
# -----------------------------
def format_cell_vp(fp: dict, brand: str = "Voyage Privé") -> str:
    url = fp.get("url")
    title = fp.get("meta", {}).get("title")
    meta_desc = fp.get("meta", {}).get("meta_description")
    h1 = fp.get("headings", {}).get("h1")
    h2_count = fp.get("headings", {}).get("h2_count")
    toc = "Oui" if fp.get("headings", {}).get("toc_present") else "Non"

    m = fp.get("modules", {})
    listing = "Oui" if m.get("listing_present") else "Non"
    listing_n = m.get("listing_count_est")
    filtres = "Oui" if m.get("filters_present") else "Non"
    faq = "Oui" if m.get("faq_visible") else "Non"

    wc = fp.get("content", {}).get("word_count_est")
    schema = fp.get("structured_data", {}).get("jsonld_types") or []
    canonical = fp.get("meta", {}).get("canonical")

    return "\n".join([
        f"VP : {brand}",
        f"URL : {url}",
        "",
        "VISIBLE (page)",
        f"• H1 : {h1}",
        f"• Header : {'Oui' if m.get('header_present') else 'Non'} | Breadcrumb : {'Oui' if m.get('breadcrumb_present') else 'Non'}",
        f"• Filtres : {filtres}",
        f"• Listing : {listing} | Nb encarts (est.) : {listing_n}",
        f"• Q&A visible : {faq}",
        f"• Contenu éditorial base : {wc} mots | H2 : {h2_count} | TOC : {toc}",
        "",
        "META (SEO)",
        f"• Title : {title}",
        f"• Meta description : {meta_desc or 'NC'}",
        f"• Canonical : {'OK' if canonical else 'KO'}",
        f"• Données structurées (JSON-LD types) : {', '.join(schema) if schema else 'Non détecté'}",
    ])


def format_cell_vp_presence_missing(fp: dict) -> str:
    v_present, v_missing, v_score = score_features(fp, VISIBLE_FEATURES)
    m_present, m_missing, m_score = score_features(fp, META_FEATURES)

    url = fp.get("url")
    wc = fp.get("content", {}).get("word_count_est") or 0

    notes = []
    if wc < 300:
        notes.append(f"Contenu éditorial base faible ({wc} mots) → renforcer la partie éditoriale hors listing.")
    if fp.get("modules", {}).get("listing_present") and not fp.get("modules", {}).get("filters_present"):
        notes.append("Listing détecté sans filtres → peut être injecté en JS, ou filtres non exposés en HTML.")
    notes.append("Nb encarts / filtres : indicateurs 'fragiles' en multi-domaines → privilégier Oui/Non + sélecteurs par site si besoin.")

    recos_schema = schema_recommendations_for_vp(fp)

    lines = [
        "VP — RÉCAP (VISIBLE vs META)",
        f"URL : {url}",
        "",
        f"SCORE VISIBLE : {v_score}",
        "✅ Visible — Présent",
        "• " + " | ".join(v_present) if v_present else "• (aucun)",
        "❌ Visible — Manquant / Non détecté",
        "• " + " | ".join(v_missing) if v_missing else "• (aucun)",
        "",
        f"SCORE META : {m_score}",
        "✅ Meta — Présent",
        "• " + " | ".join(m_present) if m_present else "• (aucun)",
        "❌ Meta — Manquant / Non détecté",
        "• " + " | ".join(m_missing) if m_missing else "• (aucun)",
        "",
        "📌 Schémas recommandés (selon ce qui est visible)",
        "• " + "\n• ".join(recos_schema) if recos_schema else "• RAS",
        "",
        "🔎 Notes",
        "• " + "\n• ".join(notes) if notes else "• RAS",
    ]
    return "\n".join(lines)


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
        wc = fp.get("content", {}).get("word_count_est")
        schema = fp.get("structured_data", {}).get("jsonld_types") or []
        canonical = fp.get("meta", {}).get("canonical")

        blocks += [
            f"Domaine {i} : {dom}",
            f"URL : {url}",
            f"• Title : {title}",
            f"• H1 : {h1}",
            f"• Header : {'Oui' if m.get('header_present') else 'Non'} | Breadcrumb : {'Oui' if m.get('breadcrumb_present') else 'Non'}",
            f"• Filtres : {'Oui' if m.get('filters_present') else 'Non'}",
            f"• Listing : {'Oui' if m.get('listing_present') else 'Non'} | Nb encarts (est.) : {m.get('listing_count_est')}",
            f"• Q&A visible : {'Oui' if m.get('faq_visible') else 'Non'}",
            f"• Contenu éditorial base : {wc} mots | TOC : {'Oui' if fp.get('headings', {}).get('toc_present') else 'Non'}",
            f"• Canonical : {'OK' if canonical else 'KO'}",
            f"• Données structurées (JSON-LD types) : {', '.join(schema) if schema else 'Non détecté'}",
            "",
            "---",
            "",
        ]

    return "\n".join(blocks).strip()


# -----------------------------
# STREAMLIT UI
# -----------------------------
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

    fps = []
    prog = st.progress(0)
    for i, url in enumerate(serp_urls, start=1):
        fp = extract_fingerprint(url, mode=mode)
        fps.append(fp)
        prog.progress(int(i / max(1, len(serp_urls)) * 100))

    fp_vp = extract_fingerprint(vp_url, mode=mode)

    # Debug table (simplifiée + focus fiabilité)
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
            "word_count_est": fp.get("content", {}).get("word_count_est"),
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

    # Excel cells
    st.subheader("3) Sorties Excel (copier-coller)")
    cell_comp = format_cell_competitors(fps)
    cell_vp = format_cell_vp(fp_vp)
    cell_vp_pm = format_cell_vp_presence_missing(fp_vp)

    c1, c2 = st.columns(2)
    with c1:
        st.text_area("Cellule CONCURRENTS", value=cell_comp, height=420)
        st.text_area("Cellule VP", value=cell_vp, height=520)
    with c2:
        st.text_area("Cellule VP — Récap (Visible vs Meta)", value=cell_vp_pm, height=720)
