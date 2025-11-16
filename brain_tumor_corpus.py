# brain_tumor_corpus.py
# Build/refresh corpus JSONL from links, optionally ingest Q/A pairs, and (re)build FAISS.

import os, re, json, csv, hashlib, time, random, pathlib, shutil
from urllib.parse import urlparse, urljoin
from datetime import datetime
from typing import List
from collections import Counter

import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import tiktoken
import trafilatura

# LangChain bits
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # modern import

# ---------- CONFIG ----------
SAVE_ROOT = "corpus"
PDF_DIR   = os.path.join(SAVE_ROOT, "pdfs")
HTML_DIR  = os.path.join(SAVE_ROOT, "html")
JSONL_DIR = os.path.join(SAVE_ROOT, "jsonl")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(JSONL_DIR, exist_ok=True)

# Where we store the main text corpus JSONL and the FAISS index
JSONL_PATH = os.path.join(JSONL_DIR, "brain_tumor_corpus.jsonl")
RAG_DIR    = os.path.join(SAVE_ROOT, "rag_index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Q/A sources (produced by extract_qa_from_jsonl.py)
QA_JSONL_PATH = os.getenv("QA_JSONL_PATH", os.path.join(SAVE_ROOT, "qa", "qa_pairs.jsonl"))
QA_CSV_PATH   = os.getenv("QA_CSV_PATH",   os.path.join(SAVE_ROOT, "qa", "qa_pairs.csv"))

# Force rebuilding FAISS each run (delete old index files first)
FORCE_REBUILD = os.getenv("FORCE_REBUILD", "1").lower() not in ("0", "false", "")

# ---------- SOURCE LINKS ---------- TARGET
LINKS = [
    
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8328013/pdf",  # WHO 2021
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9427889/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8563316/pdf",
    "https://www.nature.com/articles/s41571-020-00447-z",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10158123/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/pdf",
    "https://kjronline.org/pdf/10.3348/kjr.2024.0016",
    "https://www.nccn.org/patients/guidelines/content/PDF/brain-gliomas-patient.pdf",
    "https://bookcafe.yuntsg.com/ueditor/jsp/upload/file/20241017/1729130847445055961.pdf",
    "https://publications.iarc.who.int/Book-And-Report-Series/Who-Classification-Of-Tumours/Central-Nervous-System-Tumours-2021",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11801446/pdf",
    "https://zenodo.org/records/4575162/files/RSNA-MICCAIBrainTumorSegmentation%28BraTS%29Challenge2021_02-26-2021_08-29-02.pdf?download=1",
    "https://arxiv.org/abs/2107.02314",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10441440/pdf",
    "https://www.cancerimagingarchive.net/collection/tcga-gbm/",
    "https://www.cancerimagingarchive.net/collection/rider-neuro-mri/",
    "https://www.cancerimagingarchive.net/collection/rhuh-gbm/",
    "https://www.cancer.gov/ccg/research/genome-sequencing/tcga",
    "https://www.biorxiv.org/content/10.1101/2023.02.19.529134v3.full.pdf",
    "https://www.cureus.com/articles/156034-non-enhancing-glioblastoma-a-case-report.pdf",
    "https://www.cureus.com/articles/309174-atypical-presentation-of-glioblastoma-a-case-report.pdf",
    "https://www.cureus.com/articles/236615-integrating-physiotherapy-for-enhancing-functional-recovery-in-glioblastoma-multiforme-a-case-report.pdf",
    "https://www.cureus.com/articles/178035-intradural-intramedullary-spinal-cord-glioblastoma-a-case-report.pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9379095/pdf",
    "https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2022.1086591/pdf",
    "https://www.cureus.com/articles/230625-meningioma-of-the-fourth-ventricle-of-the-brain-a-case-report.pdf",
    # Etiology/epidemiology set (shortened here for brevity – keep yours)
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC5123811/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8004084/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9911713/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC5563115/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10477988/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10571999/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12185496/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC2945461/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11596971/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4995146/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4257885/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC5790603/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4110181/pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10248585/pdf",
    # General-knowledge Q/A-friendly (shortened – keep yours)
    "https://www.uhcw.nhs.uk/download/clientfiles/files/Patient%20Information%20Leaflets/Trauma%20and%20Neuro%20services/Neurosurgery/Brain%20tumours.pdf",
    "https://www.abta.org/wp-content/uploads/2020/06/About-Brain-Tumors_2020_web_en.pdf",
    "https://www.cancer.org.au/assets/pdf/understanding-brain-tumour-booklet",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3842445/",
    "https://www.nhs.uk/conditions/brain-tumours/",
    "https://www.nhs.uk/conditions/malignant-brain-tumour/",
    "https://medlineplus.gov/braintumors.html",
    "https://www.cancer.gov/types/brain/hp/adult-brain-treatment-pdq",
    "https://www.abta.org/wp-content/uploads/2018/03/pituitary-tumors-brochure.pdf",  # ABTA pituitary overview
    "https://www.urmc.rochester.edu/medialibraries/urmcmedia/neurosurgery/specialties/neuroendocrine/documents/pituitarytumors.pdf",  # URMC pituitary
    "https://www.nanosweb.org/files/Patient%20Brochures/English/2021/Pituitary_Tumor.pdf",  # NANOS pituitary
    "https://www.astro.org/ASTRO/media/ASTRO/AffiliatePages/arro/PDFs/ARROCase_Pituitary.pdf",  # ASTRO pituitary case
    "https://www.nejm.org/doi/pdf/10.1056/nejmra1810772",  # NEJM pituitary endocrinopathies
    "https://www.btrt.org/Synapse/Data/PDFData/0212BTRT/btrt-11-173.pdf",  # BTRT pituitary classification
    "https://www.ese-hormones.org/media/gpfdoe2i/ese-patient-leaflet_pregnancy_final.pdf",  # ESE pituitary in pregnancy
    "https://www.ccjm.org/content/ccjom/75/11/793.full.pdf",  # CCJM pituitary incidentalomas
    "https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2021.604644/pdf",  # Frontiers pituitary classification
    "https://www.nanosweb.org/files/public/pituitarytumor.pdf",  # Additional NANOS pituitary
    "https://www.cancertherapyadvisor.com/wp-content/uploads/sites/12/2025/04/glioma-vs-glioblastoma.pdf",  # CancerTherapyAdvisor glioma factsheet (adds more glioma depth)
    "https://www.mayoclinic.org/diseases-conditions/meningioma/symptoms-causes/syc-20355643",
"https://www.cancer.gov/rare-brain-spine-tumor/tumors/meningioma",
"https://my.clevelandclinic.org/health/diseases/17858-meningioma",
"https://pmc.ncbi.nlm.nih.gov/articles/PMC8004084/",
"https://rarediseases.org/rare-diseases/meningioma/",
"https://pmc.ncbi.nlm.nih.gov/articles/PMC2945460/",
"https://nyulangone.org/conditions/meningioma/diagnosis",
"https://stanfordhealthcare.org/medical-conditions/brain-and-nerves/meningioma-skull-base/about-this-condition/diagnosis.html",
"https://www.nature.com/articles/s41598-023-41576-6",
"https://www.ncbi.nlm.nih.gov/books/NBK560538/",
"https://www.mayoclinic.org/diseases-conditions/meningioma/diagnosis-treatment/drc-20355648",
"https://www.hopkinsmedicine.org/health/treatment-tests-and-therapies/meningioma-treatment",
"https://pmc.ncbi.nlm.nih.gov/articles/PMC7473392/",
"https://njbrainspine.com/surgical-non-surgical-approaches-meningioma-treatment/",
"https://www.moffitt.org/cancers/meningioma/treatment/radiation-therapy/",
"https://pmc.ncbi.nlm.nih.gov/articles/PMC9722800/",
"https://www.redjournal.org/article/S0360-3016%252824%252903525-9/fulltext",
"https://www.aaroncohen-gadol.com/en/patients/meningioma/treatment/radiation-therapy",
"https://nyulangone.org/conditions/meningioma/treatments/radiation-therapy-for-meningioma",
"https://pubmed.ncbi.nlm.nih.gov/37774783/",
"https://www.sciencedirect.com/science/article/pii/S1878875023013438",
"https://journals.lww.com/annals-of-medicine-and-surgery/fulltext/2025/08000/evolving_treatment_strategies_in_meningioma__from.55.aspx",
"https://ar.iiarjournals.org/content/35/12/6391",
"https://www.barrowneuro.org/condition/meningioma/",
"https://www.redjournal.org/article/S0360-3016%2525252824%2525252903525-9/pdf",
"https://pmc.ncbi.nlm.nih.gov/articles/PMC10469847/pdf",
"https://www.e-roj.org/upload/pdf/roj-2021-00563.pdf",
"https://ajronline.org/doi/pdf/10.2214/ajr.123.3.453?download=true",
"https://www.astro.org/ASTRO/media/ASTRO/AffiliatePages/arro/PDFs/ARROCase_MeningiomaAdjuvant.pdf",
"https://www.redjournal.org/article/S0360-3016%252824%252903525-9/pdf",
"https://www.advancesradonc.org/article/S2452-1094%2824%2900272-0/pdf",
"https://www.practicalradonc.org/article/S1879-8500%2824%2900090-0/pdf",
"https://thejns.org/downloadpdf/journals/j-neurosurg/89/6/article-p933.pdf",
"https://www.astro.org/getmedia/ff0d6a20-3d88-4713-8710-e24c1efddc11/WHOGrade4Glioma_PC.pdf",
"https://www.brainlife.org/fulltext/2024/Trkova_K240207_SciRep.pdf",
"https://www.biorxiv.org/content/10.1101/2025.10.08.681196v1.full.pdf",
"https://pmc.ncbi.nlm.nih.gov/articles/PMC9617255/pdf",
"https://www.annalsofoncology.org/article/S0923-7534%2819%2941755-9/pdf",
"https://pmc.ncbi.nlm.nih.gov/articles/PMC9031600/pdf",
"https://www.eymj.org/pdf/10.3349/ymj.2022.0323",
"https://www.astro.org/ASTRO/media/ASTRO/AffiliatePages/arro/PDFs/ARROCase_Pituitary.pdf",
"https://www.redjournal.org/article/S0360-3016%2807%2904704-9/pdf",
"https://www.abta.org/wp-content/uploads/2018/03/pituitary-tumors-brochure.pdf",
"https://www.urmc.rochester.edu/medialibraries/urmcmedia/neurosurgery/specialties/neuroendocrine/documents/pituitarytumors.pdf?fbclid=IwAR0ZY5FlVIKKMaLrqWAXMuZU34gJVui4dl4BIxWwzWYHdtsXt-UeNj_laQI",
"https://www.msjonline.org/index.php/ijrms/article/download/13326/8620/61272",
"https://www.astro.org/ASTRO/media/ASTRO/MedulloblastomaAJW.pdf",
"https://www.redjournal.org/article/S0360-3016%2812%2901220-5/pdf",
"https://www.turkarchpediatr.org/Content/files/sayilar/128/TAP_May_2023-76-81.pdf",
"https://pdfs.semanticscholar.org/45f7/35ceafa022da5ba6c9c934bef3ff7de1841c.pdf",
"https://www.pbtc.org/public/Summaries/PBTC-060/PBTC-060%2520v1%2520Abstract%2C%2520Objectives%2520and%2520Eligibility%2520for%2520Website%2520090121.pdf",
"https://thejns.org/downloadpdf/view/journals/j-neurosurg-pediatr/23/3/article-p261.pdf",
"https://www.abta.org/wp-content/uploads/2018/03/about-brain-tumors-a-primer-1.pdf",
"https://www.ivybraintumorcenter.org/wp-content/uploads/2019/11/BrainTumor_Handbook.pdf",
"https://www.ucsfhealth.org/-/media/project/ucsf/ucsf-health/pdf/brain_tumor_patients_for_healthcare_providers.pdf",
"https://www.govinfo.gov/content/pkg/GOVPUB-HE20_3150-PURL-LPS118730/pdf/GOVPUB-HE20_3150-PURL-LPS118730.pdf",
"https://virtualtrials.org/Guide/BrainTumorGuidev12.pdf",
"https://www.aafp.org/pubs/afp/issues/2016/0201/p211.pdf",
"https://stachestrong.org/wp-content/uploads/2024/03/Newly-Diagnosed_PDF_ABTA1223.pdf"
]

ALL_LINKS = list(dict.fromkeys(LINKS))
dups = [u for u, c in Counter(LINKS).items() if c > 1]
if dups:
    print(f"[Corpus] {len(dups)} duplicate URL(s) removed:")
    for u in dups:
        print("  -", u)

print(f"[Corpus] {len(ALL_LINKS)} unique links")

# ---------- Robust fetch ----------
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

def new_session():
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    return s

def with_backoff(fn, tries=3, base_delay=1.0):
    for i in range(tries):
        try:
            return fn()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (403, 429, 500, 502, 503, 504):
                time.sleep(base_delay * (2 ** i) + random.random() * 0.3)
                continue
            raise
        except requests.RequestException:
            time.sleep(base_delay * (2 ** i) + random.random() * 0.3)
            if i == tries - 1:
                raise
    raise RuntimeError("exhausted retries")

def guess_is_pdf_url(u: str) -> bool:
    return u.lower().endswith(".pdf")

def page_find_pdf_link(html: str, base_url: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    a = soup.select_one("a[href$='.pdf'], a[href*='/pdf']")
    if a and a.get("href"):
        return urljoin(base_url, a["href"])
    if "arxiv.org/abs/" in base_url:
        return base_url.replace("/abs/", "/pdf/") + ".pdf"
    return None

def is_valid_pdf_bytes(b: bytes, min_bytes: int = 40_000) -> bool:
    return len(b) >= min_bytes and b.startswith(b"%PDF")

def fetch(url: str, PDF_DIR: str, HTML_DIR: str) -> tuple[str, str]:
    sess = new_session()
    if "pmc.ncbi.nlm.nih.gov/articles/" in url and not url.rstrip("/").endswith("/pdf"):
        url = url.rstrip("/") + "/pdf"

    parsed = urlparse(url)
    sess.headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}"

    def _get_main():
        r = sess.get(url, timeout=80, allow_redirects=True)
        r.raise_for_status()
        return r
    r = with_backoff(_get_main)

    ctype = r.headers.get("Content-Type", "").lower()
    looks_pdf = ("application/pdf" in ctype) or guess_is_pdf_url(r.url)

    def save_html(resp):
        sha = hashlib.sha1(resp.url.encode()).hexdigest()
        path = os.path.join(HTML_DIR, f"{sha}.html")
        with open(path, "wb") as f:
            f.write(resp.content)
        return path, "html"

    def try_save_pdf(content: bytes, final_url: str):
        if is_valid_pdf_bytes(content):
            sha = hashlib.sha1(final_url.encode()).hexdigest()
            path = os.path.join(PDF_DIR, f"{sha}.pdf")
            with open(path, "wb") as f:
                f.write(content)
            return path, "pdf"
        return None

    if looks_pdf:
        saved = try_save_pdf(r.content, r.url)
        if saved:
            return saved
        main_page = sess.get(r.url, timeout=60, allow_redirects=True)
        if "text/html" in main_page.headers.get("Content-Type", "").lower():
            pdf_url = page_find_pdf_link(main_page.text, main_page.url)
            if pdf_url:
                sess.headers["Referer"] = main_page.url
                def _get_pdf():
                    rp = sess.get(pdf_url, timeout=80, allow_redirects=True)
                    rp.raise_for_status()
                    return rp
                try:
                    rp = with_backoff(_get_pdf)
                    saved = try_save_pdf(rp.content, pdf_url)
                    if saved:
                        return saved
                except Exception:
                    pass
            return save_html(main_page)
        return save_html(r)

    if ("text/html" in ctype) or r.text.strip().startswith("<"):
        pdf_url = page_find_pdf_link(r.text, r.url)
        if pdf_url:
            sess.headers["Referer"] = r.url
            def _get_pdf():
                rp = sess.get(pdf_url, timeout=80, allow_redirects=True)
                rp.raise_for_status()
                return rp
            try:
                rp = with_backoff(_get_pdf)
                saved = try_save_pdf(rp.content, pdf_url)
                if saved:
                    return saved
            except Exception:
                pass
        return save_html(r)

    return save_html(r)

# ---------- Text extraction & chunking ----------
def read_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    parts = [p.get_text("text") for p in doc]
    return "\n".join(parts)

def read_html_text(path: str) -> str:
    raw = open(path, "rb").read()
    txt = trafilatura.extract(raw, include_tables=True, include_comments=False)
    if txt:
        return txt
    soup = BeautifulSoup(raw, "lxml")
    return soup.get_text(separator="\n")

def sanitize(s: str) -> str:
    s = re.sub(r"\r", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def extract_year(text: str):
    head = text[:10000]
    yrs = re.findall(r"\b(19\d{2}|20\d{2})\b", head)
    return max(yrs) if yrs else None

def guess_tumor_type(text: str):
    t = text.lower()
    for k in ["glioblastoma", " gbm "]:
        if k in t: return "glioblastoma"
    if "oligodendroglioma" in t: return "oligodendroglioma"
    if "astrocytoma" in t: return "astrocytoma"
    if "meningioma" in t: return "meningioma"
    if "medulloblastoma" in t: return "medulloblastoma"
    if "ependymoma" in t: return "ependymoma"
    if "diffuse midline glioma" in t or "h3k27m" in t: return "diffuse_midline_glioma"
    if "lymphoma" in t and "central nervous system" in t: return "pcnsl"
    if "glioma" in t: return "glioma"
    if "pituitary" in t and ("adenoma" in t or "tumor" in t):
        return "pituitary_tumor"
    return None

enc = tiktoken.get_encoding("cl100k_base")
def chunk_text(txt: str, max_tokens=900, overlap=120):
    toks = enc.encode(txt)
    out, i, step = [], 0, max_tokens - overlap
    while i < len(toks):
        out.append(enc.decode(toks[i:i+max_tokens]))
        i += step
    return out

# ---------- JSONL writer from links ----------
def main():
    urls = list(ALL_LINKS)
    print(f"[Corpus] Processing {len(urls)} URL(s)")

    jsonl_path = os.path.join(JSONL_DIR, "brain_tumor_corpus.jsonl")
    n_pdf, n_html = 0, 0

    with open(JSONL_PATH, "w", encoding="utf-8") as out:
        for url in urls:
            try:
                local_path, ftype = fetch(url, PDF_DIR, HTML_DIR)
                if ftype == "pdf":
                    n_pdf += 1
                    raw = read_pdf_text(local_path)
                else:
                    n_html += 1
                    raw = read_html_text(local_path)

                txt   = sanitize(raw)
                year  = extract_year(txt)
                tumor = guess_tumor_type(txt)
                chunks = chunk_text(txt, max_tokens=900, overlap=120)

                rel_local = os.path.relpath(local_path, SAVE_ROOT)
                for idx, ch in enumerate(chunks):
                    metadata = {
                        "source_url": url,
                        "local_path": rel_local,
                        "source": url or rel_local,
                        "file_type": ftype,
                        "year": year,
                        "tumor_type": tumor,
                        "doc_type": "review_or_case",
                        "chunk_id": idx,
                    }
                    rec = {
                        "id": hashlib.md5(f"{url}-{idx}".encode()).hexdigest(),
                        "metadata": metadata,
                        "text": ch,
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                print(f"OK: {url} -> {ftype} ({len(chunks)} chunks)")
            except Exception as e:
                print("FAILED:", url, e)

    ingest_local_pdfs(jsonl_path, PDF_DIR)            

    pdf_bytes  = sum(os.path.getsize(os.path.join(PDF_DIR, f))  for f in os.listdir(PDF_DIR))  if os.listdir(PDF_DIR) else 0
    html_bytes = sum(os.path.getsize(os.path.join(HTML_DIR, f)) for f in os.listdir(HTML_DIR)) if os.listdir(HTML_DIR) else 0
    jsonl_bytes = os.path.getsize(JSONL_PATH) if os.path.exists(JSONL_PATH) else 0

    print("\n------ SUMMARY ------")
    print(f"Saved PDFs:  {n_pdf} files,  ~{pdf_bytes/1e6:.1f} MB  -> {PDF_DIR}")
    print(f"Saved HTML:  {n_html} files, ~{html_bytes/1e6:.1f} MB -> {HTML_DIR}")
    print(f"JSONL: ~{jsonl_bytes/1e6:.1f} MB @ {JSONL_PATH}")

# ---------- Load text corpus JSONL as Documents ----------
def _read_jsonl_docs(jsonl_path: str) -> List[Document]:
    docs = []
    if not os.path.exists(jsonl_path):
        return docs
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            text = (item.get("text") or "").strip()
            meta = item.get("metadata") or { k: item.get(k) for k in
                ["source_url", "local_path", "file_type", "year", "tumor_type", "doc_type", "chunk_id"]
                if item.get(k) is not None}
            if text:
                docs.append(Document(page_content=text, metadata=meta))
    return docs

# ---------- NEW: Load Q/A pairs (JSONL or CSV) as Documents ----------
def _read_qa_jsonl(path: str) -> List[Document]:
    docs = []
    if not os.path.exists(path):
        return docs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            if not q or not a:
                continue
            # store as a single doc so retriever sees both question and answer
            content = f"Q: {q}\nA: {a}"
            meta = {
                "source": r.get("source")or r.get("qa_source") or "qa_unknown",
                "qa_source": r.get("source"),
                "file_type": r.get("file_type"),
                "tumor_type": r.get("tumor_type"),
                "year": r.get("year"),
                "doc_type": "qa_pair",
                "confidence": r.get("confidence", 0.0),
            }
            docs.append(Document(page_content=content, metadata=meta))
    return docs

def _read_qa_csv(path: str) -> List[Document]:
    docs = []
    if not os.path.exists(path):
        return docs
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            if not q or not a:
                continue
            content = f"Q: {q}\nA: {a}"
            meta = {
                "source": r.get("source") or r.get("qa_source") or "qa_unknown",
                "qa_source": r.get("source"),
                "file_type": r.get("file_type"),
                "tumor_type": r.get("tumor_type"),
                "year": r.get("year"),
                "doc_type": r.get("doc_type") or "qa_pair",
                "confidence": float(r.get("confidence") or 0.0),
            }
            docs.append(Document(page_content=content, metadata=meta))
    return docs

# ---------- Build (or rebuild) FAISS from combined docs ----------
def build_faiss_from_sources(jsonl_path: str = JSONL_PATH,
                             qa_jsonl_path: str = QA_JSONL_PATH,
                             qa_csv_path: str = QA_CSV_PATH,
                             rag_dir: str = RAG_DIR,
                             emb_model: str = EMBED_MODEL,
                             force_rebuild: bool = FORCE_REBUILD) -> None:

    jsonl_path = os.path.abspath(jsonl_path)
    qa_jsonl_path = os.path.abspath(qa_jsonl_path)
    qa_csv_path = os.path.abspath(qa_csv_path)
    rag_dir    = os.path.abspath(rag_dir)
    os.makedirs(rag_dir, exist_ok=True)

    print(f"[RAG] Loading main JSONL: {jsonl_path}")
    docs_text = _read_jsonl_docs(jsonl_path)
    print(f"[RAG] Text docs: {len(docs_text)}")

    docs_qa_jsonl = _read_qa_jsonl(qa_jsonl_path)
    if docs_qa_jsonl:
        print(f"[RAG] Q/A (jsonl): {len(docs_qa_jsonl)} from {qa_jsonl_path}")

    docs_qa_csv = _read_qa_csv(qa_csv_path)
    if docs_qa_csv:
        print(f"[RAG] Q/A (csv):   {len(docs_qa_csv)} from {qa_csv_path}")

    docs = docs_text + docs_qa_jsonl + docs_qa_csv
    if not docs:
        raise RuntimeError("No documents found (text+QA); nothing to index.")
    print(f"[RAG] TOTAL documents to index: {len(docs)}")

    # Clean out existing index if forced
    if force_rebuild:
        for fname in ("index.faiss", "index.pkl"):
            try:
                os.remove(os.path.join(rag_dir, fname))
            except FileNotFoundError:
                pass

    embeddings = HuggingFaceEmbeddings(
        model_name=emb_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"[RAG] Building FAISS with embeddings: {emb_model}")
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(rag_dir)

    # Verify
    vs2 = FAISS.load_local(rag_dir, embeddings, allow_dangerous_deserialization=True)
    faiss_path = os.path.join(rag_dir, "index.faiss")
    pkl_path   = os.path.join(rag_dir, "index.pkl")
    faiss_mtime = time.ctime(os.path.getmtime(faiss_path)) if os.path.exists(faiss_path) else "MISSING"
    pkl_mtime   = time.ctime(os.path.getmtime(pkl_path))   if os.path.exists(pkl_path) else "MISSING"
    print(f"[RAG] Saved FAISS -> {rag_dir} | ntotal={vs2.index.ntotal} | dim={vs2.index.d}")
    print(f"[RAG] index.faiss mtime: {faiss_mtime}")
    print(f"[RAG] index.pkl   mtime: {pkl_mtime}")

# --- Ingest locally placed PDFs in corpus/pdfs (not tied to a URL) ---
def _sha1_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha1(b).hexdigest()

def ingest_local_pdfs(jsonl_path: str, pdf_dir: str = PDF_DIR):
    """Append local PDFs that don't already exist in JSONL."""
    # 1) collect already indexed (by local_path) to avoid duplicates
    already = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                meta = item.get("metadata") or {}
                lp = meta.get("local_path")
                if lp:
                    already.add(lp.replace("\\", "/"))

    # 2) walk corpus/pdfs and ingest anything not already present
    new_count = 0
    with open(jsonl_path, "a", encoding="utf-8") as out:
        for name in os.listdir(pdf_dir):
            if not name.lower().endswith(".pdf"):
                continue
            fpath = os.path.join(pdf_dir, name)
            rel_local = os.path.relpath(fpath, SAVE_ROOT).replace("\\", "/")
            if rel_local in already:
                continue

            try:
                raw = read_pdf_text(fpath)
                txt = sanitize(raw)
                year = extract_year(txt)
                tumor = guess_tumor_type(txt)
                chunks = chunk_text(txt, max_tokens=900, overlap=120)

                for idx, ch in enumerate(chunks):
                    metadata = {
                        "source_url": f"file://{rel_local}",
                        "local_path": rel_local,
                        "source": f"file://{rel_local}",
                        "file_type": "pdf",
                        "year": year,
                        "tumor_type": tumor,
                        "doc_type": "review_or_case",
                        "chunk_id": idx,
                    }
                    rec = {
                        "id": hashlib.md5(f"{rel_local}-{idx}".encode()).hexdigest(),
                        "metadata": metadata,
                        "text": ch,
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                new_count += 1
                print(f"OK (local): {rel_local} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"FAILED (local): {rel_local} -> {e}")

    print(f"[Local PDFs] Newly ingested: {new_count}")


# ---------- Run ----------loader.load()
if __name__ == "__main__":
    # 1) Rebuild the main text JSONL from links
    main()
    # 2) Build (or rebuild) FAISS, merging in any Q/A JSONL or CSV found
    build_faiss_from_sources(JSONL_PATH, QA_JSONL_PATH, QA_CSV_PATH, RAG_DIR, EMBED_MODEL, FORCE_REBUILD)
