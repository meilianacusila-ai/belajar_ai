
import os
import re
from datetime import datetime, date
from difflib import SequenceMatcher
from typing import TypedDict, List, Dict, Any, Optional, Literal, Tuple

import streamlit as st
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    PayloadSchemaType,
)

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import StateGraph, END

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="CSO Asuransi Kesehatan", page_icon="🏥")
st.title("🏥 Chatbot CSO Asuransi Kesehatan (Grounded)")

# ============================================================
# ENV
# ============================================================
load_dotenv()

def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        st.error(f"ENV '{name}' belum di-set. Pastikan .env berisi {name}=...")
        st.stop()
    return v

QDRANT_URL = require_env("QDRANT_URL")
QDRANT_API_KEY = require_env("QDRANT_API_KEY")
OPENAI_API_KEY = require_env("OPENAI_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
emb = OpenAIEmbeddings(model="text-embedding-3-small")

# Optional LLM (hanya merapikan redaksi dari jawaban final, dilarang tambah info)
LLM_REPHRASE_PROMPT = """
Kamu adalah CSO Asuransi Kesehatan yang sopan, ramah, dan ringkas.

ATURAN KERAS:
- Kamu HANYA boleh merapikan redaksi dari teks jawaban yang sudah ada.
- DILARANG menambah fakta, angka, nama RS, plan, limit, tanggal, atau prosedur baru.
- Jangan menambahkan informasi yang tidak ada di teks asli.
- Jika teks asli menyatakan "tidak ditemukan", pertahankan maknanya.

Keluaran: versi yang lebih enak dibaca, singkat, jelas, dan tetap sopan.
"""

llm_rephrase = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# MEMORY (min 5 percakapan terakhir + memo slot)
# ============================================================
MAX_TURNS = 5

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []
if "memo" not in st.session_state:
    st.session_state.memo = {
        "no_polis": None,
        "kota": None,
        "pending_slot": None,
        "last_intent": None,
        "last_rs_mode": "all",
        "topics": [],
        "nasabah_key_no_polis": None,
        "pending_plan_for_benefit": None,
        "greeted": False,
    }

def remember(role: str, content: str):
    st.session_state.chat_memory.append({"role": role, "content": content})
    keep = 2 * MAX_TURNS
    if len(st.session_state.chat_memory) > keep:
        st.session_state.chat_memory = st.session_state.chat_memory[-keep:]

def memo_set(k: str, v: Optional[str]):
    if v:
        st.session_state.memo[k] = v

def memo_add_topic(topic: str):
    if topic and topic not in st.session_state.memo["topics"]:
        st.session_state.memo["topics"].append(topic)
        st.session_state.memo["topics"] = st.session_state.memo["topics"][-10:]

# ============================================================
# TYPES
# ============================================================
Intent = Literal[
    "rs_search",
    "policy_status",
    "policy_plan_lookup",
    "cashless_policy",
    "limit_plan",
    "claim_requirements",
    "plan_benefit",
    "provide_policy_number",
    "unknown",
]

RSMode = Literal["cashless", "non_cashless", "all"]

class GraphState(TypedDict, total=False):
    user_query: str
    intent: Intent

    no_polis: Optional[str]
    kota: Optional[str]
    plan_asked: Optional[str]      # Silver/Gold/Platinum
    rs_mode: RSMode

    missing_fields: List[str]

    nasabah: Optional[Dict[str, Any]]
    rs_list: Optional[List[Dict[str, Any]]]
    polis_evidence: Optional[List[Dict[str, Any]]]

    debug: Dict[str, Any]
    decision: Dict[str, Any]
    answer: str

# ============================================================
# UTIL
# ============================================================
def norm(x: Any) -> str:
    return str(x or "").strip()

def low(x: Any) -> str:
    return norm(x).lower()

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def compact(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())

def normalize_reimburse_words(text: str) -> str:
    t = low(text)
    t = t.replace("rembures", "reimburse").replace("reimbures", "reimburse").replace("remburse", "reimburse")
    t = t.replace("reimburs", "reimburse").replace("reimbus", "reimburse")
    return t

def cashless_to_ya_tidak(v: Any) -> str:
    s = low(v)
    if s in ("ya", "y", "true", "1"):
        return "Ya"
    if s in ("tidak", "t", "no", "false", "0"):
        return "Tidak"
    return "-"

def parse_date_any(v: Any) -> Optional[date]:
    s = norm(v)
    if not s:
        return None

    fmts = [
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y",
        "%Y/%m/%d", "%d %m %Y", "%Y.%m.%d"
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f).date()
        except Exception:
            pass

    m = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            return None

    return None

# ============================================================
# no_polis normalization
# ============================================================
def normalize_no_polis(raw: str) -> str:
    if not raw:
        return raw
    return compact(raw)

def no_polis_variants(raw: str) -> List[str]:
    if not raw:
        return []
    raw_u = raw.strip().upper()
    c = normalize_no_polis(raw_u)
    variants = {raw_u, c}

    m = re.match(r"^([A-Z]{2,6})(\d+)$", c)
    if m:
        prefix, digits = m.group(1), m.group(2)
        if len(digits) >= 7:
            variants.add(f"{prefix}-{digits[:3]}-{digits[3:]}")
            variants.add(f"{prefix}{digits}")
        if len(digits) >= 6:
            variants.add(f"{prefix}-{digits[:2]}-{digits[2:]}")
    return [v for v in variants if v]

# ============================================================
# ENTITY EXTRACTION
# ============================================================
KNOWN_CITIES = ["jakarta", "bandung", "surabaya", "medan", "semarang", "yogyakarta", "makassar", "denpasar"]

def extract_city(text: str) -> Optional[str]:
    t = low(text)
    for c in KNOWN_CITIES:
        if re.search(rf"(?:\bdi[\s\-]*|^){re.escape(c)}\b", t) or re.search(rf"\b{re.escape(c)}\b", t):
            return c.capitalize()
    if "dki jakarta" in t:
        return "Jakarta"
    return None

def extract_entities(text: str) -> Dict[str, Optional[str]]:
    polis_match = re.search(r"\b([A-Z]{2,6}\-?\d{2,6}\-?\d{0,4}|\d{6,})\b", text.upper())
    no_polis = polis_match.group(1) if polis_match else None
    kota = extract_city(text)

    plan_asked = None
    if re.search(r"\bplatinum\b", text, flags=re.IGNORECASE):
        plan_asked = "Platinum"
    elif re.search(r"\bgold\b", text, flags=re.IGNORECASE):
        plan_asked = "Gold"
    elif re.search(r"\bsilver\b", text, flags=re.IGNORECASE):
        plan_asked = "Silver"

    return {"no_polis": no_polis, "kota": kota, "plan_asked": plan_asked}

def apply_slot_filling(user_text: str, state: GraphState) -> GraphState:
    pending = st.session_state.memo.get("pending_slot")
    ents = extract_entities(user_text)

    if pending == "kota" and ents.get("kota"):
        state["kota"] = ents["kota"]
        st.session_state.memo["pending_slot"] = None
    if pending == "no_polis" and ents.get("no_polis"):
        state["no_polis"] = ents["no_polis"]
        st.session_state.memo["pending_slot"] = None
    if pending == "plan_asked" and ents.get("plan_asked"):
        state["plan_asked"] = ents["plan_asked"]
        st.session_state.memo["pending_slot"] = None

    return state

# ============================================================
# INDEX CREATION (best-effort)
# ============================================================
def ensure_payload_indexes():
    for field in ["kota", "nama_rs", "cashless"]:
        try:
            client.create_payload_index("rs_rekanan", field, PayloadSchemaType.KEYWORD)
        except Exception:
            pass
    for field in ["no_polis", "No Polis", "nomor_polis", "policy_no", "plan", "status_polis", "metode_klaim"]:
        try:
            client.create_payload_index("nasabah", field, PayloadSchemaType.KEYWORD)
        except Exception:
            pass

ensure_payload_indexes()

# ============================================================
# AUTO-DETECT KEY "no_polis" in NASABAH payload
# ============================================================
NO_POLIS_CANDIDATE_KEYS = [
    "no_polis", "nomor_polis", "policy_no", "policy_number",
    "no polis", "nomor polis", "polis", "noPolicy"
]

def detect_nasabah_no_polis_key() -> Optional[str]:
    try:
        hits, _ = client.scroll("nasabah", limit=1, with_payload=True, with_vectors=False)
        if not hits:
            return None
        payload = hits[0].payload or {}
        if not payload:
            return None

        best = None
        best_score = 0.0
        for k in payload.keys():
            for cand in NO_POLIS_CANDIDATE_KEYS:
                sc = similar(str(k), cand)
                if sc > best_score:
                    best_score = sc
                    best = k
        return best if best_score >= 0.66 else None
    except Exception:
        return None

if not st.session_state.memo.get("nasabah_key_no_polis"):
    st.session_state.memo["nasabah_key_no_polis"] = detect_nasabah_no_polis_key()

# ============================================================
# AUTO-FIND FIELD VALUE BY ALIAS
# ============================================================
def find_value_by_alias(payload: Dict[str, Any], aliases: List[str]) -> Tuple[Optional[str], Optional[Any], float]:
    if not payload:
        return None, None, 0.0
    best_key = None
    best_score = 0.0
    for k in payload.keys():
        for a in aliases:
            sc = similar(str(k), a)
            if sc > best_score:
                best_score = sc
                best_key = k
    if best_key and best_score >= 0.66:
        val = payload.get(best_key)
        if val is None or norm(val) == "":
            return best_key, None, best_score
        return best_key, val, best_score
    return None, None, best_score

# ============================================================
# INTENT
# ============================================================
FILLER_WORDS = {"klo", "kalau", "kalo", "ini", "itu", "ya", "yah", "deh", "dong", "nih", "gimana", "maksudnya"}

def extract_policy_like_text(user_text: str) -> str:
    t = low(user_text)
    t = re.sub(r"[^\w\s\-]", " ", t)
    toks = [x for x in t.split() if x and x not in FILLER_WORDS]
    return " ".join(toks)

def detect_rs_mode(q: str) -> Optional[RSMode]:
    t = normalize_reimburse_words(q)
    if any(x in t for x in ["gak cashless", "ga cashless", "tidak cashless", "non cashless", "reimburse", "reimbursement", "refund"]):
        return "non_cashless"
    if "cashless" in t:
        return "cashless"
    return None

def is_policy_only_with_fillers(user_text: str, no_polis: str) -> bool:
    core = extract_policy_like_text(user_text)
    return (compact(no_polis) in compact(core)) and (len(compact(core)) <= len(compact(no_polis)) + 2)

def classify_intent(q: str) -> Intent:
    t = normalize_reimburse_words(q)
    ents = extract_entities(q)

    if st.session_state.memo.get("pending_slot") is not None:
        return st.session_state.memo.get("last_intent") or "unknown"

    # "klo POL-002-2024?" -> provide policy number
    if ents.get("no_polis") and is_policy_only_with_fillers(q, ents["no_polis"]):
        return "provide_policy_number"

    # benefit plan
    if any(x in t for x in ["manfaat plan", "benefit plan", "dapet apa", "dapat apa", "benefitnya", "manfaatnya"]) and any(p in t for p in ["silver", "gold", "platinum"]):
        return "plan_benefit"

    # claim
    if any(k in t for k in ["persyaratan klaim", "syarat klaim", "dokumen klaim", "cara klaim", "klaim", "claim", "reimburse", "reimbursement"]):
        return "claim_requirements"

    # status
    if any(k in t for k in ["masih hidup", "masih aktif", "aktif?", "status polis", "polis saya aktif", "apakah masih aktif"]):
        return "policy_status"

    # rs
    if any(x in t for x in ["rs", "rumah sakit", "rs rekanan", "rekomendasi rs"]):
        return "rs_search"

    # plan polis
    if any(x in t for x in ["plan apa", "termasuk plan", "plan saya", "masuk plan apa"]):
        return "policy_plan_lookup"

    # cashless polis
    if "cashless" in t:
        return "cashless_policy"

    # limit plan
    if "limit" in t and any(p in t for p in ["platinum", "gold", "silver"]):
        return "limit_plan"

    return "unknown"

def required_fields(intent: Intent) -> List[str]:
    if intent == "rs_search":
        return ["kota"]
    if intent in ("policy_status", "policy_plan_lookup", "cashless_policy"):
        return ["no_polis"]
    if intent in ("limit_plan", "plan_benefit"):
        return ["plan_asked"]
    return []

# ============================================================
# TEXT EXTRACTION FOR PLAN (ONLY ONE PLAN)
# ============================================================
def extract_only_plan_section(text: str, plan: str) -> str:
    if not text:
        return ""

    t = " ".join(text.split())
    plan_u = (plan or "").strip().upper()
    if not plan_u:
        return ""

    # cari baris plan: "Platinum Rp200.000.000 Sesuai tagihan ..."
    m = re.search(rf"\b{re.escape(plan_u)}\b\s+(.{{0,220}})", t, flags=re.IGNORECASE)
    if m:
        return f"{plan_u} {m.group(1)}".strip()

    # fallback: cuplikan sekitar plan
    idx = t.lower().find(plan.lower())
    if idx != -1:
        start = max(0, idx - 120)
        end = min(len(t), idx + 220)
        return t[start:end].strip()

    return ""

# ============================================================
# POLIS LIMIT PICKER
# ============================================================
def pick_best_limit_chunk(evs: List[Dict[str, Any]], plan: str) -> Optional[Dict[str, Any]]:
    plan_l = low(plan)
    best = None
    best_score = -1
    for e in evs:
        txt = low(e.get("text"))
        if not txt:
            continue
        score = 0
        if plan_l and plan_l in txt:
            score += 3
        if "limit" in txt:
            score += 2
        if "tahunan" in txt:
            score += 2
        if "rawat" in txt or "icu" in txt:
            score += 1
        if "rp" in txt:
            score += 2
        if score > best_score:
            best_score = score
            best = e
    return best if best_score >= 4 else None

# ============================================================
# TOOLS
# ============================================================
def tool_lookup_nasabah(no_polis_raw: str) -> Dict[str, Any]:
    key = st.session_state.memo.get("nasabah_key_no_polis") or "no_polis"
    tried = no_polis_variants(no_polis_raw)

    for v in tried:
        try:
            flt = Filter(must=[FieldCondition(key=key, match=MatchValue(value=v))])
            hits, _ = client.scroll("nasabah", scroll_filter=flt, limit=1, with_payload=True, with_vectors=False)
            if hits and hits[0].payload:
                return {"found": True, "nasabah": hits[0].payload, "tried": tried, "key_used": key}
        except Exception:
            continue

    # fallback semantic
    try:
        qvec = emb.embed_query(normalize_no_polis(no_polis_raw))
        res = client.query_points("nasabah", query=qvec, limit=3, with_payload=True)
        pts = res.points if hasattr(res, "points") else res[0]
        if pts and pts[0].payload:
            return {"found": True, "nasabah": pts[0].payload, "tried": tried, "key_used": key}
    except Exception:
        pass

    return {"found": False, "nasabah": None, "tried": tried, "key_used": key}

def tool_lookup_rs(kota: str, rs_mode: RSMode) -> List[Dict[str, Any]]:
    must = [FieldCondition(key="kota", match=MatchValue(value=kota))]
    if rs_mode == "cashless":
        must.append(FieldCondition(key="cashless", match=MatchValue(value="Ya")))
    elif rs_mode == "non_cashless":
        must.append(FieldCondition(key="cashless", match=MatchValue(value="Tidak")))

    try:
        flt = Filter(must=must)
        hits, _ = client.scroll("rs_rekanan", scroll_filter=flt, limit=50, with_payload=True, with_vectors=False)
        payloads = [p.payload for p in hits if p.payload]
        cleaned = []
        for pl in payloads:
            nama_rs = pl.get("nama_rs") or pl.get("nama") or pl.get("rumah_sakit") or pl.get("rs")
            if nama_rs:
                cleaned.append(pl)
        return cleaned[:10]
    except Exception:
        return []

def tool_rag_polis(query: str, k: int = 30) -> List[Dict[str, Any]]:
    qvec = emb.embed_query(query)
    res = client.query_points("polis", query=qvec, limit=k, with_payload=True)
    pts = res.points if hasattr(res, "points") else res[0]
    out = []
    for p in pts or []:
        payload = p.payload or {}
        text = payload.get("text") or payload.get("page_content") or payload.get("content") or ""
        out.append({
            "page": payload.get("page"),
            "source": payload.get("source_file") or payload.get("source"),
            "text": text
        })
    return out

# ============================================================
# ANSWER DIRECT (ringkas & grounded)
# ============================================================
def answer_from_decision(d: Dict[str, Any]) -> str:
    status = d.get("status")

    if status == "NEED_INPUT":
        need = d.get("need") or []
        if "kota" in need:
            return "Boleh sebutkan **kota** yang ingin dicek RS rekanannya? 😊"
        if "no_polis" in need:
            return "Untuk saya cek di database, saya perlu **no_polis**. Boleh kirim ya 😊"
        if "plan_asked" in need:
            return "Anda ingin plan yang mana? **Silver / Gold / Platinum** 😊"
        return f"Saya perlu data: {', '.join(need)}"

    if status == "NEED_CHOICE":
        np = d.get("no_polis")
        return (
            f"Siap 😊 No polis **{np}** sudah saya catat.\n"
            "Mau dicek: **status / plan / cashless**?"
        )

    if d.get("intent") == "policy_status":
        if status == "FOUND":
            return f"Status polis **{d.get('no_polis_input')}** (database nasabah): **{d.get('status_polis')}**."
        if status == "NOT_FOUND":
            return f"Maaf, **{d.get('no_polis_input')}** tidak ditemukan di database nasabah."
        if status == "MISSING_FIELD":
            return f"Data polis **{d.get('no_polis_input')}** ditemukan, tapi **status/masa berlaku** tidak tersedia di database nasabah."

    if d.get("intent") == "policy_plan_lookup":
        if status == "FOUND":
            return f"Plan polis **{d.get('no_polis_input')}** (database nasabah): **{d.get('plan')}**."
        if status == "NOT_FOUND":
            return f"Maaf, **{d.get('no_polis_input')}** tidak ditemukan di database nasabah."
        if status == "MISSING_FIELD":
            return f"Data polis **{d.get('no_polis_input')}** ditemukan, tapi field **plan** kosong di database nasabah."

    if d.get("intent") == "cashless_policy":
        if status == "FOUND":
            can = "bisa" if d.get("can_cashless") else "tidak"
            return (
                f"Metode klaim polis **{d.get('no_polis_input')}** (database nasabah): **{d.get('metode_klaim')}**. "
                f"Kesimpulan: **{can} cashless**."
            )
        if status == "NOT_FOUND":
            return f"Maaf, **{d.get('no_polis_input')}** tidak ditemukan di database nasabah."
        if status == "MISSING_FIELD":
            return "Data polis ditemukan, tapi **metode_klaim** kosong di database sehingga cashless belum bisa dipastikan."

    if d.get("intent") == "rs_search":
        if status == "FOUND":
            mode = d.get("rs_mode", "all")
            mode_txt = "cashless" if mode == "cashless" else ("non-cashless (reimburse)" if mode == "non_cashless" else "semua")
            lines = [f"RS rekanan di **{d.get('kota')}** ({mode_txt}) (database):"]
            for it in d.get("rs_items", []):
                lines.append(f"- {it.get('nama_rs')} | cashless: {it.get('cashless')}")
            return "\n".join(lines)
        return f"Maaf, saya belum menemukan RS rekanan di database untuk kota **{d.get('kota')}**."

    if d.get("intent") == "limit_plan":
        if status == "FOUND":
            plan = d.get("plan") or ""
            raw = norm(d.get("quote"))
            plan_only = extract_only_plan_section(raw, plan) or " ".join(raw.split())[:220]
            return (
                f"Limit plan **{plan}** (rujukan buku polis):\n"
                f"- Hal: {d.get('page')} | Sumber: {d.get('source')}\n"
                f"- Kutipan: {plan_only}"
            )
        return "Maaf, bagian limit plan tersebut **belum ditemukan** di buku polis pada data yang tersimpan."

    if d.get("intent") == "plan_benefit":
        if status == "FOUND":
            plan = d.get("plan") or ""
            raw = norm(d.get("quote"))
            plan_only = extract_only_plan_section(raw, plan)
            if not plan_only:
                plan_only = " ".join(raw.split())[:220]
            return (
                f"Manfaat/ketentuan terkait plan **{plan}** (rujukan buku polis):\n"
                f"- Hal: {d.get('page')} | Sumber: {d.get('source')}\n"
                f"- Kutipan: {plan_only}"
            )
        return "Maaf, bagian manfaat plan tersebut **belum ditemukan** di buku polis pada data yang tersimpan."

    if d.get("intent") == "claim_requirements":
        if status == "FOUND":
            quote = " ".join(norm(d.get("quote")).split())[:420]
            return f"Cara/ketentuan klaim (rujukan buku polis):\n- Hal: {d.get('page')} | Sumber: {d.get('source')}\n- Kutipan: {quote}"
        return "Maaf, bagian **cara/ketentuan klaim** belum ditemukan di buku polis pada data yang tersimpan."

    return "Maaf, saya belum bisa menjawab dari database yang tersedia. Bisa jelaskan maksudnya sedikit? 😊"

# ============================================================
# Greeting & Closing (FIX)
# ============================================================
def should_greet() -> bool:
    return not st.session_state.memo.get("greeted", False)

def greet_text() -> str:
    return (
        "Halo 😊 Saya CSO Asuransi Kesehatan.\n\n"
    )

def is_closing_message(text: str) -> bool:
    t = (text or "").strip().lower()
    if "?" in t or any(w in t for w in ["gimana", "bagaimana", "cara", "syarat", "kenapa", "kapan", "berapa", "apa", "apakah"]):
        return False
    closing_phrases = [
        "terima kasih", "makasih", "thanks", "thank you",
        "oke makasih", "ok makasih", "sip makasih",
        "selesai", "sudah cukup", "udah cukup", "cukup"
    ]
    return any(p in t for p in closing_phrases)

# ============================================================
# NODES
# ============================================================
def ensure_dict(node_name, fn):
    def wrapped(state: GraphState) -> GraphState:
        out = fn(state)
        if not isinstance(out, dict):
            raise TypeError(f"[{node_name}] must return dict")
        return out
    return wrapped

def supervisor_node(state: GraphState) -> GraphState:
    q = state["user_query"]
    state = apply_slot_filling(q, state)
    ents = extract_entities(q)

    # carry from memo
    state["no_polis"] = state.get("no_polis") or ents.get("no_polis") or st.session_state.memo.get("no_polis")
    state["kota"] = state.get("kota") or ents.get("kota") or st.session_state.memo.get("kota")
    state["plan_asked"] = state.get("plan_asked") or ents.get("plan_asked")

    state["intent"] = classify_intent(q)
    st.session_state.memo["last_intent"] = state["intent"]

    memo_set("no_polis", state.get("no_polis"))
    memo_set("kota", state.get("kota"))

    rm = detect_rs_mode(q)
    if rm:
        state["rs_mode"] = rm
        st.session_state.memo["last_rs_mode"] = rm
    else:
        state["rs_mode"] = st.session_state.memo.get("last_rs_mode") or "all"

    state["debug"] = {"nasabah_key_no_polis": st.session_state.memo.get("nasabah_key_no_polis")}

    # if user asks plan_benefit and plan_asked detected, store pending plan for display
    if state["intent"] == "plan_benefit" and state.get("plan_asked"):
        st.session_state.memo["pending_plan_for_benefit"] = state["plan_asked"]

    return state

def requirements_node(state: GraphState) -> GraphState:
    need = required_fields(state["intent"])
    missing = [f for f in need if not state.get(f)]
    state["missing_fields"] = missing
    st.session_state.memo["pending_slot"] = missing[0] if missing else None
    return state

def nasabah_node(state: GraphState) -> GraphState:
    state["nasabah"] = None
    if state.get("no_polis"):
        res = tool_lookup_nasabah(state["no_polis"])
        state["nasabah"] = res["nasabah"]
        state["debug"].update({
            "nasabah_key_used": res["key_used"],
            "nasabah_tried": res["tried"],
            "nasabah_found": res["found"]
        })
    return state

def rs_node(state: GraphState) -> GraphState:
    kota = state.get("kota")
    state["rs_list"] = tool_lookup_rs(kota, state.get("rs_mode") or "all") if kota else []
    return state

def polis_node(state: GraphState) -> GraphState:
    intent = state["intent"]

    # FIX: query klaim lebih tajam
    if intent == "claim_requirements":
        q = (
            "prosedur klaim cara klaim langkah klaim dokumen klaim formulir klaim "
            "resume medis kwitansi rincian biaya batas waktu pengajuan "
            "klaim cashless verifikasi rumah sakit rekanan klaim reimbursement"
        )
        state["polis_evidence"] = tool_rag_polis(q, k=35)
        return state

    if intent == "limit_plan":
        plan = state.get("plan_asked") or ""
        q1 = f"BAB IV LIMIT DAN PLAN {plan} Limit Tahunan Rawat Inap ICU Rawat Jalan Rp"
        evs = tool_rag_polis(q1, k=35)
        best = pick_best_limit_chunk(evs, plan)
        if not best:
            q2 = f"LIMIT DAN PLAN {plan} Limit Tahunan Rp Rawat"
            state["polis_evidence"] = tool_rag_polis(q2, k=40)
        else:
            state["polis_evidence"] = evs
        return state

    if intent == "plan_benefit":
        plan = state.get("plan_asked") or st.session_state.memo.get("pending_plan_for_benefit") or ""
        q = f"manfaat {plan} plan {plan} rawat inap rawat jalan icu manfaat tambahan"
        state["polis_evidence"] = tool_rag_polis(q, k=40)
        return state

    state["polis_evidence"] = tool_rag_polis(state["user_query"], k=30)
    return state

def decision_node(state: GraphState) -> GraphState:
    intent = state["intent"]
    missing = state.get("missing_fields") or []
    d: Dict[str, Any] = {"intent": intent}

    if missing:
        d["status"] = "NEED_INPUT"
        d["need"] = missing
        state["decision"] = d
        return state

    if intent == "provide_policy_number":
        d["status"] = "NEED_CHOICE"
        d["no_polis"] = state.get("no_polis")
        state["decision"] = d
        return state

    if intent == "policy_status":
        memo_add_topic("Status polis")
        d["no_polis_input"] = state.get("no_polis")
        nas = state.get("nasabah")

        if not nas:
            d["status"] = "NOT_FOUND"
            state["decision"] = d
            return state

        _, status_val, _ = find_value_by_alias(
            nas,
            ["status_polis", "status polis", "policy_status", "status", "aktif", "is_active", "active_status"]
        )
        if status_val is not None:
            d["status"] = "FOUND"
            d["status_polis"] = norm(status_val)
            state["decision"] = d
            return state

        _, end_val, _ = find_value_by_alias(
            nas,
            ["tanggal_akhir", "akhir_polis", "masa_berlaku_sampai", "expired_date", "expiry_date",
             "end_date", "tanggal_expired", "tgl_akhir", "valid_until", "berlaku_sampai"]
        )
        end_dt = parse_date_any(end_val)
        if end_dt:
            today = date.today()
            if end_dt >= today:
                d["status"] = "FOUND"
                d["status_polis"] = f"Aktif (berdasarkan tanggal akhir {end_dt.isoformat()})"
            else:
                d["status"] = "FOUND"
                d["status_polis"] = f"Tidak aktif (berdasarkan tanggal akhir {end_dt.isoformat()})"
            state["decision"] = d
            return state

        d["status"] = "MISSING_FIELD"
        state["decision"] = d
        return state

    if intent == "policy_plan_lookup":
        memo_add_topic("Plan polis")
        d["no_polis_input"] = state.get("no_polis")
        nas = state.get("nasabah")

        if not nas:
            d["status"] = "NOT_FOUND"
            state["decision"] = d
            return state

        _, val, _ = find_value_by_alias(nas, ["plan", "jenis_plan", "plan polis", "policy_plan", "tipe_plan"])
        if val is None:
            d["status"] = "MISSING_FIELD"
        else:
            d["status"] = "FOUND"
            d["plan"] = norm(val)

        state["decision"] = d
        return state

    if intent == "cashless_policy":
        memo_add_topic("Cashless")
        d["no_polis_input"] = state.get("no_polis")
        nas = state.get("nasabah")

        if not nas:
            d["status"] = "NOT_FOUND"
            state["decision"] = d
            return state

        _, val, _ = find_value_by_alias(nas, ["metode_klaim", "metode klaim", "claim_method", "jenis_klaim", "cashless"])
        if val is None:
            d["status"] = "MISSING_FIELD"
        else:
            d["status"] = "FOUND"
            d["metode_klaim"] = norm(val)
            d["can_cashless"] = ("cashless" in low(val)) or (low(val) in ("ya", "true", "1"))

        state["decision"] = d
        return state

    if intent == "rs_search":
        memo_add_topic("RS rekanan")
        rs_list = state.get("rs_list") or []
        d["status"] = "FOUND" if rs_list else "NOT_FOUND"
        d["kota"] = state.get("kota")
        d["rs_mode"] = state.get("rs_mode")
        d["rs_items"] = []
        for pl in rs_list[:5]:
            nama = pl.get("nama_rs") or pl.get("nama") or pl.get("rumah_sakit") or pl.get("rs")
            d["rs_items"].append({
                "nama_rs": nama,
                "cashless": cashless_to_ya_tidak(pl.get("cashless")),
            })
        state["decision"] = d
        return state

    # FIX: scoring polis evidence (claim penalti "LIMIT DAN PLAN"; plan_benefit tampil plan-only)
    if intent in ("claim_requirements", "limit_plan", "plan_benefit"):
        evs = state.get("polis_evidence") or []
        best = None
        best_score = -10

        plan = state.get("plan_asked") or st.session_state.memo.get("pending_plan_for_benefit") or ""
        plan_l = low(plan)

        for e in evs:
            txt = low(e.get("text"))
            if not txt:
                continue

            score = 0

            if intent == "claim_requirements":
                for kw in ["klaim", "claim", "cashless", "reimbursement", "dokumen", "formulir", "kwitansi", "resume", "verifikasi", "pengajuan", "batas waktu"]:
                    if kw in txt:
                        score += 2
                if "limit" in txt and "plan" in txt:
                    score -= 4
                if "bab iv" in txt or "limit dan plan" in txt:
                    score -= 6

            elif intent == "limit_plan":
                if plan_l and plan_l in txt:
                    score += 4
                for kw in ["limit", "tahunan", "rawat", "icu", "rp"]:
                    if kw in txt:
                        score += 2

            elif intent == "plan_benefit":
                if plan_l and plan_l in txt:
                    score += 3
                for kw in ["manfaat", "rawat", "icu", "persalinan", "kritis"]:
                    if kw in txt:
                        score += 2
                if "limit dan plan" in txt and "manfaat" not in txt:
                    score -= 2

            if len(txt) > 300:
                score += 1

            if score > best_score:
                best_score = score
                best = e

        min_ok = 3 if intent == "claim_requirements" else 1

        if not best or best_score < min_ok:
            d["status"] = "NOT_FOUND"
        else:
            d["status"] = "FOUND"
            d["source"] = best.get("source")
            d["page"] = best.get("page")
            d["quote"] = norm(best.get("text"))[:900]
            if intent in ("limit_plan", "plan_benefit"):
                d["plan"] = plan

        state["decision"] = d
        return state

    d["status"] = "UNKNOWN"
    state["decision"] = d
    return state

def compose_node(state: GraphState) -> GraphState:
    base = answer_from_decision(state.get("decision") or {})

    # optional: rephrase only (no new info)
    use_llm = st.session_state.get("use_llm_rephrase", True)
    if use_llm and base:
        try:
            msg = f"{LLM_REPHRASE_PROMPT}\n\nTEKS ASLI:\n{base}\n\nTEKS RAPi:\n"
            out = llm_rephrase.invoke(msg)
            cleaned = (out.content or "").strip()
            # guard: kalau output kosong, fallback
            state["answer"] = cleaned if cleaned else base
        except Exception:
            state["answer"] = base
    else:
        state["answer"] = base

    return state

# ============================================================
# ROUTING
# ============================================================
def route_after_requirements(state: GraphState) -> str:
    if state.get("missing_fields"):
        return "decision"
    intent = state["intent"]
    if intent == "rs_search":
        return "rs"
    if intent in ("policy_status", "policy_plan_lookup", "cashless_policy"):
        return "nasabah"
    if intent in ("limit_plan", "claim_requirements", "plan_benefit"):
        return "polis"
    return "decision"

# ============================================================
# BUILD GRAPH
# ============================================================
g = StateGraph(GraphState)
g.add_node("supervisor", ensure_dict("supervisor", supervisor_node))
g.add_node("requirements", ensure_dict("requirements", requirements_node))
g.add_node("nasabah", ensure_dict("nasabah", nasabah_node))
g.add_node("rs", ensure_dict("rs", rs_node))
g.add_node("polis", ensure_dict("polis", polis_node))
g.add_node("decision", ensure_dict("decision", decision_node))
g.add_node("compose", ensure_dict("compose", compose_node))

g.set_entry_point("supervisor")
g.add_edge("supervisor", "requirements")
g.add_conditional_edges("requirements", route_after_requirements, {
    "rs": "rs",
    "nasabah": "nasabah",
    "polis": "polis",
    "decision": "decision",
})
g.add_edge("rs", "decision")
g.add_edge("nasabah", "decision")
g.add_edge("polis", "decision")
g.add_edge("decision", "compose")
g.add_edge("compose", END)

app = g.compile()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Mode")
    show_debug = st.toggle("Tampilkan debug", value=True)
    st.session_state.use_llm_rephrase = st.toggle("Rapiin jawaban pakai LLM (no new info)", value=True)
    st.caption(f"Memory aktif: {MAX_TURNS} percakapan terakhir")
    st.write("Memo:", st.session_state.memo)

# ============================================================
# Render history
# ============================================================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Greeting (once)
if should_greet():
    gtxt = greet_text()
    st.session_state.messages.append({"role": "assistant", "content": gtxt})
    remember("assistant", gtxt)
    st.session_state.memo["greeted"] = True
    with st.chat_message("assistant"):
        st.markdown(gtxt)

# ============================================================
# Chat input
# ============================================================
user_input = st.chat_input("Tulis pertanyaan kamu...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    remember("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Closing only if truly closing
    if is_closing_message(user_input):
        closing = "Sama-sama 😊 Senang bisa membantu. Jika ada pertanyaan lain, silakan chat lagi ya."
        st.session_state.messages.append({"role": "assistant", "content": closing})
        remember("assistant", closing)
        with st.chat_message("assistant"):
            st.markdown(closing)
        st.stop()

    try:
        out = app.invoke({"user_query": user_input})
    except Exception as e:
        st.error("Terjadi error saat memproses permintaan.")
        st.exception(e)
        st.stop()

    ans = out.get("answer", "Maaf, saya belum bisa menjawab saat ini.")
    st.session_state.messages.append({"role": "assistant", "content": ans})
    remember("assistant", ans)

    with st.chat_message("assistant"):
        st.markdown(ans)

        if show_debug:
            with st.expander("🔎 Debug"):
                st.write("Intent:", out.get("intent"))
                st.write("Missing:", out.get("missing_fields"))
                st.write("No Polis:", out.get("no_polis"))
                st.write("Kota:", out.get("kota"))
                st.write("Plan asked:", out.get("plan_asked"))
                st.write("RS mode:", out.get("rs_mode"))
                st.write("Nasabah found:", bool(out.get("nasabah")))
                st.write("Decision:", out.get("decision"))
