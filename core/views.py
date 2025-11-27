# path: core/views.py
import os, json, time, uuid, re
from django.shortcuts import render, redirect
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.utils.safestring import mark_safe
from django.views.decorators.http import require_POST
from django.views.decorators.http import require_http_methods
from django.template.loader import render_to_string

# .env ачаалах
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from openai import OpenAI
import pickle
from rag_chainn import rag_answer   

# --- OpenAI / Data ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DF_PATH = os.path.join(BASE_DIR, "data", "df_emb_muis.pkl")
FEEDBACK_PATH = os.path.join(BASE_DIR, "data", "feedback.jsonl")

DF = None
if os.path.exists(DF_PATH):
    with open(DF_PATH, "rb") as f:
        DF = pickle.load(f)

# === RAG урт хариултын Тохиргоо ===
RAG_STYLE      = os.getenv("RAG_STYLE", "long")         # "long" | "short"
RAG_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "900"))
RAG_TOP_K      = int(os.getenv("RAG_TOP_K", "20"))
RAG_MAX_CTX    = int(os.getenv("RAG_MAX_CTX", "8"))
REFS_LIMIT     = int(os.getenv("RAG_REFS_LIMIT", "2"))

# --- Markdown → Plain text formatter (#, **, *… арилгана) ---
_MD_HASH = re.compile(r'^\s*#{1,6}\s*', flags=re.M)
_MD_BOLD = re.compile(r'\*\*(.+?)\*\*')
_MD_EMPH = re.compile(r'(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)')
_MD_CODE = re.compile(r'`{1,3}([^`]+?)`{1,3}')

def prettify_plain(s: str) -> str:
    if not s:
        return s
    s = _MD_HASH.sub('', s)
    s = _MD_BOLD.sub(r'\1', s)
    s = _MD_EMPH.sub(r'\1', s)
    s = _MD_CODE.sub(r'\1', s)
    # bullets: "-", "*" → "• "
    s = re.sub(r'^[ \t]*[-*]\s+', '• ', s, flags=re.M)
    # олон хоосон мөрийг 2-оос ихгүй байлгана
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

# --- Greetings ---
_GREET = {
    "hi","hello","hey","yo","sain uu","sain baina uu",
    "сайн уу","сайн байна уу","мэнд","сайн өдрийн мэнд",
    "yu bn","yu baina","andaa"
}
def _is_greeting(q: str) -> bool:
    s = q.lower().strip()
    return any(s.startswith(g) or s == g for g in _GREET)

# --- Session helpers (multi-chat) ---
def _ensure_state(request: HttpRequest):
    if "chats" not in request.session:
        request.session["chats"] = {}
    if (
        "current_chat" not in request.session
        or request.session["current_chat"] not in request.session["chats"]
    ):
        cid = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
        request.session["chats"][cid] = {
            "title": "New chat",
            "created": int(time.time()),
            "messages": [],
        }
        request.session["current_chat"] = cid
    request.session.modified = True


def _current_chat(request: HttpRequest):
    _ensure_state(request)
    cid = request.session["current_chat"]
    return cid, request.session["chats"][cid]


def _list_chats(request: HttpRequest):
    _ensure_state(request)
    chats = request.session["chats"]

    def last_ts(c):
        msgs = c.get("messages", [])
        return msgs[-1]["ts"] if msgs else c.get("created", 0)

    items = [(cid, cdata) for cid, cdata in chats.items()]
    items.sort(key=lambda t: last_ts(t[1]), reverse=True)
    return items


# === Эх сурвалжийн контекст шүүх ===

def _extract_keywords(text: str):
    """
    Асуултаас 4+ тэмдэгттэй гол үгсийг гаргана.
    Mongolian/Latin аль алинд нь ажиллана.
    """
    tokens = re.split(r"[^\wА-Яа-яҮүӨөЁё]+", text.lower())
    return [t for t in tokens if len(t) >= 4]


def _filter_contexts_for_refs(contexts, question: str, limit: int):
    """
    Эх сурвалжид оруулах контекстүүдийг шүүнэ.

    Шүүлт:
      1) c['score'] нь хамгийн өндөр онооны ≥ 75% байх
      2) Асуултын гол үгсээс ядаж нэг нь тухайн context текстэд агуулагдсан байх
    Ингэж шүүгээд хоосон бол эхний limit context-оо (score дарааллаар) fallback болгон авна.
    """
    if not contexts:
        return []

    keywords = _extract_keywords(question)
    max_score = max(float(c.get("score", 0.0)) for c in contexts) or 0.0
    score_cut = max_score * 0.75 if max_score > 0 else 0.0

    filtered = []
    for c in contexts:
        if len(filtered) >= limit:
            break

        score = float(c.get("score", 0.0))
        if score < score_cut:
            continue

        text = (c.get("text") or "").lower()
        if keywords:
            if not any(k in text for k in keywords):
                continue

        filtered.append(c)

    if not filtered:
        # оноо муу эсвэл keyword огт таараагүй бол – худал citation хийхээс зайлсхийж,
        # ерөөсөө эх сурвалж хэсэг нэмэхгүй байж болно.
        return []

    return filtered[:limit]


def _clean_source(raw_source: str, title: str) -> str:
    """
    OCR / chunk замаас жинхэнэ баримтын нэрийг гаргах.
    Жишээ: 'sisiData/NUM2121-...-журам.docx' → 'NUM2121-...-журам'
    """
    if not raw_source:
        return title or "(unknown)"

    base = os.path.basename(raw_source)

    # PDF/DOCX нэр бол шууд ашиглана
    if base.lower().endswith((".pdf", ".docx")):
        return base.replace(".docx", "").replace(".pdf", "")

    # өөр өргөтгөл байвал сууриар нь авна
    if "." in base:
        base = base.split(".")[0]

    return base if base else (title or "(unknown)")


# ======================== VIEWS ========================

def chat(request: HttpRequest) -> HttpResponse:
    """Гол нүүр. Серверийн бэлэн эсэх болон одоогийн чатын түүхийг дамжуулна."""
    _ensure_state(request)
    cid, cur = _current_chat(request)
    ctx = {
        "ready": bool(CLIENT and DF is not None),
        "reason": None
        if (CLIENT and DF is not None)
        else ("OPENAI_API_KEY тохируулаагүй" if not CLIENT else "data/df_emb_muis.pkl олдсонгүй"),
        "history": cur.get("messages", []),
        "chats": _list_chats(request),
        "current_chat": cid,
        "search_q": "",
    }
    return render(request, "core/chat.html", ctx)


@require_POST
def ask(request: HttpRequest) -> HttpResponse:
    """
    Клиент талд user bubble + typing placeholder зурдаг.
    Сервер бодит хариултыг OOB swap-аар placeholder-ийг сольж буцаана.
    """
    if not (CLIENT and DF is not None):
        return JsonResponse({"error": "Server not ready"}, status=500)

    _ensure_state(request)
    cid, cur = _current_chat(request)

    q = (request.POST.get("q") or "").strip()
    ph = (request.POST.get("ph") or "").strip()  
    ts = int(time.time())
    if not q or not ph:
        return HttpResponse("")

    # Хариулт бэлтгэх
    if _is_greeting(q):
        bot = "Сайн байна уу! 😊 SISIBOT танд туслахад бэлэн. МУИС-ийн дүрэм, журмын талаар асуугаарай."
    else:
        try:
            res = rag_answer(
                DF,
                q,
                CLIENT,
                top_k=RAG_TOP_K,
                max_ctx_blocks=RAG_MAX_CTX,
                use_faiss=True,
                style=RAG_STYLE,           # long
                max_tokens_out=RAG_MAX_TOKENS,
            )
            answer = prettify_plain(res["answer"])

            # Эх сурвалжийг зөвхөн өндөр холбогдолтой контекстоос сонгоно
            contexts = res.get("contexts", [])
            ref_blocks = _filter_contexts_for_refs(contexts, q, REFS_LIMIT)

            refs = []
            seen = set()
            for c in ref_blocks:
                key = (c.get("source"), c.get("title"))
                if key in seen:
                    continue
                seen.add(key)

                raw_src = c.get("source", "") or ""
                title = c.get("title", "") or ""
                clean_src = _clean_source(raw_src, title)

                refs.append(f"{clean_src} ({title})" if title else clean_src)
                if len(refs) >= REFS_LIMIT:
                    break

            # Хэрэв бодит, утгатай эх сурвалж олдсон үед л хавсаргана
            if refs and answer != "Өгөгдсөн баримтад тодорхой заагаагүй байна.":
                answer = (
                    answer.rstrip()
                    + "\n\nЭх сурвалж:\n• "
                    + "\n• ".join(refs)
                    + "\n\nhttps://www.num.edu.mn/regulations/"
                )

            bot = answer
        except Exception as e:
            bot = f"Алдаа: {e}"

    # History-д хадгалах
    item = {"user": q, "bot": bot, "ts": ts}
    cur["messages"].append(item)
    if cur.get("title", "New chat") == "New chat" and q:
        cur["title"] = (q[:40] + "…") if len(q) > 40 else q
    request.session.modified = True

    # Placeholder-г бодит хариултаар солих (OOB)
    return render(request, "partials/bot_oob.html", {
        "ph": ph,
        "ts": ts,
        "bot": mark_safe(bot),
    })


def new_chat(request: HttpRequest) -> HttpResponse:
    """Шинэ хоосон чат үүсгээд тийш нь шилжинэ."""
    _ensure_state(request)
    cid = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    request.session["chats"][cid] = {
        "title": "New chat",
        "created": int(time.time()),
        "messages": [],
    }
    request.session["current_chat"] = cid
    request.session.modified = True
    return redirect("chat")


def switch_chat(request: HttpRequest, chat_id: str) -> HttpResponse:
    """Идэвхтэй чатыг өөрчлөөд чат нүүр рүү буцаах."""
    _ensure_state(request)
    if chat_id in request.session["chats"]:
        request.session["current_chat"] = chat_id
        request.session.modified = True
    return redirect("chat")


@require_POST
def search_chats(request: HttpRequest) -> HttpResponse:
    _ensure_state(request)
    q = (request.POST.get("q") or "").strip().lower()
    items = _list_chats(request)
    if q:
        filt = []
        for cid, c in items:
            title = c.get("title", "").lower()
            msgs = c.get("messages", [])
            hit = q in title or any(
                q in m.get("user", "").lower() or q in m.get("bot", "").lower()
                for m in msgs
            )
            if hit:
                filt.append((cid, c))
        items = filt
    return render(request, "partials/chat_list.html", {
        "chats": items,
        "current_chat": request.session["current_chat"],
    })


@require_POST
def feedback(request: HttpRequest) -> HttpResponse:
    try:
        payload = {
            "ts": int(time.time()),
            "vote": request.POST.get("vote"),
            "question": request.POST.get("question"),
            "answer": request.POST.get("answer"),
            "ua": request.META.get("HTTP_USER_AGENT", ""),
            "ip": request.META.get("REMOTE_ADDR"),
        }
        os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
        with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return HttpResponse('<span class="text-xs text-emerald-600">Saved ✓</span>')
    except Exception as e:
        return HttpResponse(
            f'<span class="text-xs text-rose-600">Error: {e}</span>',
            status=500,
        )


def clear_history(request: HttpRequest) -> HttpResponse:
    _ensure_state(request)
    cid, cur = _current_chat(request)
    cur["messages"] = []
    request.session.modified = True
    return redirect("chat")


# === Sidebar item + rename/delete ===

def chat_item(request, cid):
    _ensure_state(request)
    c = request.session["chats"].get(cid)
    if not c:
        return HttpResponse("", status=404)
    return render(request, "partials/chat_item.html", {
        "cid": cid,
        "c": c,
        "current_chat": request.session["current_chat"],
    })


@require_http_methods(["GET", "POST"])
def rename_chat(request, cid):
    _ensure_state(request)
    chats = request.session["chats"]
    if cid not in chats:
        return HttpResponse("Not found", status=404)

    if request.method == "POST":
        title = (request.POST.get("title") or "").strip()
        if title:
            chats[cid]["title"] = title
            request.session.modified = True
        html = render_to_string(
            "partials/chat_item.html",
            {"cid": cid, "c": chats[cid], "current_chat": request.session["current_chat"]},
            request=request,
        )
        return HttpResponse(html)

    return render(request, "partials/chat_rename_form.html", {
        "cid": cid,
        "c": chats[cid],
    })


@require_http_methods(["POST"])
def delete_chat(request, cid):
    _ensure_state(request)
    chats = request.session["chats"]
    if cid in chats:
        del chats[cid]
        if request.session["current_chat"] == cid:
            if chats:
                new_cid, _ = _list_chats(request)[0]
                request.session["current_chat"] = new_cid
            else:
                request.session.pop("current_chat", None)
                _ensure_state(request)
        request.session.modified = True
    return redirect("chat")
