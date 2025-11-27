#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG pipeline (long-form ready):
- JSONL -> DataFrame (text, source, title, embeds)
- Embedding ба хайлт (numpy cosine / FAISS)
- Сонгосон контекстээр МОНГОЛ УРТ, БҮТЭЦТЭЙ хариу үүсгэх
"""

import os, sys, json, argparse, pickle, time
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# .env (OPENAI_API_KEY) ачаална
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from openai import OpenAI

_FAISS_OK = True
try:
    import faiss  
except Exception:
    _FAISS_OK = False

# -------- Файлын IO туслах --------
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# -------- Вектор туслах --------
def norm_mat(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def cosine_topk(q: np.ndarray, M: np.ndarray, k: int):
    """Cosine төстэй байдлын numpy хувилбар (dot product on L2-normalized)."""
    q = q.reshape(1, -1)
    sims = (q @ M.T).ravel()
    idx = np.argpartition(-sims, min(k, len(sims)-1))[:k]
    idx = idx[np.argsort(-sims[idx])]
    return sims[idx], idx

# -------- Embedding үүсгэх --------
def embed_texts(client: OpenAI, texts, model="text-embedding-3-small", batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        r = client.embeddings.create(model=model, input=chunk)
        out.extend([e.embedding for e in r.data])
        time.sleep(0.01)
    return out

# JSON мөрүүдэд боломжит түлхүүрүүд
TEXT_KEYS   = ["text", "content", "chunk", "page_text", "body", "preview"]
TITLE_KEYS  = ["title", "doc_title", "header", "section", "name", "heading_path"]
SOURCE_KEYS = ["source", "source_id", "file", "filepath", "path", "document", "doc_path", "url"]

def _first_non_empty(d: Dict[str, Any], keys):
    """
    Dict дотроос keys дарааллаар эхний хоосон биш утгыг буцаах.
    heading_path (List[str]) тохиолдолд "A > B > C" болгон нийлүүлнэ.
    """
    for k in keys:
        if k not in d:
            continue
        v = d.get(k)
        if k == "heading_path" and isinstance(v, list) and v:
            j = " > ".join([str(x).strip() for x in v if str(x).strip()])
            if j:
                return j
        if isinstance(v, (str, int, float)) and str(v).strip():
            return str(v).strip()
    return ""

def _infer_title_from_source(src: str) -> str:
    """title олдоогүй үед файлын нэрийн үндсээр гарчиг таамаглана."""
    if not src: return ""
    base = os.path.basename(src)
    if "." in base: base = ".".join(base.split(".")[:-1]) or base
    return base

# -------- JSONL -> DataFrame + Embeds --------
def build_df_from_jsonl(jsonl_path: str, client: OpenAI) -> pd.DataFrame:
    rows = load_jsonl(jsonl_path)
    texts, sources, titles = [], [], []
    for r in rows:
        t = _first_non_empty(r, TEXT_KEYS) or _first_non_empty(r.get("meta", {}) if isinstance(r.get("meta"), dict) else {}, TEXT_KEYS)
        if not t: continue
        s = _first_non_empty(r, SOURCE_KEYS) or _first_non_empty(r.get("meta", {}) if isinstance(r.get("meta"), dict) else {}, SOURCE_KEYS)
        ti = _first_non_empty(r, TITLE_KEYS) or _first_non_empty(r.get("meta", {}) if isinstance(r.get("meta"), dict) else {}, TITLE_KEYS)
        if not ti: ti = _infer_title_from_source(s) or "(untitled)"
        if not s:  s  = "(unknown)"
        texts.append(t); sources.append(s); titles.append(ti)

    embeds = embed_texts(client, texts, model="text-embedding-3-small")
    return pd.DataFrame({"text": texts, "source": sources, "title": titles, "embeds": embeds})

# -------- Хайлтын индекс --------
class RagIndex:
    def __init__(self, df: pd.DataFrame, use_faiss: bool = True):
        """
        DF-ийн embeds-ийг хэвийнжүүлээд:
          - FAISS боломжтой бол IndexFlatIP (dot product)
          - Эс бөгөөс numpy cosine_topk
        """
        self.df = df.reset_index(drop=True)
        self.use_faiss = use_faiss and _FAISS_OK
        self.emb = norm_mat(np.array(self.df["embeds"].tolist(), dtype=np.float32))
        if self.use_faiss:
            d = self.emb.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.emb)
        else:
            self.index = None

    def search(self, qvec: np.ndarray, top_k: int):
        """qvec-аар хамгийн ойрын top_k мөрийг буцаана."""
        q = qvec.astype(np.float32); q = q/(np.linalg.norm(q)+1e-12)
        if self.use_faiss:
            D, I = self.index.search(q.reshape(1,-1), top_k)
            return [(int(I[0,i]), float(D[0,i])) for i in range(I.shape[1]) if I[0,i]!=-1]
        sims, idx = cosine_topk(q, self.emb, top_k)
        return [(int(idx[i]), float(sims[i])) for i in range(len(idx))]

# -------- Prompts  --------
def build_system_prompt(style: str = "short") -> str:
    base = (
        "You are a careful RAG assistant for MUIS rules. "
        "Answer in clear Mongolian. Base your answer strictly on the provided context. "
        "If the answer is not in the context, reply exactly: "
        "‘Өгөгдсөн баримтад тодорхой заагаагүй байна.’ "
        "Do NOT use Markdown syntax (#, **, *, headings). "
        "Write plain text. For lists, start lines with '• ' only. "
    )
    if style == "long":
        extra = (
            "Write a well-structured answer with short section labels (plain text only) "
            "and bullet points using '• '. Prefer 2–4 short paragraphs + a compact bullet list. "
        )
        return base + extra
    return base + "Keep it concise (3–5 sentences). "



def _format_ctx(blocks):
    """Контекст блок бүрийг шошготойгоор нэг мөр болгон форматлана."""
    parts = []
    for i, c in enumerate(blocks, 1):
        head = f"[{i}] title: {c['title']} | source: {c['source']}"
        parts.append(head + "\n" + c["text"].strip())
    return "\n\n---\n\n".join(parts)

def build_user_prompt(q: str, blocks, style: str = "short") -> str:
    """Асуулт + контекстийг нэгтгэсэн user prompt."""
    ctx = _format_ctx(blocks)
    if style == "long":
        instructions = (
            "From the context below, produce a detailed and well-structured Mongolian answer. "
            "Use short headings and bullet points where appropriate. "
            "Do not invent facts beyond the context.\n\n"
        )
    else:
        instructions = (
            "From the context below, give a clear and concise Mongolian answer. "
            "Do not add numeric citations.\n\n"
        )
    return (
        f"{instructions}"
        f"Question: {q}\n\nCONTEXT:\n{ctx}\n"
    )

def _generate(client: OpenAI, q: str, blocks, model="gpt-4o-mini", style="short", max_tokens_out=400) -> str:
    """OpenAI Chat Completion-оор хариу гаргах (урт/богино тохиргоотой)."""
    msgs = [
        {"role": "system", "content": build_system_prompt(style=style)},
        {"role": "user", "content": build_user_prompt(q, blocks, style=style)}
    ]
    # temperature бага хэвээр (фактууд алдагдуулахгүйн тулд), харин max_tokens_out-ыг өсгөнө
    r = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=0.2,
        max_tokens=max_tokens_out
    )
    return r.choices[0].message.content.strip()

def rag_answer(
    df: pd.DataFrame,
    q: str,
    client: OpenAI,
    top_k=30,
    max_ctx_blocks=8,
    use_faiss=True,
    style="short",
    max_tokens_out=400
):
    """
    Гол API:
    1) асуултыг embed болгоно
    2) индексээс top_k мөр авна
    3) давхардалгүйгээр max_ctx_blocks контекст сонгоно
    4) генераци дуудаж хариу үүсгэнэ
    """
    qvec = client.embeddings.create(model="text-embedding-3-small", input=[q]).data[0].embedding
    qvec = np.array(qvec, dtype=np.float32)
    index = RagIndex(df, use_faiss=use_faiss)
    hits = index.search(qvec, top_k=top_k)

    blocks, used = [], set()
    for idx, score in hits:
        if idx in used: continue
        row = df.iloc[idx]
        blocks.append({
            "text": row["text"],
            "source": row.get("source","(unknown)"),
            "title": row.get("title","(untitled)"),
            "score": score
        })
        used.add(idx)
        if len(blocks) >= max_ctx_blocks: break

    answer = _generate(
        client,
        q,
        blocks,
        model="gpt-4o-mini",
        style=style,
        max_tokens_out=max_tokens_out
    )
    return {"answer": answer, "contexts": blocks}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", required=True)
    ap.add_argument("-q","--query")
    ap.add_argument("--rebuild-from-jsonl")
    ap.add_argument("--inspect-jsonl", action="store_true")
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--max_ctx_blocks", type=int, default=8)
    ap.add_argument("--no-faiss", action="store_true")
    # урт/богино хэв маяг ба гаралтын токен
    ap.add_argument("--style", choices=["short","long"], default="short")
    ap.add_argument("--max_tokens_out", type=int, default=400)
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY байхгүй байна (.env эсвэл export).", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    if args.inspect_jsonl and args.rebuild_from_jsonl:
        rows = load_jsonl(args.rebuild_from_jsonl)[:3]
        for i, r in enumerate(rows, 1):
            print(f"\n--- Row {i} keys ---"); print(sorted(list(r.keys())))
            print("preview:", (str(r)[:400]+"...") if len(str(r))>400 else r)
        return

    if args.rebuild_from_jsonl:
        print(f"[ingest] JSONL → DF → {args.db_path}")
        df = build_df_from_jsonl(args.rebuild_from_jsonl, client)
        save_pickle(df, args.db_path)
        print(f"OK: rows={len(df)} saved: {args.db_path}")
        if not args.query: return

    if not os.path.exists(args.db_path):
        print(f"DB not found: {args.db_path}", file=sys.stderr); sys.exit(1)
    df = load_pickle(args.db_path)

    if args.query:
        res = rag_answer(
            df,
            args.query,
            client,
            top_k=args.top_k,
            max_ctx_blocks=args.max_ctx_blocks,
            use_faiss=(not args.no_faiss),
            style=args.style,
            max_tokens_out=args.max_tokens_out
        )
        print("\n=== ANSWER ==="); print(res["answer"])
        print("\n=== CONTEXTS ===")
        for i, c in enumerate(res["contexts"], 1):
            print(f"[{i}] {c['title']} | {c['source']} | score={c['score']:.4f}")
    else:
        print(
            f'Жишээ:\n'
            f'python {os.path.basename(__file__)} --db-path data/df_emb_muis.pkl '
            f'-q "Сургалтын төлбөр" --style long --max_tokens_out 700 --top_k 40 --max_ctx_blocks 10'
        )

if __name__ == "__main__":
    main()
