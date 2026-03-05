"""RAG Knowledge Base — Phase 8b (AnythingLLM-inspired).

Provides a retrieval-augmented reasoning layer for the MetaWeigher.  Stores
embeddings of:
  - Recent news headlines (from CryptoPanic / fetch_news_sentiment.py)
  - Past trade outcomes with market context
  - Notable market events (annotated manually or via LLM)

At signal-generation time, ``retrieve_relevant_context()`` finds the K most
similar past market states (by cosine similarity on feature embeddings) and
returns their outcomes — enabling the MetaWeigher to reason: "in the 5 most
similar past conditions, the bot made money X% of the time."

Implementation uses ChromaDB (local, no server) + sentence-transformers.
Falls back to a lightweight TF-IDF retriever if heavy dependencies are absent.

Usage::

    from hogan_bot.rag_knowledge import RAGKnowledgeBase

    rag = RAGKnowledgeBase()
    rag.index_trade_outcome(fill_id="abc", features=[...], pnl_pct=2.5,
                             context_text="BTC pumped after FOMC hold")
    rag.index_news("BTC breaks ATH as ETF inflows surge", ts_ms=...)

    context = rag.retrieve_relevant_context(current_features, k=5)
    # context["similar_win_rate"]    — float [0,1]
    # context["similar_trades"]      — list of past trade dicts
    # context["relevant_headlines"]  — list of similar news strings
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_PERSIST_DIR = "data/rag_db"
_TRADE_COLLECTION = "trade_outcomes"
_NEWS_COLLECTION = "news_headlines"

# --------------------------------------------------------------------------
# Embedding helpers
# --------------------------------------------------------------------------

def _embed_text(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Encode *texts* to embedding vectors using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(model_name)
        return np.array(model.encode(texts, convert_to_numpy=True))
    except ImportError:
        # Fallback: very simple TF-IDF-like bag-of-chars embedding
        return _tfidf_embed(texts)


def _tfidf_embed(texts: list[str], dim: int = 128) -> np.ndarray:
    """Minimal fallback embedding — char n-gram hashing."""
    out = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        chars = text.lower()
        for i in range(len(chars) - 1):
            bigram = chars[i:i+2]
            idx = hash(bigram) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        out.append(vec / (norm + 1e-9))
    return np.array(out)


def _embed_features(features: list[float]) -> np.ndarray:
    """Convert a numeric feature vector to an embedding (normalized)."""
    v = np.array(features, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / (norm + 1e-9)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# --------------------------------------------------------------------------
# RAG Knowledge Base
# --------------------------------------------------------------------------

class RAGKnowledgeBase:
    """Local vector store for trade outcomes and news headlines.

    Automatically uses ChromaDB when available, falling back to a simple
    in-memory list with cosine similarity search.
    """

    def __init__(
        self,
        persist_dir: str = _DEFAULT_PERSIST_DIR,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self._chroma_client = None
        self._trade_collection = None
        self._news_collection = None
        self._fallback_trades: list[dict] = []
        self._fallback_news: list[dict] = []
        self._init_store()

    def _init_store(self) -> None:
        try:
            import chromadb  # type: ignore
            self._chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._trade_collection = self._chroma_client.get_or_create_collection(
                _TRADE_COLLECTION,
                metadata={"description": "Trade outcomes with market context"},
            )
            self._news_collection = self._chroma_client.get_or_create_collection(
                _NEWS_COLLECTION,
                metadata={"description": "News headlines for retrieval"},
            )
            logger.info("RAG: ChromaDB initialized at %s", self.persist_dir)
        except ImportError:
            logger.info("RAG: ChromaDB not installed — using in-memory fallback.")
            self._load_fallback()

    def _load_fallback(self) -> None:
        """Load fallback from JSON files."""
        trades_path = self.persist_dir / "trades.jsonl"
        news_path = self.persist_dir / "news.jsonl"
        if trades_path.exists():
            with open(trades_path) as f:
                self._fallback_trades = [json.loads(l) for l in f if l.strip()]
        if news_path.exists():
            with open(news_path) as f:
                self._fallback_news = [json.loads(l) for l in f if l.strip()]

    def _save_fallback_trade(self, record: dict) -> None:
        path = self.persist_dir / "trades.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _save_fallback_news(self, record: dict) -> None:
        path = self.persist_dir / "news.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_trade_outcome(
        self,
        fill_id: str,
        features: list[float],
        pnl_pct: float,
        symbol: str = "BTC/USD",
        context_text: str = "",
        ts_ms: int | None = None,
    ) -> None:
        """Embed and store a completed trade outcome."""
        ts = ts_ms or int(time.time() * 1000)
        label = 1 if pnl_pct > 0 else 0
        text = context_text or f"{symbol} trade pnl={pnl_pct:.2f}%"
        embedding = _embed_features(features).tolist()

        metadata = {
            "fill_id": fill_id,
            "symbol": symbol,
            "pnl_pct": round(pnl_pct, 4),
            "label": label,
            "ts_ms": ts,
            "text": text[:500],
        }

        if self._trade_collection is not None:
            try:
                self._trade_collection.add(
                    ids=[fill_id],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[metadata],
                )
                return
            except Exception as exc:
                logger.warning("ChromaDB trade index failed: %s", exc)

        # Fallback
        record = {"id": fill_id, "embedding": embedding, **metadata}
        self._fallback_trades.append(record)
        self._save_fallback_trade(record)

    def index_news(
        self,
        headline: str,
        ts_ms: int | None = None,
        symbol: str = "BTC/USD",
        sentiment_score: float = 0.0,
    ) -> None:
        """Embed and store a news headline."""
        ts = ts_ms or int(time.time() * 1000)
        doc_id = f"news_{ts}_{hash(headline) % 1_000_000}"
        embedding = _embed_text([headline], self.embedding_model)[0].tolist()
        metadata = {
            "symbol": symbol,
            "ts_ms": ts,
            "sentiment_score": round(sentiment_score, 4),
            "headline": headline[:500],
        }

        if self._news_collection is not None:
            try:
                self._news_collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[headline],
                    metadatas=[metadata],
                )
                return
            except Exception as exc:
                logger.warning("ChromaDB news index failed: %s", exc)

        record = {"id": doc_id, "embedding": embedding, **metadata}
        self._fallback_news.append(record)
        self._save_fallback_news(record)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_relevant_context(
        self,
        current_features: list[float],
        k: int = 5,
        current_headline: str | None = None,
    ) -> dict[str, Any]:
        """Find K most similar past market states and return their outcomes.

        Returns:
            {
              "similar_win_rate": float [0,1],
              "similar_trades": list[dict],
              "relevant_headlines": list[str],
              "k_found": int,
            }
        """
        query_emb = _embed_features(current_features)
        similar_trades = self._query_trades(query_emb, k)
        headlines = self._query_news(
            current_headline or "", k=3
        )

        if not similar_trades:
            return {
                "similar_win_rate": 0.5,
                "similar_trades": [],
                "relevant_headlines": headlines,
                "k_found": 0,
            }

        wins = sum(1 for t in similar_trades if t.get("label", 0) == 1)
        win_rate = wins / len(similar_trades)

        return {
            "similar_win_rate": round(win_rate, 4),
            "similar_trades": similar_trades[:k],
            "relevant_headlines": headlines,
            "k_found": len(similar_trades),
        }

    def _query_trades(self, query_emb: np.ndarray, k: int) -> list[dict]:
        if self._trade_collection is not None:
            try:
                count = self._trade_collection.count()
                if count == 0:
                    return []
                results = self._trade_collection.query(
                    query_embeddings=[query_emb.tolist()],
                    n_results=min(k, count),
                    include=["metadatas", "distances"],
                )
                return results.get("metadatas", [[]])[0]
            except Exception as exc:
                logger.debug("ChromaDB trade query failed: %s", exc)

        # Fallback cosine search
        if not self._fallback_trades:
            return []
        sims = []
        for rec in self._fallback_trades:
            emb = np.array(rec.get("embedding", []), dtype=np.float32)
            if len(emb) != len(query_emb):
                continue
            sim = _cosine_similarity(query_emb, emb)
            sims.append((sim, rec))
        sims.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in sims[:k]]

    def _query_news(self, query_text: str, k: int = 3) -> list[str]:
        if not query_text:
            # Return most recent news
            if self._news_collection is not None:
                try:
                    count = self._news_collection.count()
                    if count == 0:
                        return []
                    results = self._news_collection.peek(limit=min(k, count))
                    return results.get("documents", [])
                except Exception:
                    pass
            return [r.get("headline", "") for r in self._fallback_news[-k:]]

        query_emb = _embed_text([query_text], self.embedding_model)[0]

        if self._news_collection is not None:
            try:
                count = self._news_collection.count()
                if count == 0:
                    return []
                results = self._news_collection.query(
                    query_embeddings=[query_emb.tolist()],
                    n_results=min(k, count),
                    include=["documents"],
                )
                return results.get("documents", [[]])[0]
            except Exception as exc:
                logger.debug("ChromaDB news query failed: %s", exc)

        # Fallback
        if not self._fallback_news:
            return []
        sims = [(
            _cosine_similarity(query_emb, np.array(r.get("embedding", []))),
            r.get("headline", "")
        ) for r in self._fallback_news]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in sims[:k] if h]

    # ------------------------------------------------------------------
    # Batch import from DB
    # ------------------------------------------------------------------

    def import_from_db(self, db_path: str = "data/hogan.db", limit: int = 1000) -> int:
        """Import labeled trades and news from the SQLite DB into the vector store.

        Returns the number of items indexed.
        """
        from hogan_bot.storage import get_connection
        conn = get_connection(db_path)
        total = 0

        # Trades from online_training_buffer
        cur = conn.execute(
            """
            SELECT b.row_id, b.symbol, b.ts_ms, b.features_json,
                   b.pnl_pct, b.label, f.fill_id
            FROM online_training_buffer b
            LEFT JOIN fills f ON f.symbol=b.symbol AND ABS(f.ts_ms - b.fill_ts_ms) < 5000
            WHERE b.label IS NOT NULL
            ORDER BY b.ts_ms DESC LIMIT ?
            """,
            (limit,),
        )
        for row in cur.fetchall():
            row_id, symbol, ts_ms, feat_json, pnl_pct, label, fill_id = row
            try:
                features = json.loads(feat_json)
                fid = fill_id or f"row_{row_id}"
                self.index_trade_outcome(
                    fill_id=fid,
                    features=features,
                    pnl_pct=float(pnl_pct or 0),
                    symbol=symbol,
                    ts_ms=ts_ms,
                )
                total += 1
            except Exception:
                pass

        # News from onchain_metrics (news_sentiment_score as proxy)
        cur = conn.execute(
            """
            SELECT symbol, date, value FROM onchain_metrics
            WHERE metric='news_sentiment_score'
            ORDER BY date DESC LIMIT ?
            """,
            (limit,),
        )
        for symbol, date_str, sentiment in cur.fetchall():
            headline = f"{symbol} news sentiment={sentiment:.3f} on {date_str}"
            self.index_news(headline=headline, sentiment_score=float(sentiment), symbol=symbol)
            total += 1

        conn.close()
        logger.info("RAG import_from_db: indexed %d items from %s", total, db_path)
        return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="RAG Knowledge Base CLI")
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--persist-dir", default=_DEFAULT_PERSIST_DIR)
    p.add_argument("--import-db", action="store_true",
                   help="Import labeled trades + news from SQLite into vector store")
    p.add_argument("--limit", type=int, default=1000)
    args = p.parse_args()

    rag = RAGKnowledgeBase(persist_dir=args.persist_dir)
    if args.import_db:
        n = rag.import_from_db(db_path=args.db, limit=args.limit)
        print(f"Indexed {n} items.")
    else:
        p.print_help()


if __name__ == "__main__":
    _main()
