"""LLM Trade Explainer — Phase 8c.

After each paper trade executes, generates a 2-sentence natural language
explanation of WHY the bot traded, stored in the ``trade_explanations``
table and surfaced in the dashboard Execution Log tab and the MCP server's
``hogan_get_recent_trades`` tool.

LLM provider priority:
1. Local Ollama (``http://localhost:11434``) — zero cost, fully private.
2. OpenAI API (``OPENAI_API_KEY``) — high quality, small cost per trade.
3. Template fallback — deterministic rule-based explanation, no LLM needed.

Usage::

    from hogan_bot.trade_explainer import explain_trade

    explanation = explain_trade(
        fill_id="abc123",
        symbol="BTC/USD",
        action="buy",
        price=65000.0,
        agent_signal=pipeline_result,
        conn=conn,
    )
    # Returns a 2-sentence string stored in trade_explanations table

Configure::

    HOGAN_LLM_PROVIDER=ollama   (default)
    HOGAN_OLLAMA_MODEL=llama3   (default)
    HOGAN_OLLAMA_URL=http://localhost:11434
    OPENAI_API_KEY=sk-...       (used if provider=openai)
    OPENAI_MODEL=gpt-4o-mini    (default)
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_MODEL = "llama3"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_MAX_PROMPT_CHARS = 800


def _build_prompt(
    symbol: str,
    action: str,
    price: float,
    signal_details: dict,
) -> str:
    """Build a concise prompt for the LLM."""
    tech = signal_details.get("tech", {})
    sent = signal_details.get("sentiment", {})
    macro = signal_details.get("macro", {})
    confidence = signal_details.get("confidence", 0.0)
    existing_explanation = signal_details.get("explanation", "")

    context = (
        f"Symbol: {symbol}\n"
        f"Action: {action.upper()} at ${price:,.2f}\n"
        f"Confidence: {confidence:.0%}\n"
    )
    if tech:
        context += f"Technical: {tech.get('action','hold')} (conf {tech.get('confidence',0):.2f})\n"
    if sent:
        context += f"Sentiment: {sent.get('bias','neutral')} (strength {sent.get('strength',0):.2f})\n"
    if macro:
        context += f"Macro regime: {macro.get('regime','neutral')}\n"
    if existing_explanation:
        context += f"Agent summary: {existing_explanation}\n"

    prompt = (
        f"You are a crypto trading analyst. Explain in exactly 2 short sentences why "
        f"the trading bot made this trade decision:\n\n{context}\n"
        f"Be concise, specific, and factual. Do not exceed 2 sentences."
    )
    return prompt[:_MAX_PROMPT_CHARS]


def _call_ollama(prompt: str) -> str | None:
    """Call local Ollama API."""
    import urllib.error
    import urllib.request

    url = os.getenv("HOGAN_OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("HOGAN_OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 120, "temperature": 0.3},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except Exception as exc:
        logger.debug("Ollama call failed: %s", exc)
        return None


def _call_openai(prompt: str) -> str | None:
    """Call OpenAI API."""
    import urllib.error
    import urllib.request

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None

    model = os.getenv("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL)
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise crypto trading analyst."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 100,
        "temperature": 0.3,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.debug("OpenAI call failed: %s", exc)
        return None


def _template_explanation(
    symbol: str,
    action: str,
    price: float,
    signal_details: dict,
) -> str:
    """Deterministic rule-based explanation (no LLM required)."""
    tech = signal_details.get("tech", {}) or {}
    sent = signal_details.get("sentiment", {}) or {}
    macro = signal_details.get("macro", {}) or {}
    conf = signal_details.get("confidence", 0.0)

    tech_action = tech.get("action", "hold")
    sent_bias = sent.get("bias", "neutral")
    macro_regime = macro.get("regime", "neutral")
    conf_pct = int(conf * 100)

    if action in ("buy", "sell"):
        sent_part = (
            f" Sentiment is {sent_bias}" if sent_bias != "neutral"
            else " Sentiment is neutral"
        )
        macro_part = (
            f" and the macro regime is {macro_regime}."
            if macro_regime != "neutral"
            else "."
        )
        return (
            f"Technical analysis on {symbol} generated a {tech_action.upper()} signal with "
            f"{conf_pct}% confidence.{sent_part}{macro_part} "
            f"The trade was executed at ${price:,.2f} as all risk checks passed."
        )
    return (
        f"No trade was taken for {symbol} at ${price:,.2f}; "
        f"the signal was HOLD with {conf_pct}% confidence."
    )


def generate_explanation(
    symbol: str,
    action: str,
    price: float,
    signal_details: dict,
) -> tuple[str, str]:
    """Generate a trade explanation using the configured LLM provider.

    Returns ``(explanation_text, model_used)``.
    """
    provider = os.getenv("HOGAN_LLM_PROVIDER", "ollama").lower()
    prompt = _build_prompt(symbol, action, price, signal_details)

    if provider == "ollama":
        text = _call_ollama(prompt)
        if text:
            return text, f"ollama:{os.getenv('HOGAN_OLLAMA_MODEL', _DEFAULT_OLLAMA_MODEL)}"

    if provider == "openai" or (provider == "ollama" and not text):
        text = _call_openai(prompt)
        if text:
            return text, f"openai:{os.getenv('OPENAI_MODEL', _DEFAULT_OPENAI_MODEL)}"

    # Template fallback
    text = _template_explanation(symbol, action, price, signal_details)
    return text, "template"


def explain_trade(
    fill_id: str,
    symbol: str,
    action: str,
    price: float,
    agent_signal: Any | None = None,
    conn=None,
) -> str:
    """Generate and store a trade explanation.

    Parameters
    ----------
    fill_id:    The fill ID from the fills table.
    symbol:     Trading pair.
    action:     "buy" or "sell".
    price:      Execution price.
    agent_signal: AgentSignal object (from agent_pipeline.py) or dict, optional.
    conn:       Open SQLite connection. If provided, stores the explanation.

    Returns the explanation text.
    """
    signal_details: dict = {}
    if agent_signal is not None:
        if hasattr(agent_signal, "__dict__"):
            signal_details = {
                "action": getattr(agent_signal, "action", action),
                "confidence": getattr(agent_signal, "confidence", 0.0),
                "explanation": getattr(agent_signal, "explanation", ""),
                "tech": {
                    "action": getattr(agent_signal.tech, "action", "hold"),
                    "confidence": getattr(agent_signal.tech, "confidence", 0.0),
                } if agent_signal.tech else {},
                "sentiment": {
                    "bias": getattr(agent_signal.sentiment, "bias", "neutral"),
                    "strength": getattr(agent_signal.sentiment, "strength", 0.0),
                } if agent_signal.sentiment else {},
                "macro": {
                    "regime": getattr(agent_signal.macro, "regime", "neutral"),
                } if agent_signal.macro else {},
            }
        elif isinstance(agent_signal, dict):
            signal_details = agent_signal

    explanation, model_used = generate_explanation(symbol, action, price, signal_details)

    if conn is not None:
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO trade_explanations
                    (fill_id, symbol, ts_ms, explanation, model_used)
                VALUES (?, ?, ?, ?, ?)
                """,
                (fill_id, symbol, int(time.time() * 1000), explanation, model_used),
            )
            conn.commit()
        except Exception as exc:
            logger.warning("Failed to store trade explanation: %s", exc)

    logger.info(
        "Trade explanation [%s] via %s: %s...",
        fill_id, model_used, explanation[:80],
    )
    return explanation
