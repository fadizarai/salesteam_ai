"""
LAYER 3 — Services
Generates French-language explanations for each product suggestion.

Strategy:
1. PRIMARY: Call HuggingFace Router Inference API (Llama-3.3-70B-Instruct)
   with a structured chat-completion prompt describing the suggestion context.
2. CACHE: Responses are cached for 24h (CACHE_TTL from .env)
   to avoid redundant API calls for identical contexts.
3. FALLBACK: If the API is unavailable or rate-limited,
   generate a rule-based template explanation in French.

Cache key = hash(client_id + code_article + month)
Cache storage = in-memory dict (reset on API restart)
"""

import os
import logging
import hashlib
import requests
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# In-memory explanation cache: {cache_key: (explanation, expires_at)}
_explanation_cache: dict[str, tuple[str, datetime]] = {}


def _get_cache_key(
    client_id: str,
    code_article: str,
    month: int,
    quantite_suggeree: int,
    score_confiance: float,
    qty_source: str,
) -> str:
    """
    Build a deterministic cache key for the explanation.

    Args:
        client_id: client code
        code_article: product code
        month: current month number
        quantite_suggeree: suggested quantity
        score_confiance: raw ML confidence score
        qty_source: quantity prediction source ("IA", "historique", etc.)

    Returns:
        MD5 hex digest string
    """
    prob_bucket = round(score_confiance, 2)
    raw_str = f"{client_id}_{code_article}_{month}_{quantite_suggeree}_{prob_bucket}_{qty_source}"
    return hashlib.md5(raw_str.encode("utf-8")).hexdigest()


def _call_huggingface_api(
    prompt: str,
    model: str = "meta-llama/Llama-3.3-70B-Instruct:fastest",
    max_tokens: int = 150,
) -> Optional[str]:
    """
    Call the HuggingFace Router Inference API to generate explanation text.

    Args:
        prompt: French instruction prompt for the model
        model: HuggingFace model ID
        max_tokens: maximum tokens to generate

    Returns:
        Generated text string, or None if API call fails
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if not token or token.strip() == "":
        logger.warning("No HuggingFace token found (HF_TOKEN or HUGGINGFACE_API_KEY). Using fallback.")
        return None

    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "top_p": 0.9,
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            result = response.json()
            choices = result.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "").strip()
                return text
            logger.error(f"Unexpected HuggingFace response format: {result}")
        else:
            logger.error(f"HuggingFace API error {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Failed to call HuggingFace API: {e}", exc_info=True)

    return None


def _rule_based_explanation(
    designation: str,
    quantite_suggeree: int,
    recency_days: int,
    frequency: int,
    trend: float,
    is_new_product: bool,
    score_confiance: float,
    qty_source: str = "IA",
) -> str:
    """
    Fallback: generate a French explanation using simple rules
    when the HuggingFace API is unavailable.
    """
    prob_pct = min(99, round(score_confiance * 100))

    if "variance trop élevée" in qty_source.lower():
        return (
            f"Quantité de {quantite_suggeree} unités basée sur la moyenne historique du client en raison d'achats très irréguliers. "
            f"Confiance IA : {prob_pct}%."
        )

    if is_new_product:
        return (
            f"Nouveau produit suggéré pour tester la gamme. "
            f"Confiance IA : {prob_pct}%. Quantité recommandée : {quantite_suggeree} unités."
        )

    if frequency >= 5 and recency_days <= 30:
        return (
            f"Client fidèle sur cet article (acheté {frequency} fois, dernier achat il y a {recency_days} jours). "
            f"Réapprovisionnement suggéré de {quantite_suggeree} unités (Confiance {prob_pct}%)."
        )

    if trend > 0.2:
        return (
            f"Tendance à la hausse détectée pour cet article. "
            f"Volume de {quantite_suggeree} unités suggéré pour répondre à la demande (Confiance {prob_pct}%)."
        )

    if trend < -0.2:
        return (
            f"Baisse légère de la demande historique. "
            f"Quantité prudente recommandée de {quantite_suggeree} unités (Confiance {prob_pct}%)."
        )

    return (
        f"Réassort classique de {quantite_suggeree} unités basé sur l'historique et la saisonnalité. "
        f"Confiance IA : {prob_pct}%."
    )


def explain_suggestion(
    client_id: str,
    code_article: str,
    designation: str,
    categorie: str,
    quantite_suggeree: int,
    score_confiance: float,
    recency_days: int,
    frequency: int,
    trend: float,
    is_new_product: bool,
    qty_source: str = "IA",
    cache_ttl: int = 86400,
) -> str:
    """
    Generate a French explanation for a single product suggestion.

    Tries HuggingFace API first, falls back to rule-based template.
    """
    current_month = datetime.now().month
    cache_key = _get_cache_key(
        client_id=client_id,
        code_article=code_article,
        month=current_month,
        quantite_suggeree=quantite_suggeree,
        score_confiance=score_confiance,
        qty_source=qty_source,
    )

    # 1. Check cache
    if cache_key in _explanation_cache:
        explanation, expires_at = _explanation_cache[cache_key]
        if datetime.now() < expires_at:
            print("-> Réponse depuis le CACHE")
            return explanation

    # 2. Try LLM
    try:
        # Instruction if quantity comes from CV fallback
        variance_instruction = ""
        if "variance trop élevée" in qty_source.lower():
            variance_instruction = (
                "Instruction spéciale: Cette quantité est basée sur la moyenne historique du client car son comportement d'achat est "
                "trop irrégulier pour une prédiction précise — formule l'explication en conséquence, sans mentionner de tendance ou de prédiction IA.\n"
            )

        # Structured prompt instruction in French
        prompt = (
            "Tu es un assistant IA pour une équipe de vente. Rédige une phrase courte et professionnelle "
            "en français (15-25 mots maximum) justifiant au commercial pourquoi il doit recommander ce produit à son client.\n"
            f"Client: {client_id}\n"
            f"Produit: {designation} ({categorie})\n"
            f"Quantité suggérée: {quantite_suggeree} unités\n"
            f"Source de la quantité: {qty_source}\n"
            f"Score de confiance de l'IA: {int(score_confiance * 100)}%\n"
            f"Historique d'achat du client pour ce produit: {frequency} fois acheté au total, dernier achat il y a {recency_days} jours.\n"
            f"Tendance d'achat récente: {'en hausse' if trend > 0 else 'stable' if abs(trend) <= 0.1 else 'en baisse'}.\n"
            f"Nouveau produit dans la gamme: {'Oui' if is_new_product else 'Non'}.\n"
            f"{variance_instruction}"
            "Règle stricte: Retourne UNIQUEMENT l'explication, sans introduction, sans salutations, ni guillemets."
        )

        llm_model = os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct:fastest") 
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "150"))

        explanation = _call_huggingface_api(prompt, model=llm_model, max_tokens=max_tokens)

        if not explanation or len(explanation.strip()) < 10:
            raise ValueError("Réponse LLM vide ou invalide")

        print("-> Réponse depuis LLM (réel)")
    except Exception as e:
        print(f"-> FALLBACK activé — raison : {e}")
        explanation = _rule_based_explanation(
            designation=designation,
            quantite_suggeree=quantite_suggeree,
            recency_days=recency_days,
            frequency=frequency,
            trend=trend,
            is_new_product=is_new_product,
            score_confiance=score_confiance,
            qty_source=qty_source,
        )

    # 4. Save to cache
    ttl_seconds = int(os.getenv("CACHE_TTL", str(cache_ttl)))
    expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
    _explanation_cache[cache_key] = (explanation, expires_at)

    return explanation


def clear_cache() -> int:
    """Clear all cached explanations."""
    global _explanation_cache
    num_entries = len(_explanation_cache)
    _explanation_cache.clear()
    return num_entries