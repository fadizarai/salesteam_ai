"""
LAYER 3 — Services
Generates French-language explanations for each product suggestion.

Strategy:
1. PRIMARY: Call HuggingFace Router Inference API (Mistral-7B-Instruct)
   with a structured chat-completion prompt describing the suggestion context.
2. CACHE: Responses are cached for 24h (CACHE_TTL from .env)
   to avoid redundant API calls for identical contexts.
3. FALLBACK: If the API is unavailable or rate-limited,
   generate a rule-based template explanation in French.

Cache key = hash(client_id + code_article + month + category)
Cache storage = in-memory dict (reset on API restart)
"""
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
) -> str:
 """   """
    Build a deterministic cache key for the explanation.

    Args:
        client_id: client code
        code_article: product code
        month: current month number

    Returns:
        MD5 hex digest string
 """   """
    raw_str = f"{client_id}_{code_article}_{month}"
    return hashlib.md5(raw_str.encode("utf-8")).hexdigest()


def _call_huggingface_api(
    prompt: str,
    model: str = "meta-llama/Llama-3.3-70B-Instruct:fastest",
    max_tokens: int = 150,
) -> Optional[str]:
    """
"""    Call the HuggingFace Router Inference API to generate explanation text.

    NOTE: HuggingFace deprecated the old `api-inference.huggingface.co`
    endpoint. The current endpoint is `router.huggingface.co`, and it
    uses the OpenAI-compatible chat completions format (`messages`),
    not the old raw-text `inputs` format.

    Args:
        prompt: French instruction prompt for the model
        model: HuggingFace model ID
        max_tokens: maximum tokens to generate

    Returns:
        Generated text string, or None if API call fails
  """  """
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
) -> str:
"""    """
    Fallback: generate a French explanation using simple rules
    when the HuggingFace API is unavailable.

    Rules applied:
    - is_new_product → new product launch template
    - trend > 0.2    → growing demand template
    - trend < -0.2   → declining (lower quantity) template
    - recency < 30   → frequent buyer template
    - default        → standard reorder template
    """
"""   prob_pct = min(99, round(score_confiance * 100))

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
    cache_ttl: int = 86400,
) -> str:
    """
"""    Generate a French explanation for a single product suggestion.

    Tries HuggingFace API first, falls back to rule-based template.
    """
"""    current_month = datetime.now().month
    cache_key = _get_cache_key(client_id, code_article, current_month)

    # 1. Check cache
    if cache_key in _explanation_cache:
        explanation, expires_at = _explanation_cache[cache_key]
        if datetime.now() < expires_at:
            print("-> Réponse depuis le CACHE")
            return explanation

    # 2. Try LLM
    try:
        # Structured prompt instruction in French
        prompt = (
            "Tu es un assistant IA pour une équipe de vente. Rédige une phrase courte et professionnelle "
            "en français (15-25 mots maximum) justifiant au commercial pourquoi il doit recommander ce produit à son client.\n"
            f"Client: {client_id}\n"
            f"Produit: {designation} ({categorie})\n"
            f"Quantité suggérée: {quantite_suggeree} unités\n"
            f"Score de confiance de l'IA: {int(score_confiance * 100)}%\n"
            f"Historique d'achat du client pour ce produit: {frequency} fois acheté au total, dernier achat il y a {recency_days} jours.\n"
            f"Tendance d'achat récente: {'en hausse' if trend > 0 else 'stable' if abs(trend) <= 0.1 else 'en baisse'}.\n"
            f"Nouveau produit dans la gamme: {'Oui' if is_new_product else 'Non'}.\n"
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
            score_confiance=score_confiance
        )

    # 4. Save to cache
    ttl_seconds = int(os.getenv("CACHE_TTL", str(cache_ttl)))
    expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
    _explanation_cache[cache_key] = (explanation, expires_at)

    return explanation


def clear_cache() -> int:
    """
"""   Clear all cached explanations.
"""  
"""    global _explanation_cache
    num_entries = len(_explanation_cache)
    _explanation_cache.clear()
    return num_entries """
""""""""""""""""""""""""""""""""""""""""""
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

Cache key = hash(client_id + code_article + month + quantite_suggeree + score_confiance)
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
) -> str:
    """
    Build a deterministic cache key for the explanation.

    FIX: the key now includes quantite_suggeree and score_confiance.
    Without them, two different predictions for the same
    (client, product, month) — e.g. 7 units at 09h00 vs 300 units
    at 14h00 after a pipeline re-run — would collide on the same
    cache key, and the STALE explanation text (still describing the
    old quantity) would be served alongside the NEW, different number.
    score_confiance is rounded to 2 decimals to absorb floating-point
    noise without ignoring an actual change in prediction.

    Args:
        client_id: client code
        code_article: product code
        month: current month number
        quantite_suggeree: predicted quantity (regressor output)
        score_confiance: purchase probability (classifier output)

    Returns:
        MD5 hex digest string
    """
    raw_str = (
        f"{client_id}_{code_article}_{month}_"
        f"{quantite_suggeree}_{round(score_confiance, 2)}"
    )
    return hashlib.md5(raw_str.encode("utf-8")).hexdigest()


def _call_huggingface_api(
    prompt: str,
    model: str = "meta-llama/Llama-3.3-70B-Instruct:fastest",
    max_tokens: int = 150,
) -> Optional[str]:
    """
    Call the HuggingFace Router Inference API to generate explanation text.

    Uses router.huggingface.co (chat completions format), since the old
    api-inference.huggingface.co endpoint is deprecated and no longer
    resolves.

    Args:
        prompt: French instruction prompt for the model
        model: HuggingFace model ID (with :provider suffix)
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
) -> str:
    """
    Fallback: generate a French explanation using simple rules
    when the HuggingFace API is unavailable.

    Rules applied:
    - is_new_product → new product launch template
    - trend > 0.2    → growing demand template
    - trend < -0.2   → declining (lower quantity) template
    - recency < 30   → frequent buyer template
    - default        → standard reorder template
    """
    prob_pct = min(99, round(score_confiance * 100))

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
    cache_ttl: int = 86400,
) -> str:
    """
    Generate a French explanation for a single product suggestion.

    Tries HuggingFace API first, falls back to rule-based template.
    """
    current_month = datetime.now().month
    cache_key = _get_cache_key(
        client_id, code_article, current_month,
        quantite_suggeree, score_confiance,
    )

    # 1. Check cache
    if cache_key in _explanation_cache:
        explanation, expires_at = _explanation_cache[cache_key]
        if datetime.now() < expires_at:
            print("-> Réponse depuis le CACHE")
            return explanation

    # 2. Try LLM
    try:
        # Structured prompt instruction in French
        prompt = (
            "Tu es un assistant IA pour une équipe de vente. Rédige une phrase courte et professionnelle "
            "en français (15-25 mots maximum) justifiant au commercial pourquoi il doit recommander ce produit à son client.\n"
            f"Client: {client_id}\n"
            f"Produit: {designation} ({categorie})\n"
            f"Quantité suggérée: {quantite_suggeree} unités\n"
            f"Score de confiance de l'IA: {int(score_confiance * 100)}%\n"
            f"Historique d'achat du client pour ce produit: {frequency} fois acheté au total, dernier achat il y a {recency_days} jours.\n"
            f"Tendance d'achat récente: {'en hausse' if trend > 0 else 'stable' if abs(trend) <= 0.1 else 'en baisse'}.\n"
            f"Nouveau produit dans la gamme: {'Oui' if is_new_product else 'Non'}.\n"
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
            score_confiance=score_confiance
        )

    # 4. Save to cache
    ttl_seconds = int(os.getenv("CACHE_TTL", str(cache_ttl)))
    expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
    _explanation_cache[cache_key] = (explanation, expires_at)

    return explanation


def clear_cache() -> int:
    """
    Clear all cached explanations.
    """
    global _explanation_cache
    num_entries = len(_explanation_cache)
    _explanation_cache.clear()
    return num_entries