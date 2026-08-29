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

# Set to True on first HTTP 402 (quota exhausted) to avoid hammering the API.
# Resets only on process restart or explicit clear_cache() call.
_quota_exhausted: bool = False


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

    Returns None on any error. Sets _quota_exhausted=True on HTTP 402 so that
    subsequent calls skip the HTTP round-trip and go straight to the fallback.
    """
    global _quota_exhausted

    if _quota_exhausted:
        logger.debug("HuggingFace quota exhausted — skipping API call, using fallback.")
        return None

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
        elif response.status_code == 402:
            _quota_exhausted = True
            logger.warning(
                "HuggingFace quota exhausted (HTTP 402). All subsequent LLM calls "
                "will use the rule-based fallback until the server restarts or the "
                "token is recharged. Recharge at https://huggingface.co/settings/billing"
            )
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


def _build_detailed_prompt(context: dict) -> str:
    """
    Builds the exact prompt string sent to the LLM for a detailed,
    4-part explanation. All numbers come verbatim from the context dict
    produced by retrieve_deep_context() — nothing is summarised or omitted.

    Banned words in the model response (enforced by explicit instruction):
        score, probabilité, coefficient, CV, boost, seuil, threshold
    """
    client_id   = context.get("client_id", "inconnu")
    code_art    = context.get("code_article", "inconnu")

    # ── Section 1 : order history ────────────────────────────────────────
    oh = context.get("order_history", {})
    limited = oh.get("limited_history", True)
    total_found = oh.get("total_orders_found", 0)

    if limited or not oh.get("orders"):
        history_block = (
            f"Historique des commandes : donnée non disponible "
            f"(raison : {oh.get('reason', 'fichier source absent')})."
        )
    else:
        orders = oh["orders"]
        lines = "\n".join(
            f"  - {o['date']} : {o['quantite']} unités  (facture {o['code_facture']})"
            for o in orders
        )
        history_block = (
            f"6 dernières commandes réelles sur {total_found} commandes au total :\n"
            + lines
        )

    # ── Section 2 : feature values ───────────────────────────────────────
    fv = context.get("feature_values", {})

    def _fmt(val, fmt=".1f", fallback="donnée non disponible"):
        if val is None:
            return fallback
        try:
            return format(float(val), fmt)
        except (TypeError, ValueError):
            return fallback

    frequency       = fv.get("frequency")
    avg_qty         = _fmt(fv.get("avg_qty"))
    median_qty      = _fmt(fv.get("median_qty"))
    min_qty         = _fmt(fv.get("min_qty"))
    avg_qty         = _fmt(fv.get("avg_qty"), ".0f")
    median_qty      = _fmt(fv.get("median_qty"), ".0f")
    min_qty         = _fmt(fv.get("min_qty"), ".0f")
    max_qty         = _fmt(fv.get("max_qty"), ".0f")
    last_qty        = _fmt(fv.get("last_qty"), ".0f")
    recency_days    = _fmt(fv.get("recency_days"), ".0f")
    avg_delay_days  = _fmt(fv.get("avg_delay_days"), ".0f")
    recency_rel     = _fmt(fv.get("recency_relative"), ".1f")
    trend_raw       = fv.get("trend")
    cv_val          = _fmt(fv.get("cv"), ".2f")
    seasonal_coef   = _fmt(
        fv.get("cat_quarterly_coef") or fv.get("current_month_coef"),
        ".2f",
    )

    # Human-readable trend label (no jargon in the prompt itself)
    if trend_raw is not None:
        if float(trend_raw) > 0.10:
            trend_label = "en hausse"
        elif float(trend_raw) < -0.15:
            trend_label = "en baisse"
        else:
            trend_label = "stable"
    else:
        trend_label = "donnée non disponible"

    features_block = (
        f"Nombre total de commandes passées                : {frequency if frequency else 'donnée non disponible'}\n"
        f"Quantité moyenne par commande                    : {avg_qty} unités\n"
        f"Quantité médiane par commande                    : {median_qty} unités\n"
        f"Quantité minimale jamais commandée               : {min_qty} unités\n"
        f"Quantité maximale jamais commandée               : {max_qty} unités\n"
        f"Dernière quantité commandée                      : {last_qty} unités\n"
        f"Jours depuis la dernière commande                : {recency_days} jours\n"
        f"Rythme habituel de réapprovisionnement           : tous les {avg_delay_days} jours\n"
        f"Ratio délai actuel / rythme habituel             : {recency_rel}\n"
        f"  (> 1.0 = en retard, < 1.0 = pas encore dû)\n"
        f"Tendance des volumes sur les dernières commandes : {trend_label}\n"
        f"Irrégularité des commandes (variabilité/moyenne) : {cv_val}\n"
        f"Coefficient saisonnier du trimestre courant      : {seasonal_coef}"
    )

    # ── Section 3 : score decomposition ─────────────────────────────────
    sd = context.get("score_decomposition", {})
    qty_suggeree  = sd.get("quantite_suggeree", "donnée non disponible")
    qty_min       = sd.get("quantite_min", "donnée non disponible")
    qty_max       = sd.get("quantite_max", "donnée non disponible")
    qty_source    = sd.get("source_quantite", "donnée non disponible")
    urgency_group = sd.get("urgency_group", "inconnu")

    score_block = (
        f"Quantité suggérée   : {qty_suggeree} unités\n"
        f"Fourchette réaliste : de {qty_min} à {qty_max} unités\n"
        f"Source de calcul    : {qty_source}\n"
        f"Groupe d'urgence    : {urgency_group}"
    )

    # ── Section 4 : rank context ─────────────────────────────────────────
    rc = context.get("rank_context", {})
    global_rank  = rc.get("global_rank", "?")
    total_sugg   = rc.get("total_suggestions", "?")
    rank_in_grp  = rc.get("rank_in_group", "?")
    group_size   = rc.get("group_size", "?")
    above        = rc.get("neighbour_above")
    below        = rc.get("neighbour_below")

    above_str = (
        f"{above['designation']}"
        if above else "aucun (ce produit est en tête)"
    )
    below_str = (
        f"{below['designation']}"
        if below else "aucun (ce produit est en dernière position)"
    )

    rank_block = (
        f"Rang global dans la liste complète : {global_rank} sur {total_sugg}\n"
        f"Rang dans son groupe ({urgency_group})  : {rank_in_grp} sur {group_size}\n"
        f"Produit juste au-dessus : {above_str}\n"
        f"Produit juste en-dessous: {below_str}"
    )

    # ── Section 5 : threshold analysis ──────────────────────────────────
    ta = context.get("threshold_analysis", {})
    explication_seuil = ta.get("explanation", "donnée non disponible")

    threshold_block = (
        f"Groupe d'urgence assigné : {urgency_group}\n"
        f"Résumé : {explication_seuil}"
    )

    # ── Assemble final prompt ────────────────────────────────────────────
    prompt = f"""\
Tu es un assistant IA expert en aide à la vente. Tu dois aider un commercial terrain à comprendre \
pourquoi l'IA recommande ce produit à ce client aujourd'hui.

====== DONNÉES VÉRIFIÉES POUR TA RÉPONSE ======

Client        : {client_id}
Produit       : {code_art}

--- HISTORIQUE RÉEL DES COMMANDES ---
{history_block}

--- CARACTÉRISTIQUES DU COMPORTEMENT D'ACHAT ---
{features_block}

--- DÉCOMPOSITION DE LA RECOMMANDATION ---
{score_block}

--- POSITION DANS LE CLASSEMENT ---
{rank_block}

--- ANALYSE DU NIVEAU D'URGENCE ---
{threshold_block}

====== RÈGLES ABSOLUES ======

1. Réponds UNIQUEMENT en français, dans un style professionnel et direct, compréhensible par un commercial \
sans formation technique.

2. Structure ta réponse en EXACTEMENT 4 parties, chacune avec son titre :

## Pourquoi ce produit ?
[Explique en langage naturel pourquoi ce produit est pertinent pour ce client aujourd'hui, \
en t'appuyant sur la fréquence d'achat, le délai depuis la dernière commande par rapport au rythme habituel, \
et la tendance des volumes.]

## Pourquoi cette quantité ?
[Explique la quantité suggérée en citant les commandes réelles de l'historique ci-dessus, \
la source de calcul (IA ou moyenne historique), et la fourchette réaliste min/max.]

## Pourquoi ce classement ?
[Explique pourquoi ce produit se trouve à ce rang, en comparant avec le produit voisin juste en-dessous \
et ce qui différencie leurs positions.]

## Pourquoi ce niveau d'urgence ?
[Explique concrètement pourquoi le produit est classé "{urgency_group}", en traduisant les marges \
en phrases compréhensibles sur le rythme d'achat du client.]

3. MOTS INTERDITS dans ta réponse — remplace-les systématiquement :
   - "score" → "niveau de priorité" ou "importance"
   - "probabilité" → "régularité d'achat" ou "habitude d'achat"
   - "coefficient" → "facteur" ou "ajustement"
   - "CV" ou "variabilité" → "régularité" ou "irrégularité des commandes"
   - "boost" → "ajustement" ou "amplification"
   - "seuil" → "limite" ou "critère"
   - "feature" → supprime le mot ou reformule
   - "valeur brute" → supprime, reformule en mots simples
   - "marge" → reformule en phrase naturelle

4. RÈGLE STRICTE SUR LES CHIFFRES :
   - N'utilise QUE les chiffres entiers et dates listés ci-dessus.
   - Ne cite JAMAIS de nombre à virgule complexe (pas de 0,265 ni de 2,2422 ni de 83,5). Arrondis au nombre entier le plus proche.
   - Pour les tendances, dis simplement "en hausse", "en baisse" ou "stable" sans valeur chiffrée.
   - Pour comparer deux produits, dis "largement prioritaire" ou "légèrement devant" au lieu de citer des niveaux numériques.
   - Si une information manque, écris "donnée non disponible" plutôt que d'estimer ou d'inventer.

Commence directement par "## Pourquoi ce produit ?" sans aucune introduction.
"""
    return prompt


def _rule_based_detailed_explanation(context: dict) -> str:
    """
    Fallback: generates a structured 4-part explanation using rule-based
    logic when the HuggingFace API is unavailable or returns an empty response.
    No jargon, no invented numbers — all values come from context.
    """
    client_id  = context.get("client_id", "ce client")
    code_art   = context.get("code_article", "ce produit")
    fv         = context.get("feature_values", {})
    sd         = context.get("score_decomposition", {})
    rc         = context.get("rank_context", {})
    ta         = context.get("threshold_analysis", {})
    oh         = context.get("order_history", {})

    frequency      = fv.get("frequency", "?")
    avg_qty        = fv.get("avg_qty")
    recency_days   = fv.get("recency_days")
    avg_delay      = fv.get("avg_delay_days")
    recency_rel    = fv.get("recency_relative")
    trend_raw      = fv.get("trend")
    qty_suggeree   = sd.get("quantite_suggeree", "?")
    qty_min        = sd.get("quantite_min", "?")
    qty_max        = sd.get("quantite_max", "?")
    qty_source     = sd.get("source_quantite", "IA")
    urgency_group  = sd.get("urgency_group", "recommande")
    global_rank    = rc.get("global_rank", "?")
    total_sugg     = rc.get("total_suggestions", "?")
    rank_in_grp    = rc.get("rank_in_group", "?")
    group_size     = rc.get("group_size", "?")
    below          = rc.get("neighbour_below")

    # Trend in plain French
    if trend_raw is not None:
        trend_txt = (
            "en hausse" if float(trend_raw) > 0.10
            else "en baisse" if float(trend_raw) < -0.15
            else "stable"
        )
    else:
        trend_txt = "à un niveau stable"

    # Part 1 — why this product
    recency_txt = (
        f"La dernière commande remonte à {int(recency_days)} jours, "
        f"soit {format(float(recency_rel), '.1f')} fois son rythme habituel de {format(float(avg_delay), '.0f')} jours."
        if recency_days and avg_delay and recency_rel else
        "Le délai depuis la dernière commande dépasse son rythme habituel."
    )
    part1 = (
        f"## Pourquoi ce produit ?\n"
        f"Ce client a commandé ce produit {frequency} fois au total, "
        f"ce qui témoigne d'un achat régulier et récurrent. "
        f"{recency_txt} "
        f"La tendance des volumes récents est {trend_txt}."
    )

    # Part 2 — why this quantity
    source_txt = (
        "calculée par le modèle IA à partir de l'ensemble des achats"
        if qty_source == "IA"
        else "basée sur la moyenne historique du client"
    )
    last_orders = oh.get("orders", [])
    history_txt = (
        f" Les dernières commandes enregistrées sont : "
        + ", ".join(f"{o['quantite']} unités le {o['date']}" for o in last_orders[-3:])
        + "."
        if last_orders else ""
    )
    avg_txt = f" Quantité moyenne habituelle : {format(float(avg_qty), '.0f')} unités." if avg_qty else ""
    part2 = (
        f"\n\n## Pourquoi cette quantité ?\n"
        f"La quantité de {qty_suggeree} unités a été {source_txt}.{avg_txt}{history_txt} "
        f"Une fourchette réaliste se situe entre {qty_min} et {qty_max} unités."
    )

    # Part 3 — why this rank
    below_txt = (
        f"Il devance « {below['designation']} » (niveau de priorité : {below['score_final']})."
        if below else "Il n'est devancé par aucun autre produit dans la liste."
    )
    part3 = (
        f"\n\n## Pourquoi ce classement ?\n"
        f"Ce produit occupe la position {rank_in_grp} sur {group_size} dans le groupe « {urgency_group} », "
        f"et la position {global_rank} sur {total_sugg} dans la liste complète. {below_txt}"
    )

    # Part 4 — why this urgency level
    if urgency_group == "urgent":
        urgency_txt = (
            f"Ce produit est classé URGENT car le client n'a pas commandé depuis {int(recency_days) if recency_days else '?'} jours, "
            f"soit bien au-delà de son rythme habituel de réapprovisionnement. "
            f"Son niveau de priorité est également très élevé."
        )
    else:
        urgency_txt = (
            f"Ce produit est classé RECOMMANDÉ car son niveau de priorité dépasse la limite de déclenchement, "
            f"mais le délai depuis la dernière commande ne dépasse pas encore le rythme habituel du client."
        )

    part4 = f"\n\n## Pourquoi ce niveau d'urgence ?\n{urgency_txt}"

    return part1 + part2 + part3 + part4


def explain_suggestion_detailed(context: dict) -> str:
    """
    Generate a detailed, 4-part French explanation for a single product
    recommendation, using all the verifiable evidence assembled by
    retrieve_deep_context().

    Distinct from explain_suggestion() which produces a short card text
    (15-25 words). This function targets ~300-400 words split across four
    clearly labelled sections, suitable for a detail drawer or an audit log.

    Parameters
    ----------
    context : dict
        The dict returned by retrieve_deep_context(). Must contain at least
        the keys: client_id, code_article, order_history, feature_values,
        score_decomposition, rank_context, threshold_analysis.

    Returns
    -------
    str
        A structured French explanation with four Markdown headings:
        ## Pourquoi ce produit ?
        ## Pourquoi cette quantité ?
        ## Pourquoi ce classement ?
        ## Pourquoi ce niveau d'urgence ?

    Notes
    -----
    * Uses _call_huggingface_api() with max_tokens=500, temperature=0.4.
    * Falls back to _rule_based_detailed_explanation() if the API is
      unavailable or returns an unusable response.
    * Not cached — detailed explanations are on-demand and not expected to
      be called on every recommend() invocation.
    """
    prompt = _build_detailed_prompt(context)

    llm_model = os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct:fastest")

    try:
        # Override temperature for the detailed call via a one-off payload
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        if not token or token.strip() == "":
            raise ValueError("No HuggingFace token configured")

        import requests as _requests
        api_url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.4,
            "top_p": 0.9,
        }
        response = _requests.post(api_url, json=payload, headers=headers, timeout=20)
        if response.status_code != 200:
            raise RuntimeError(
                f"HuggingFace API error {response.status_code}: {response.text[:200]}"
            )
        choices = response.json().get("choices", [])
        if not choices:
            raise ValueError("Empty choices in HuggingFace response")

        explanation = choices[0].get("message", {}).get("content", "").strip()
        if not explanation or len(explanation) < 50:
            raise ValueError("LLM response too short to be valid")

        logger.info(
            "explain_suggestion_detailed: LLM response (%d chars) for %s/%s",
            len(explanation),
            context.get("client_id"),
            context.get("code_article"),
        )
        return explanation

    except Exception as exc:
        logger.warning(
            "explain_suggestion_detailed: LLM unavailable (%s). Using rule-based fallback.",
            exc,
        )
        return _rule_based_detailed_explanation(context)


def clear_cache() -> int:
    """Clear all cached explanations and reset the quota-exhausted flag."""
    global _explanation_cache, _quota_exhausted
    num_entries = len(_explanation_cache)
    _explanation_cache.clear()
    _quota_exhausted = False
    logger.info("Explanation cache cleared and quota flag reset.")
    return num_entries