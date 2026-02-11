import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import urlparse

import dotenv
import httpx

from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    ConditionalPrediction,
    ConditionalQuestion,
    DatePercentile,
    DateQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    PredictionAffirmed,
    PredictionTypes,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# Precompiled whitespace normalizer (avoids backslashes inside f-string expressions)
_WS_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------
# IMPORTANT: OpenRouter + litellm compatibility
# - Some litellm OpenRouter adapters strip the first "openrouter/".
# - If you set model="openrouter/free", the request may become model="free".
# - To be robust, default to "openrouter/openrouter/free" so stripping still leaves "openrouter/free".
# ---------------------------------------------------------------------
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "openrouter/openrouter/free")
OPENROUTER_PARSER_MODEL = os.getenv("OPENROUTER_PARSER_MODEL", OPENROUTER_DEFAULT_MODEL)

LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")

LINKUP_ENDPOINT = os.getenv("LINKUP_ENDPOINT", "https://api.linkup.so/v1/search")
EXA_ENDPOINT = os.getenv("EXA_ENDPOINT", "https://api.exa.ai/search")

HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "25"))

# Added tournament slug
MARKET_PULSE_TOURNAMENT_SLUG = "market-pulse-26q1"

# Keys observed in Metaculus/forecasting_tools payloads that can carry numeric bounds
_BOUND_KEYS = {
    "upper_bound",
    "lower_bound",
    "nominal_upper_bound",
    "nominal_lower_bound",
    "upperBound",
    "lowerBound",
    "nominalUpperBound",
    "nominalLowerBound",
}


def _coerce_int_bounds_to_float(obj: Any) -> Any:
    """
    Recursively walk dict/list payloads and convert int bounds into float bounds.
    This prevents forecasting_tools assertions like: assert isinstance(upper_bound, float)
    when the API returns e.g. 200 (int) instead of 200.0 (float).
    """
    if isinstance(obj, dict):
        out: Dict[Any, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and k in _BOUND_KEYS and isinstance(v, int):
                out[k] = float(v)
            else:
                out[k] = _coerce_int_bounds_to_float(v)
        return out
    if isinstance(obj, list):
        return [_coerce_int_bounds_to_float(x) for x in obj]
    return obj


class PatchedMetaculusClient(MetaculusClient):
    """
    Patch point: forecasting_tools parses Metaculus post JSON and may assert bounds are floats.
    Some posts return ints for bounds. We coerce, then defer to upstream logic (important for groups/lists).
    """

    def _post_json_to_questions_while_handling_groups(self, post_json_from_api, group_question_mode=None):
        post_json_from_api = _coerce_int_bounds_to_float(post_json_from_api)
        return super()._post_json_to_questions_while_handling_groups(
            post_json_from_api,
            group_question_mode=group_question_mode,
        )


@dataclass
class EvidenceSummary:
    direction: Literal["yes", "no", "unclear"]
    strength: float  # 0..1
    key_points: List[str]
    sources: List[str]


@dataclass
class EvidenceSummaryOut:
    # used for structure_output parsing
    direction: Literal["yes", "no", "unclear"]
    strength: float
    key_points: List[str]
    sources: List[str]


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.5
    return max(0.0, min(1.0, x))


def _to_prob_decimal(x: float) -> float:
    return max(0.01, min(0.99, x))


def _softmax_normalize(pairs: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in pairs.values())
    if total <= 0:
        n = max(1, len(pairs))
        return {k: 1.0 / n for k in pairs}
    return {k: max(0.0, v) / total for k, v in pairs.items()}


def _median_merge_lists(values: List[float]) -> float:
    values = [v for v in values if isinstance(v, (int, float)) and v == v]
    if not values:
        return 0.5
    return float(median(values))


async def _gather_tournament_questions(
    client: PatchedMetaculusClient,
    tournament_slug_or_id: str,
) -> List[MetaculusQuestion]:
    """
    Critical: Ensure we use the PATCHED client for tournament retrieval, so integer bounds are coerced.
    Depending on forecasting_tools version, the method name might differ. We try common options.
    """
    # Try a few known method names used across versions
    for name in ("get_questions_from_tournament", "get_questions_in_tournament", "get_tournament_questions"):
        fn = getattr(client, name, None)
        if callable(fn):
            out = fn(tournament_slug_or_id)
            # Some implementations may be async; handle both
            if asyncio.iscoroutine(out):
                out = await out
            return list(out)
    raise AttributeError(
        "PatchedMetaculusClient does not have a tournament-question retrieval method "
        "(expected one of: get_questions_from_tournament / get_questions_in_tournament / get_tournament_questions)."
    )


# -----------------------------
# Source quality ranking
# -----------------------------
_HIGH_TRUST_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "ft.com",
    "wsj.com",
    "bloomberg.com",
    "economist.com",
    "bbc.co.uk",
    "bbc.com",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "sec.gov",
    "federalregister.gov",
    "europa.eu",
    "ec.europa.eu",
    "gov.uk",
    "who.int",
    "un.org",
    "worldbank.org",
    "imf.org",
    "oecd.org",
    "arxiv.org",
    "nature.com",
    "science.org",
    "ieee.org",
    "acm.org",
}

_MED_TRUST_HINTS = (
    "investor",
    "ir.",
    "investors.",
    "press",
    "newsroom",
    "docs.",
    "github.com",
)

_LOW_TRUST_HINTS = (
    "pinterest.",
    "quora.",
    "medium.com",
    "substack.com",
    "blogspot.",
    "wordpress.",
    "tumblr.",
    "tiktok.",
    "facebook.",
    "x.com",
    "twitter.com",
)


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        host = host[4:] if host.startswith("www.") else host
        return host
    except Exception:
        return ""


def _score_source(url: str, title: str = "", snippet: str = "") -> float:
    d = _domain_of(url)
    if not d:
        return 0.0

    score = 0.0
    if d in _HIGH_TRUST_DOMAINS:
        score += 2.5
    if d.endswith(".gov") or d.endswith(".edu") or d.endswith(".org"):
        score += 1.7
    if "github.com" in d:
        score += 1.0

    low = (title + " " + snippet).lower()
    if any(h in d for h in _MED_TRUST_HINTS) or any(h in low for h in _MED_TRUST_HINTS):
        score += 0.6

    if any(h in d for h in _LOW_TRUST_HINTS):
        score -= 1.0
    if len(snippet.strip()) < 120:
        score -= 0.2

    return score


def _rank_and_format_sources(items: List[Dict], max_to_keep: int = 14) -> Tuple[str, List[str]]:
    scored: List[Tuple[float, str, str, str]] = []
    for it in items:
        url = (it.get("url") or "").strip()
        if not url:
            continue

        title = (it.get("title") or it.get("name") or "").strip()

        text = ""
        if isinstance(it.get("highlights"), list) and it["highlights"]:
            text = str(it["highlights"][0])
        else:
            text = (it.get("content") or it.get("text") or "").strip()

        snippet = _WS_RE.sub(" ", text)[:420]
        score = _score_source(url, title=title, snippet=snippet)
        scored.append((score, url, title, snippet))

    best_by_url: Dict[str, Tuple[float, str, str]] = {}
    for score, url, title, snippet in scored:
        cur = best_by_url.get(url)
        if cur is None or score > cur[0]:
            best_by_url[url] = (score, title, snippet)

    ranked = sorted(((s, u, t, sn) for u, (s, t, sn) in best_by_url.items()), reverse=True)

    bullets: List[str] = []
    urls: List[str] = []
    for score, url, title, snippet in ranked[:max_to_keep]:
        label = title if title else url
        bullets.append(f"- [{score:+.2f}] {label}: {snippet} ({url})")
        urls.append(url)

    return ("\n".join(bullets) if bullets else "(no sources retrieved)"), urls


class LinkupExaSpringBot2026(ForecastBot):
    BOT_NAME = "nike"

    _max_concurrent_questions = int(os.getenv("MAX_CONCURRENT_QUESTIONS", "1"))
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = int(os.getenv("STRUCTURE_VALIDATION_SAMPLES", "2"))
    _fermi_samples = int(os.getenv("FERMI_SAMPLES", "5"))  # odd -> median
    _evidence_extremize_threshold = float(os.getenv("EVIDENCE_EXTREMIZE_THRESHOLD", "0.60"))
    _extremize_target = float(os.getenv("EXTREMIZE_TARGET", "0.90"))

    def __init__(self, *args, **kwargs):
        kwargs["llms"] = {
            "default": GeneralLlm(
                model=OPENROUTER_DEFAULT_MODEL,
                temperature=0.2,
                timeout=60,
                allowed_tries=2,
            ),
            "parser": GeneralLlm(
                model=OPENROUTER_PARSER_MODEL,
                temperature=0.0,
                timeout=60,
                allowed_tries=2,
            ),
            "summarizer": GeneralLlm(
                model=OPENROUTER_DEFAULT_MODEL,
                temperature=0.2,
                timeout=60,
                allowed_tries=2,
            ),
            "researcher": GeneralLlm(
                model=OPENROUTER_DEFAULT_MODEL,
                temperature=0.2,
                timeout=60,
                allowed_tries=2,
            ),
        }

        super().__init__(*args, **kwargs)
        self.bot_name = self.BOT_NAME

        # Reuse one HTTP client to avoid connection churn
        self._http = httpx.AsyncClient(timeout=HTTP_TIMEOUT_S)

    async def aclose(self) -> None:
        try:
            await self._http.aclose()
        except Exception:
            pass

    async def _post_json(self, url: str, headers: Dict[str, str], payload: Dict) -> Dict:
        r = await self._http.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

    async def _safe_search(self, coro):
        try:
            return await coro
        except Exception as e:
            logger.warning("Search failed: %s", e)
            return []

    async def linkup_search(self, query: str, max_results: int = 8, depth: str = "deep") -> List[Dict]:
        if not LINKUP_API_KEY:
            return []
        headers = {"Authorization": f"Bearer {LINKUP_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "q": query,
            "depth": depth,
            "outputType": "searchResults",
            "includeSources": False,
            "includeImages": False,
            "includeInlineCitations": False,
            "maxResults": max_results,
        }
        data = await self._post_json(LINKUP_ENDPOINT, headers, payload)
        return data.get("results", []) or []

    async def exa_search(self, query: str, max_results: int = 8, max_age_hours: Optional[int] = None) -> List[Dict]:
        if not EXA_API_KEY:
            return []
        headers = {"x-api-key": EXA_API_KEY, "Content-Type": "application/json"}
        payload: Dict[str, object] = {
            "query": query,
            "numResults": max_results,
            "type": "auto",
            "useAutoprompt": True,
            "contents": {"highlights": {"max_characters": 2000}},
        }
        if max_age_hours is not None:
            payload["maxAgeHours"] = int(max_age_hours)
        data = await self._post_json(EXA_ENDPOINT, headers, payload)
        return data.get("results", []) or []

    # -----------------------------
    # Prompt de-correlation variants
    # -----------------------------
    def _variant_prefix(self, i: int) -> str:
        variants = [
            "Variant A (Outside view heavy): Start from base rates/reference classes and update cautiously.",
            "Variant B (Skeptical): Assume status quo persists unless strong evidence; look for disconfirming facts.",
            "Variant C (Mechanistic/Fermi): Decompose into causal drivers; recombine and sanity-check units/timelines.",
            "Variant D (Devil’s advocate): Argue the opposite side first, then reconcile.",
            "Variant E (Surprises): Reserve mass for weird tails; identify failure modes and unknown unknowns.",
        ]
        return variants[i % len(variants)]

    # -----------------------------
    # Numeric/date sanitization
    # -----------------------------
    def _enforce_monotone_non_decreasing(self, xs: List[float]) -> List[float]:
        out: List[float] = []
        last: Optional[float] = None
        for x in xs:
            if last is None:
                last = x
            else:
                if x < last:
                    x = last
                last = x
            out.append(x)
        return out

    def _clamp_numeric_bounds(self, question: NumericQuestion, vals: List[float]) -> List[float]:
        lo = float(question.lower_bound)
        hi = float(question.upper_bound)

        out = vals[:]
        if not getattr(question, "open_lower_bound", False):
            out = [max(lo, v) for v in out]
        if not getattr(question, "open_upper_bound", False):
            out = [min(hi, v) for v in out]
        return out

    def _clamp_date_bounds(self, question: DateQuestion, vals_ts: List[float]) -> List[float]:
        lo = float(question.lower_bound.timestamp())
        hi = float(question.upper_bound.timestamp())

        out = vals_ts[:]
        if not getattr(question, "open_lower_bound", False):
            out = [max(lo, v) for v in out]
        if not getattr(question, "open_upper_bound", False):
            out = [min(hi, v) for v in out]
        return out

    def _sanitize_percentiles(
        self,
        question: Union[NumericQuestion, DateQuestion],
        plist: List[Percentile],
        is_date: bool,
    ) -> List[Percentile]:
        plist = sorted(plist, key=lambda p: float(p.percentile))
        vals = [float(p.value) for p in plist]

        if is_date:
            vals = self._clamp_date_bounds(question, vals)  # type: ignore[arg-type]
        else:
            vals = self._clamp_numeric_bounds(question, vals)  # type: ignore[arg-type]

        vals = self._enforce_monotone_non_decreasing(vals)

        if is_date:
            vals = self._clamp_date_bounds(question, vals)  # type: ignore[arg-type]
        else:
            vals = self._clamp_numeric_bounds(question, vals)  # type: ignore[arg-type]

        return [Percentile(percentile=plist[i].percentile, value=vals[i]) for i in range(len(plist))]

    # -----------------------------
    # Research (rank sources + top only)
    # -----------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            q = question.question_text.strip()
            criteria = (question.resolution_criteria or "").strip()
            fine_print = (question.fine_print or "").strip()

            query_core = q
            query_resolution = f"{q}\nResolution criteria keywords:\n{criteria[:600]}"

            linkup_task1 = asyncio.create_task(
                self._safe_search(self.linkup_search(query_core, max_results=8, depth="deep"))
            )
            linkup_task2 = asyncio.create_task(
                self._safe_search(self.linkup_search(query_resolution, max_results=6, depth="deep"))
            )
            exa_task1 = asyncio.create_task(self._safe_search(self.exa_search(query_core, max_results=10)))
            exa_task2 = asyncio.create_task(self._safe_search(self.exa_search(query_resolution, max_results=8)))

            linkup_1, linkup_2, exa_1, exa_2 = await asyncio.gather(
                linkup_task1, linkup_task2, exa_task1, exa_task2, return_exceptions=False
            )

            combined: List[Dict] = []
            combined.extend(linkup_1 or [])
            combined.extend(linkup_2 or [])
            combined.extend(exa_1 or [])
            combined.extend(exa_2 or [])

            sources_block, urls = _rank_and_format_sources(combined, max_to_keep=14)

            prompt = clean_indents(
                f"""
                You are a research assistant to a superforecaster.
                Task: produce a concise, decision-relevant briefing grounded in the retrieved sources.
                Do NOT produce a final forecast. Do NOT invent facts.

                Question:
                {q}

                Resolution criteria:
                {criteria}

                Fine print:
                {fine_print}

                Retrieved web snippets (ranked; each includes a URL):
                {sources_block}

                Output format:
                1) Key facts (6 bullets max)
                2) What would make this resolve YES vs NO (brief)
                3) Timeline / what's likely before resolution (brief)
                4) Source list (just the URLs, one per line)
                """
            )

            research = await self.get_llm("summarizer", "llm").invoke(prompt)
            url_list = "\n".join(urls[:30]) if urls else ""
            research_full = clean_indents(
                f"""
                {research}

                --- SOURCES (ranked URLs) ---
                {url_list}
                """
            ).strip()

            logger.info(f"Found Research for URL {getattr(question, 'page_url', '')}:\n{research_full}")
            return research_full

    async def _extract_evidence_summary(self, question: MetaculusQuestion, research: str) -> EvidenceSummary:
        prompt = clean_indents(
            f"""
            You are helping a forecaster judge how strong the current evidence is.

            Question:
            {question.question_text}

            Resolution criteria:
            {question.resolution_criteria}

            Research briefing + sources:
            {research}

            Decide:
            - direction: "yes", "no", or "unclear"
            - strength: a number from 0.0 to 1.0 meaning "how decisive the evidence is right now"
              (0.6 means: leaning outcome is supported by multiple credible sources; 0.9 means: near-certain / essentially already determined)
            - key_points: 3-5 short bullets, each must be grounded in the research text
            - sources: 3-8 URLs copied from the research

            Return ONLY valid JSON with keys: direction, strength, key_points, sources
            """
        )
        raw = await self.get_llm("default", "llm").invoke(prompt)

        # Preferred: structured parsing
        try:
            parsed: EvidenceSummaryOut = await structure_output(
                text_to_structure=raw,
                output_type=EvidenceSummaryOut,  # type: ignore
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )
            direction = parsed.direction if parsed.direction in ("yes", "no", "unclear") else "unclear"
            strength = _clamp01(float(parsed.strength))
            key_points = [str(x) for x in (parsed.key_points or [])][:6]
            sources = [str(s).strip() for s in (parsed.sources or []) if str(s).strip()][:12]
            return EvidenceSummary(direction=direction, strength=strength, key_points=key_points, sources=sources)
        except Exception:
            pass

        # Fallback: regex JSON extraction
        try:
            m = re.search(r"\{.*\}", raw, flags=re.S)
            obj = json.loads(m.group(0) if m else raw)
            direction = obj.get("direction", "unclear")
            if direction not in ("yes", "no", "unclear"):
                direction = "unclear"
            strength = _clamp01(float(obj.get("strength", 0.0)))
            key_points = obj.get("key_points") or []
            if not isinstance(key_points, list):
                key_points = [str(key_points)]
            sources = obj.get("sources") or []
            if not isinstance(sources, list):
                sources = [str(sources)]
            sources = [str(s).strip() for s in sources if str(s).strip()]
            return EvidenceSummary(
                direction=direction,
                strength=strength,
                key_points=[str(x) for x in key_points][:6],
                sources=sources[:12],
            )
        except Exception:
            return EvidenceSummary(direction="unclear", strength=0.0, key_points=[], sources=[])

    def _good_judgment_principles_block(self) -> str:
        return clean_indents(
            """
            Use two Good Judgment principles:
            1) Outside view first: start from base rates / reference classes, then update with specifics.
            2) Decompose + recombine: break the question into 3–7 drivers, estimate each, then aggregate.

            Also use a Fermi-style decomposition when estimating quantities/timelines:
            - Define units, horizon, and a baseline.
            - Estimate components with ranges.
            - Combine conservatively and sanity-check against known constraints.
            """
        )

    def _maybe_extremize_binary(self, p: float, ev: EvidenceSummary) -> float:
        if ev.strength >= self._evidence_extremize_threshold and ev.direction in ("yes", "no"):
            return _to_prob_decimal(
                self._extremize_target if ev.direction == "yes" else (1.0 - self._extremize_target)
            )
        return _to_prob_decimal(p)

    def _maybe_extremize_multichoice(self, probs: Dict[str, float], ev: EvidenceSummary) -> Dict[str, float]:
        if ev.strength < self._evidence_extremize_threshold:
            return _softmax_normalize(probs)
        top = max(probs.items(), key=lambda kv: kv[1])[0] if probs else None
        if not top:
            return probs
        out = {k: 0.0 for k in probs.keys()}
        out[top] = self._extremize_target
        rem = [k for k in probs.keys() if k != top]
        if rem:
            each = (1.0 - self._extremize_target) / len(rem)
            for k in rem:
                out[k] = each
        return _softmax_normalize(out)

    async def _sample_median_binary(
        self, question: BinaryQuestion, prompt: str, ev: EvidenceSummary
    ) -> ReasonedPrediction[float]:
        samples: List[float] = []
        reasonings: List[str] = []
        for i in range(self._fermi_samples):
            variant_prompt = clean_indents(f"{self._variant_prefix(i)}\n---\n{prompt}")
            reasoning = await self.get_llm("default", "llm").invoke(variant_prompt)
            reasonings.append(reasoning)
            parsed: BinaryPrediction = await structure_output(
                reasoning,
                BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )
            samples.append(_to_prob_decimal(parsed.prediction_in_decimal))

        p_med = _median_merge_lists(samples)
        p_final = self._maybe_extremize_binary(p_med, ev)

        compressed_rationales = "\n".join(
            [f"[Sample {j+1}] {_WS_RE.sub(' ', r)[:900]}" for j, r in enumerate(reasonings)]
        )

        combined_reasoning = clean_indents(
            f"""
            Bot name: {self.BOT_NAME}

            Evidence summary:
            - direction: {ev.direction}
            - strength: {ev.strength:.2f}
            - key points:
            {chr(10).join([f"- {kp}" for kp in ev.key_points]) if ev.key_points else "- (none)"}
            - sources:
            {chr(10).join(ev.sources) if ev.sources else "(none)"}

            Median-of-samples probability: {p_med:.3f}
            Final probability (after evidence rule): {p_final:.3f}

            --- Model rationales (compressed) ---
            {compressed_rationales}
            """
        ).strip()

        logger.info(f"Binary median samples for URL {getattr(question, 'page_url', '')}: {samples} -> {p_final}")
        return ReasonedPrediction(prediction_value=p_final, reasoning=combined_reasoning)

    async def _sample_median_multichoice(
        self, question: MultipleChoiceQuestion, prompt: str, ev: EvidenceSummary
    ) -> ReasonedPrediction[PredictedOptionList]:
        option_probs_samples: List[Dict[str, float]] = []
        reasonings: List[str] = []

        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            If the text prepends options with "Option" or similar, remove that prefix unless it is part of the option.
            Do not drop 0% options; include them explicitly.
            """
        )

        for i in range(self._fermi_samples):
            variant_prompt = clean_indents(f"{self._variant_prefix(i)}\n---\n{prompt}")
            reasoning = await self.get_llm("default", "llm").invoke(variant_prompt)
            reasonings.append(reasoning)
            pol: PredictedOptionList = await structure_output(
                text_to_structure=reasoning,
                output_type=PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
                additional_instructions=parsing_instructions,
            )
            d = {po.option: float(po.probability) for po in pol.predictions}
            for opt in question.options:
                d.setdefault(opt, 0.0)
            option_probs_samples.append(d)

        med_probs: Dict[str, float] = {}
        for opt in question.options:
            med_probs[opt] = float(median([s.get(opt, 0.0) for s in option_probs_samples]))

        med_probs = _softmax_normalize(med_probs)
        final_probs = self._maybe_extremize_multichoice(med_probs, ev)
        final_list = PredictedOptionList.from_dict(final_probs)

        compressed_rationales = "\n".join(
            [f"[Sample {j+1}] {_WS_RE.sub(' ', r)[:900]}" for j, r in enumerate(reasonings)]
        )

        combined_reasoning = clean_indents(
            f"""
            Bot name: {self.BOT_NAME}

            Evidence summary:
            - direction: {ev.direction}
            - strength: {ev.strength:.2f}
            - key points:
            {chr(10).join([f"- {kp}" for kp in ev.key_points]) if ev.key_points else "- (none)"}

            Median probabilities (pre-rule):
            {chr(10).join([f"{k}: {v:.4f}" for k, v in med_probs.items()])}

            Final probabilities (post-rule):
            {chr(10).join([f"{k}: {v:.4f}" for k, v in final_probs.items()])}

            --- Model rationales (compressed) ---
            {compressed_rationales}
            """
        ).strip()

        logger.info(f"MC median for URL {getattr(question, 'page_url', '')}: {final_probs}")
        return ReasonedPrediction(prediction_value=final_list, reasoning=combined_reasoning)

    async def _sample_median_numeric(
        self, question: Union[NumericQuestion, DateQuestion], prompt: str, is_date: bool
    ) -> ReasonedPrediction[NumericDistribution]:
        reasonings: List[str] = []
        percentile_samples: List[List[Percentile]] = []

        if is_date:
            parsing_instructions = clean_indents(
                """
                Parse a percentile distribution for a date question.

                IMPORTANT:
                - Output dates as ISO datetimes: YYYY-MM-DDT00:00:00Z (midnight UTC).
                - Do NOT output bare YYYY-MM-DD.
                - If percentiles are missing/ambiguous, indicate parsing failure rather than hallucinating.
                """
            )
        else:
            parsing_instructions = clean_indents(
                f"""
                Parse a percentile distribution for a numeric question.
                Respect units: {getattr(question, "unit_of_measure", "")}
                Never use scientific notation; convert if needed.
                If percentiles are missing, indicate parsing failure rather than hallucinating.
                """
            )

        for i in range(self._fermi_samples):
            variant_prompt = clean_indents(f"{self._variant_prefix(i)}\n---\n{prompt}")
            reasoning = await self.get_llm("default", "llm").invoke(variant_prompt)
            reasonings.append(reasoning)

            if is_date:
                date_percentiles: List[DatePercentile] = await structure_output(
                    reasoning,
                    List[DatePercentile],  # type: ignore
                    model=self.get_llm("parser", "llm"),
                    additional_instructions=parsing_instructions,
                    num_validation_samples=self._structure_output_validation_samples,
                )
                plist = [Percentile(percentile=dp.percentile, value=dp.value.timestamp()) for dp in date_percentiles]
                plist = self._sanitize_percentiles(question, plist, is_date=True)
                percentile_samples.append(plist)
            else:
                plist: List[Percentile] = await structure_output(
                    reasoning,
                    List[Percentile],  # type: ignore
                    model=self.get_llm("parser", "llm"),
                    additional_instructions=parsing_instructions,
                    num_validation_samples=self._structure_output_validation_samples,
                )
                plist = self._sanitize_percentiles(question, plist, is_date=False)
                percentile_samples.append(plist)

        target_percentiles = [10, 20, 40, 60, 80, 90]
        merged: List[Percentile] = []
        for pctl in target_percentiles:
            vals: List[float] = []
            for s in percentile_samples:
                hit = next((x for x in s if int(round(x.percentile)) == pctl), None)
                if hit is not None:
                    vals.append(float(hit.value))
            if not vals:
                fallback = float(getattr(question, "lower_bound", 0.0))
                vals = [fallback]
            merged.append(Percentile(percentile=pctl, value=float(median(vals))))

        merged = self._sanitize_percentiles(question, merged, is_date=is_date)
        dist = NumericDistribution.from_question(merged, question)  # type: ignore

        compressed_rationales = "\n".join(
            [f"[Sample {j+1}] {_WS_RE.sub(' ', r)[:900]}" for j, r in enumerate(reasonings)]
        )

        combined_reasoning = clean_indents(
            f"""
            Bot name: {self.BOT_NAME}

            Median-of-samples distribution:
            {chr(10).join([f"P{int(p.percentile)}: {p.value}" for p in merged])}

            --- Model rationales (compressed) ---
            {compressed_rationales}
            """
        ).strip()

        logger.info(f"Numeric/date median for URL {getattr(question, 'page_url', '')}: {dist.declared_percentiles}")
        return ReasonedPrediction(prediction_value=dist, reasoning=combined_reasoning)

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        ev = await self._extract_evidence_summary(question, research)
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            {self._good_judgment_principles_block()}

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}

            Research:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Write:
            (a) Time left until resolution.
            (b) Status quo outcome if nothing changes.
            (c) Outside view: base rate / reference class (brief).
            (d) Inside view: 3–7 drivers with rough probabilities, then recombine.
            (e) One plausible NO scenario and one plausible YES scenario.
            (f) A quick calibration check: "What evidence would flip me?"

            Last line:
            Probability: ZZ%
            """
        )
        return await self._sample_median_binary(question, prompt, ev)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        ev = await self._extract_evidence_summary(question, research)
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            {self._good_judgment_principles_block()}

            Question:
            {question.question_text}

            Options (must use these exact names):
            {question.options}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}

            Research:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Write:
            (a) Time left until resolution.
            (b) Status quo / default option if nothing changes.
            (c) Outside view (base rates / analogues).
            (d) Inside view: 3–7 drivers and which options they favor.
            (e) Surprise scenario: how a low-probability option could win.

            Final lines (exact format, all options, sum to ~100%):
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            """
        )
        return await self._sample_median_multichoice(question, prompt, ev)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        lower_bound_message, upper_bound_message = self._create_lower_and_upper_bound_messages(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            {self._good_judgment_principles_block()}

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}

            Units: {question.unit_of_measure if question.unit_of_measure else "Not stated (infer carefully)"}

            Research:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Instructions:
            - No scientific notation.
            - Percentiles must be strictly increasing.

            Write:
            (a) Time left until resolution.
            (b) Outside view: base rate / reference class.
            (c) Inside view: decompose into components; estimate each with ranges; recombine.
            (d) Market/expert expectations if any.
            (e) Low and high surprise scenarios.

            Final answer exactly:
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            """
        )
        return await self._sample_median_numeric(question, prompt, is_date=False)

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        lower_bound_message, upper_bound_message = self._create_lower_and_upper_bound_messages(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            {self._good_judgment_principles_block()}

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}

            Research:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Date formatting:
            - Use ISO datetime at midnight UTC: YYYY-MM-DDT00:00:00Z
            - Percentiles must be chronological increasing.

            Write:
            (a) Time left until resolution.
            (b) Outside view baseline timeline.
            (c) Inside view decomposition (milestones / gating steps).
            (d) Low/high surprise scenarios.

            Final answer exactly:
            Percentile 10: YYYY-MM-DDT00:00:00Z
            Percentile 20: YYYY-MM-DDT00:00:00Z
            Percentile 40: YYYY-MM-DDT00:00:00Z
            Percentile 60: YYYY-MM-DDT00:00:00Z
            Percentile 80: YYYY-MM-DDT00:00:00Z
            Percentile 90: YYYY-MM-DDT00:00:00Z
            """
        )
        return await self._sample_median_numeric(question, prompt, is_date=True)

    def _create_lower_and_upper_bound_messages(
        self, question: Union[NumericQuestion, DateQuestion]
    ) -> Tuple[str, str]:
        # Return (lower_message, upper_message) to match variable names and prevent order confusion.
        if isinstance(question, NumericQuestion):
            upper_bound_number = (
                question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
            )
            lower_bound_number = (
                question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
            )
            unit_of_measure = question.unit_of_measure or ""
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError("Unsupported question type for bounds")

        if getattr(question, "open_upper_bound", False):
            upper_bound_message = (
                f"The question creator thinks the outcome is likely not higher/later than {upper_bound_number} {unit_of_measure}."
            )
        else:
            upper_bound_message = f"The outcome cannot be higher/later than {upper_bound_number} {unit_of_measure}."

        if getattr(question, "open_lower_bound", False):
            lower_bound_message = (
                f"The question creator thinks the outcome is likely not lower/earlier than {lower_bound_number} {unit_of_measure}."
            )
        else:
            lower_bound_message = f"The outcome cannot be lower/earlier than {lower_bound_number} {unit_of_measure}."

        return lower_bound_message, upper_bound_message

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(question.parent, research, "parent")
        child_info, full_research = await self._get_question_prediction_info(question.child, full_research, "child")
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(question.question_no, full_research, "no")

        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
            """
        ).strip()

        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(reasoning=full_reasoning, prediction_value=full_prediction)

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> Tuple[ReasonedPrediction[Union[PredictionTypes, PredictionAffirmed]], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if previous_forecast.timestamp_end is None or previous_forecast.timestamp_end > current_utc_time:
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return prediction, research  # type: ignore

        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self, research: str, reasoning: ReasonedPrediction[PredictionTypes], question_type: str
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        qt = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {qt} Question Information
            You have previously forecasted the {qt} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast.

            Reasoning used previously:
            ```
            {reasoning.reasoning}
            ```
            """
        ).strip()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description='Run the Linkup+Exa Spring 2026 Bot ("nike")')
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode  # type: ignore

    bot = LinkupExaSpringBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )

    client = PatchedMetaculusClient()

    try:
        if run_mode == "tournament":
            # CRITICAL CHANGE:
            # - fetch tournament questions using PATCHED client
            # - forecast using forecast_questions() to ensure patched parsing applies
            seasonal_qs = await _gather_tournament_questions(client, client.CURRENT_AI_COMPETITION_ID)
            minibench_qs = await _gather_tournament_questions(client, client.CURRENT_MINIBENCH_ID)
            market_pulse_qs = await _gather_tournament_questions(client, MARKET_PULSE_TOURNAMENT_SLUG)

            questions = seasonal_qs + minibench_qs + market_pulse_qs
            reports = await bot.forecast_questions(questions, return_exceptions=True)

        elif run_mode == "metaculus_cup":
            bot.skip_previously_forecasted_questions = False
            cup_qs = await _gather_tournament_questions(client, client.CURRENT_METACULUS_CUP_ID)
            reports = await bot.forecast_questions(cup_qs, return_exceptions=True)

        else:
            EXAMPLE_QUESTIONS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
                "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
                "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
                "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
            ]
            bot.skip_previously_forecasted_questions = False
            questions = [client.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
            reports = await bot.forecast_questions(questions, return_exceptions=True)

        bot.log_report_summary(reports)

    finally:
        await bot.aclose()


if __name__ == "__main__":
    asyncio.run(main())
