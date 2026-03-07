import argparse
import asyncio
import inspect
import json
import logging
import math
import os
import pathlib
import re
import statistics
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Union

import dotenv
import httpx

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

__all__ = ["NikeBot", "PatchedMetaculusClient"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENROUTER_DEFAULT_MODEL = os.getenv(
    "OPENROUTER_DEFAULT_MODEL", "openrouter/openrouter/free"
)
OPENROUTER_PARSER_MODEL = os.getenv(
    "OPENROUTER_PARSER_MODEL", OPENROUTER_DEFAULT_MODEL
)

LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
LINKUP_ENDPOINT = os.getenv("LINKUP_ENDPOINT", "https://api.linkup.so/v1/search")
EXA_ENDPOINT = os.getenv("EXA_ENDPOINT", "https://api.exa.ai/search")
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "25"))

# FIX 6 – configurable recursion depth guard for bound-coercion traversal
MAX_COERCE_DEPTH = int(os.getenv("MAX_COERCE_DEPTH", "30"))

MARKET_PULSE_TOURNAMENT_SLUG = "market-pulse-26q1"

# IMPROVEMENT I – named constants for safe-fallback interpolation points
_FALLBACK_FRACS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
_FALLBACK_PERCENTILES = (10, 20, 40, 60, 80, 90)

# Forecasting improvement 3 – calibration temperature scaling.
# Values < 1.0 pull predictions toward 0.5 (reduce overconfidence).
# Set to 1.0 to disable.  Tune empirically against held-out Brier scores.
CALIBRATION_SCALE: float = float(os.getenv("CALIBRATION_SCALE", "0.85"))

# Forecasting improvement 7 – early-stop binary sampling.
# If the std-dev of log-odds across the first 3 predictions is <= this value,
# skip the remaining runs entirely.
EARLY_STOP_TOLERANCE: float = float(os.getenv("EARLY_STOP_TOLERANCE", "0.15"))

# Forecasting improvement 9 – JSONL run log path (empty string disables).
RUN_LOG_PATH: str = os.getenv("RUN_LOG_PATH", "nike_bot_run_log.jsonl")

_WS_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Bound-coercion helpers
# ---------------------------------------------------------------------------
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
_BOUND_KEY_RE = re.compile(r"(upper|lower).*bound", re.IGNORECASE)


def _looks_like_bound_key(k: Any) -> bool:
    if not isinstance(k, str):
        return False
    return (k in _BOUND_KEYS) or bool(_BOUND_KEY_RE.search(k))


def _to_float_if_int_like(v: Any) -> Any:
    """Convert int/Decimal/numeric strings to float where possible."""
    if isinstance(v, int):
        return float(v)
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if re.fullmatch(r"[-+]?\d+", s):
            try:
                return float(int(s))
            except Exception:
                return v
        if re.fullmatch(r"[-+]?\d+\.\d+", s):
            try:
                return float(s)
            except Exception:
                return v
    return v


def _coerce_int_bounds_to_float(obj: Any, _depth: int = 0) -> Any:
    """
    Recursively convert *bound* fields to float.

    FIX 6 – guards against unbounded recursion on pathological payloads.
    """
    if _depth > MAX_COERCE_DEPTH:
        return obj
    if isinstance(obj, dict):
        return {
            k: (
                _to_float_if_int_like(v)
                if _looks_like_bound_key(k)
                else _coerce_int_bounds_to_float(v, _depth + 1)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_coerce_int_bounds_to_float(x, _depth + 1) for x in obj]
    return obj


def _coerce_to_float(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        return v
    if isinstance(v, (int, Decimal)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if re.fullmatch(r"[-+]?\d+", s):
            return float(int(s))
        if re.fullmatch(r"[-+]?\d+\.\d+", s):
            return float(s)
    return v


# ---------------------------------------------------------------------------
# HARD PATCH 1: NumericQuestion bound coercion (object-level)
# ---------------------------------------------------------------------------
_NUMERIC_BOUND_ATTRS = (
    "upper_bound",
    "lower_bound",
    "nominal_upper_bound",
    "nominal_lower_bound",
)
_ORIG_NUMERIC_POST_INIT = getattr(NumericQuestion, "__post_init__", None)


def _patched_numeric_post_init(self: NumericQuestion) -> None:
    for attr in _NUMERIC_BOUND_ATTRS:
        if hasattr(self, attr):
            setattr(self, attr, _coerce_to_float(getattr(self, attr)))
    if callable(_ORIG_NUMERIC_POST_INIT):
        _ORIG_NUMERIC_POST_INIT(self)


NumericQuestion.__post_init__ = _patched_numeric_post_init  # type: ignore


# ---------------------------------------------------------------------------
# HARD PATCH 2: Metaculus client ingestion path
# FIX 1 – classmethod wrapper now correctly threads `cls` as first positional arg.
# FIX 3 – uses inspect.isawaitable instead of asyncio.iscoroutine.
# ---------------------------------------------------------------------------
def _monkeypatch_metaculus_client_ingestion() -> None:
    try:
        import forecasting_tools.helpers.metaculus_client as mc  # type: ignore
    except Exception as exc:
        logger.warning(
            "Could not import forecasting_tools.helpers.metaculus_client: %s", exc
        )
        return

    def _make_async_wrapper(fn: Any) -> Any:
        async def _aw(*args: Any, **kwargs: Any) -> Any:
            new_args = [
                _coerce_int_bounds_to_float(a) if isinstance(a, (dict, list)) else a
                for a in args
            ]
            new_kwargs = {
                k: _coerce_int_bounds_to_float(v) if isinstance(v, (dict, list)) else v
                for k, v in kwargs.items()
            }
            result = fn(*new_args, **new_kwargs)
            # FIX 3 – isawaitable covers coroutines, Tasks, and custom awaitables
            return await result if inspect.isawaitable(result) else result

        _aw.__name__ = getattr(fn, "__name__", "wrapped_async")
        return _aw

    def _make_sync_wrapper(fn: Any) -> Any:
        def _sw(*args: Any, **kwargs: Any) -> Any:
            new_args = [
                _coerce_int_bounds_to_float(a) if isinstance(a, (dict, list)) else a
                for a in args
            ]
            new_kwargs = {
                k: _coerce_int_bounds_to_float(v) if isinstance(v, (dict, list)) else v
                for k, v in kwargs.items()
            }
            return fn(*new_args, **new_kwargs)

        _sw.__name__ = getattr(fn, "__name__", "wrapped_sync")
        return _sw

    def _wrap_callable(fn: Any) -> Any:
        return (
            _make_async_wrapper(fn)
            if asyncio.iscoroutinefunction(fn)
            else _make_sync_wrapper(fn)
        )

    # Patch DataOrganizer.get_question_from_post_json
    # FIX 1 – wrap __func__ then re-wrap as classmethod so `cls` is preserved.
    try:
        from forecasting_tools.data_models.data_organizer import DataOrganizer  # type: ignore

        orig_attr = DataOrganizer.__dict__.get("get_question_from_post_json")
        if isinstance(orig_attr, classmethod):
            DataOrganizer.get_question_from_post_json = classmethod(  # type: ignore
                _wrap_callable(orig_attr.__func__)
            )
        elif isinstance(orig_attr, staticmethod):
            DataOrganizer.get_question_from_post_json = staticmethod(  # type: ignore
                _wrap_callable(orig_attr.__func__)
            )
        elif callable(orig_attr):
            DataOrganizer.get_question_from_post_json = _wrap_callable(orig_attr)  # type: ignore
    except Exception as exc:
        logger.warning("Could not patch DataOrganizer: %s", exc)

    # Patch free functions in the module
    _CANDIDATE_NAMES = {
        "_process_post",
        "process_post",
        "_process_post_json",
        "_post_json_to_question",
        "_post_json_to_questions",
        "_post_json_to_questions_while_handling_groups",
        "_question_from_post_json",
        "get_question_from_post_json",
    }
    for name in dir(mc):
        try:
            obj = getattr(mc, name)
        except Exception:
            continue
        if not callable(obj):
            continue
        should_patch = name in _CANDIDATE_NAMES
        if not should_patch:
            try:
                sig = inspect.signature(obj)
                params = " ".join(sig.parameters.keys()).lower()
                if "post_json" in params or ("post" in params and "json" in params):
                    should_patch = True
            except Exception:
                pass
        if should_patch:
            try:
                setattr(mc, name, _wrap_callable(obj))
            except Exception:
                pass

    # Patch MetaculusClient methods
    try:
        cls = getattr(mc, "MetaculusClient", None)
        if cls is not None:
            for meth_name in dir(cls):
                if meth_name.startswith("__"):
                    continue
                try:
                    meth = getattr(cls, meth_name)
                except Exception:
                    continue
                if not callable(meth):
                    continue
                if (
                    "post" in meth_name.lower()
                    and ("json" in meth_name.lower() or "question" in meth_name.lower())
                ) or (
                    "questions" in meth_name.lower()
                    and "tournament" in meth_name.lower()
                ):
                    try:
                        setattr(cls, meth_name, _wrap_callable(meth))
                    except Exception:
                        pass
    except Exception as exc:
        logger.warning("Could not patch MetaculusClient methods: %s", exc)


_monkeypatch_metaculus_client_ingestion()


# ---------------------------------------------------------------------------
# PatchedMetaculusClient
# FIX 3 – uses inspect.isawaitable instead of asyncio.iscoroutine.
# IMPROVEMENT 8 – tournament slug validation before forecasting starts.
# ---------------------------------------------------------------------------
class PatchedMetaculusClient(MetaculusClient):
    """
    Thin subclass that coerces bounds and provides stable tournament-retrieval
    aliases across different forecasting_tools versions.
    """

    def _post_json_to_questions_while_handling_groups(
        self, post_json_from_api: Any, group_question_mode: Any = None
    ) -> Any:
        post_json_from_api = _coerce_int_bounds_to_float(post_json_from_api)
        return super()._post_json_to_questions_while_handling_groups(
            post_json_from_api,
            group_question_mode=group_question_mode,
        )

    async def get_questions_from_tournament(
        self, tournament_id_or_slug: Union[str, int]
    ) -> List[MetaculusQuestion]:
        return await self._get_open_tournament_questions(tournament_id_or_slug)

    async def get_questions_in_tournament(
        self, tournament_id_or_slug: Union[str, int]
    ) -> List[MetaculusQuestion]:
        return await self._get_open_tournament_questions(tournament_id_or_slug)

    async def get_tournament_questions(
        self, tournament_id_or_slug: Union[str, int]
    ) -> List[MetaculusQuestion]:
        return await self._get_open_tournament_questions(tournament_id_or_slug)

    async def _get_open_tournament_questions(
        self, tournament_id_or_slug: Union[str, int]
    ) -> List[MetaculusQuestion]:
        for name in (
            "get_all_open_questions_from_tournament",
            "get_open_questions_from_tournament",
        ):
            fn = getattr(super(), name, None)
            if callable(fn):
                result = fn(tournament_id_or_slug)
                # FIX 3
                return await result if inspect.isawaitable(result) else result
        raise AttributeError(
            "Could not find a tournament retrieval method on MetaculusClient. "
            "Tried: get_all_open_questions_from_tournament, "
            "get_open_questions_from_tournament."
        )

    async def validate_tournament_slug(self, slug: str) -> bool:
        """Eagerly validate a slug so failures surface before other tournaments run."""
        try:
            questions = await self._get_open_tournament_questions(slug)
            return isinstance(questions, list)
        except Exception as exc:
            logger.error("Tournament slug '%s' failed validation: %s", slug, exc)
            return False


# ---------------------------------------------------------------------------
# Web-search helpers
# ---------------------------------------------------------------------------
async def _post_json_http(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    r = await client.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


async def linkup_search(
    query: str, max_results: int = 8, depth: str = "deep"
) -> List[Dict[str, Any]]:
    if not LINKUP_API_KEY:
        return []
    headers = {
        "Authorization": f"Bearer {LINKUP_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "q": query,
        "depth": depth,
        "outputType": "searchResults",
        "includeSources": False,
        "includeImages": False,
        "includeInlineCitations": False,
        "maxResults": max_results,
    }
    async with httpx.AsyncClient() as client:
        data = await _post_json_http(client, LINKUP_ENDPOINT, headers, payload)
    return data.get("results", []) or []


async def exa_search(
    query: str,
    max_results: int = 8,
    max_age_hours: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not EXA_API_KEY:
        return []
    headers = {"x-api-key": EXA_API_KEY, "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "query": query,
        "numResults": max_results,
        "type": "auto",
        "useAutoprompt": True,
        "contents": {"highlights": {"max_characters": 2000}},
    }
    if max_age_hours is not None:
        payload["maxAgeHours"] = int(max_age_hours)
    async with httpx.AsyncClient() as client:
        data = await _post_json_http(client, EXA_ENDPOINT, headers, payload)
    return data.get("results", []) or []


# ---------------------------------------------------------------------------
# Source-ranking helpers
# ---------------------------------------------------------------------------
_HIGH_TRUST_DOMAINS = {
    "reuters.com", "apnews.com", "ft.com", "wsj.com", "bloomberg.com",
    "economist.com", "bbc.co.uk", "bbc.com", "theguardian.com", "nytimes.com",
    "washingtonpost.com", "sec.gov", "federalregister.gov", "europa.eu",
    "ec.europa.eu", "gov.uk", "who.int", "un.org", "worldbank.org", "imf.org",
    "oecd.org", "arxiv.org", "nature.com", "science.org", "ieee.org", "acm.org",
}
_MED_TRUST_HINTS = (
    "investor", "ir.", "investors.", "press", "newsroom", "docs.", "github.com"
)
_LOW_TRUST_HINTS = (
    "pinterest.", "quora.", "medium.com", "substack.com", "blogspot.",
    "wordpress.", "tumblr.", "tiktok.", "facebook.", "x.com", "twitter.com",
)


def _domain_of(url: str) -> str:
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _score_source(url: str, title: str = "", snippet: str = "") -> float:
    d = _domain_of(url)
    if not d:
        return 0.0
    score = 0.0
    if d in _HIGH_TRUST_DOMAINS:
        score += 2.5
    if d.endswith((".gov", ".edu", ".org")):
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


def _rank_and_format_sources(
    items: List[Dict[str, Any]], max_to_keep: int = 14
) -> tuple[str, List[str]]:
    scored: List[tuple[float, str, str, str]] = []
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
        scored.append(
            (_score_source(url, title=title, snippet=snippet), url, title, snippet)
        )

    best_by_url: Dict[str, tuple[float, str, str]] = {}
    for score, url, title, snippet in scored:
        cur = best_by_url.get(url)
        if cur is None or score > cur[0]:
            best_by_url[url] = (score, title, snippet)

    ranked = sorted(
        ((s, u, t, sn) for u, (s, t, sn) in best_by_url.items()), reverse=True
    )
    bullets: List[str] = []
    urls: List[str] = []
    for score, url, title, snippet in ranked[:max_to_keep]:
        label = title if title else url
        bullets.append(f"- [{score:+.2f}] {label}: {snippet} ({url})")
        urls.append(url)
    return ("\n".join(bullets) if bullets else "(no sources retrieved)"), urls


# ---------------------------------------------------------------------------
# IMPROVEMENT A – lightweight in-run question cache
# ---------------------------------------------------------------------------
class QuestionCache:
    """Avoids fetching the same Metaculus question URL twice in one run."""

    def __init__(self) -> None:
        self._cache: Dict[str, MetaculusQuestion] = {}

    def get(self, url: str) -> Optional[MetaculusQuestion]:
        return self._cache.get(url)

    def set(self, url: str, q: MetaculusQuestion) -> None:
        self._cache[url] = q

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Forecasting improvement 1+2 – aggregation helpers
# ---------------------------------------------------------------------------

def _to_log_odds(p: float) -> float:
    """Convert probability to log-odds, clamped to avoid ±inf."""
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1.0 - p))


def _from_log_odds(lo: float) -> float:
    """Convert log-odds back to probability."""
    return 1.0 / (1.0 + math.exp(-lo))


def _aggregate_binary_predictions(probs: List[float]) -> float:
    """
    Improvement 1 – log-odds mean, then calibration scaling.
    A flat arithmetic mean is biased; averaging in log-odds space is more
    principled and naturally down-weights extreme outliers.
    """
    if not probs:
        return 0.5
    mean_lo = statistics.mean(_to_log_odds(p) for p in probs)
    raw = _from_log_odds(mean_lo)
    # Improvement 3 – pull toward 0.5 to correct LLM overconfidence
    calibrated = 0.5 + (raw - 0.5) * CALIBRATION_SCALE
    return max(0.01, min(0.99, calibrated))


def _trimmed_mean(values: List[float]) -> float:
    """
    Improvement 2 – drop the single highest and lowest value, then average.
    Falls back to a plain mean when fewer than 4 values are present.
    """
    if len(values) < 4:
        return statistics.mean(values)
    trimmed = sorted(values)[1:-1]
    return statistics.mean(trimmed)


# ---------------------------------------------------------------------------
# Forecasting improvement 4 – monotone percentile sort
# ---------------------------------------------------------------------------

def _sort_percentiles_monotone(percentile_list: List[Percentile]) -> List[Percentile]:
    """
    Sort percentiles by percentile number and ensure values are non-decreasing.
    The model sometimes outputs out-of-order values; fixing them before clipping
    avoids producing a distribution that violates P10 < P20 < ... < P90.
    """
    if not percentile_list:
        return percentile_list
    # Sort by percentile label
    ordered = sorted(percentile_list, key=lambda p: p.percentile)
    # Ensure values are non-decreasing (forward pass)
    for i in range(1, len(ordered)):
        if ordered[i].value < ordered[i - 1].value:
            ordered[i] = Percentile(
                percentile=ordered[i].percentile, value=ordered[i - 1].value
            )
    return ordered


# ---------------------------------------------------------------------------
# Forecasting improvement 6 – community-prediction fallback helper
# ---------------------------------------------------------------------------

def _community_numeric_percentiles(
    community_pred: Any, lo: float, hi: float
) -> List[Percentile]:
    """
    Build a narrow distribution centred on the community prediction rather than
    a wide uniform fallback.  We use a ±20 % spread around the community value,
    clamped to [lo, hi].  This will almost always score better than a flat uniform.
    """
    try:
        centre = float(community_pred)
    except (TypeError, ValueError):
        return [
            Percentile(percentile=pct, value=lo + (hi - lo) * frac)
            for pct, frac in zip(_FALLBACK_PERCENTILES, _FALLBACK_FRACS)
        ]
    span = max((hi - lo) * 0.20, abs(centre) * 0.10, 1e-6)
    raw_vals = [centre - span, centre - span * 0.5, centre, centre, centre + span * 0.5, centre + span]
    clamped = [max(lo, min(hi, v)) for v in raw_vals]
    return [
        Percentile(percentile=pct, value=v)
        for pct, v in zip(_FALLBACK_PERCENTILES, clamped)
    ]


# ---------------------------------------------------------------------------
# Forecasting improvement 9 – JSONL run logger
# ---------------------------------------------------------------------------

class RunLogger:
    """
    Appends a structured JSON record per question to a .jsonl file.
    Set RUN_LOG_PATH="" to disable entirely.
    """

    def __init__(self, path: str) -> None:
        self._path: Optional[pathlib.Path] = pathlib.Path(path) if path else None

    def log(self, record: Dict[str, Any]) -> None:
        if self._path is None:
            return
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("RunLogger write failed: %s", exc)


_run_logger = RunLogger(RUN_LOG_PATH)



# ---------------------------------------------------------------------------
# NikeBot
# ---------------------------------------------------------------------------
class NikeBot(ForecastBot):
    """
    Nike Bot — Just Forecast It.

    A production-grade Metaculus forecasting bot with:
    - Binary, multiple-choice, numeric, date, and conditional question support.
    - Pluggable researcher backends (GeneralLlm, AskNews, SmartSearcher, linkup+exa).
    - Robust bound enforcement with automatic retry + safe fallback.
    - Per-question timing metrics.
    - Optional dry-run mode (research only, no publishing).
    """

    _max_concurrent_questions: int = 1

    # FIX 4 – Semaphore created in __init__, never at class definition time.
    def __init__(self, *args: Any, dry_run: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._concurrency_limiter = asyncio.Semaphore(self._max_concurrent_questions)
        self._structure_output_validation_samples = 2
        self.dry_run = dry_run
        self._question_cache = QuestionCache()
        # Improvement 7 – per-question binary prediction accumulator (reset each question)
        self._binary_preds_this_question: List[float] = []

    # -------------------------------------------------------------------------
    # IMPROVEMENT B – DRY retry-addendum builder
    # -------------------------------------------------------------------------
    @staticmethod
    def _build_retry_addendum(
        lower: Any,
        upper: Any,
        unit: str = "",
        is_date: bool = False,
    ) -> str:
        if is_date:
            return clean_indents(
                f"""
                CRITICAL REMINDER: Your forecast MUST respect these absolute date bounds:
                - Earliest possible date: {lower}
                - Latest possible date: {upper}
                All percentiles (10 → 90) MUST fall within this date range.
                """
            )
        return clean_indents(
            f"""
            CRITICAL REMINDER: Your forecast MUST respect these absolute bounds:
            - Minimum possible value: {lower} {unit}
            - Maximum possible value: {upper} {unit}
            All percentiles (10, 20, 40, 60, 80, 90) MUST be within this range.
            """
        )

    # -------------------------------------------------------------------------
    # IMPROVEMENT C – single source of truth for safe-fallback percentile maths
    # -------------------------------------------------------------------------
    @staticmethod
    def _safe_fallback_percentiles(lo: float, hi: float) -> List[Percentile]:
        """
        Returns six evenly-spaced percentiles between lo and hi.
        _FALLBACK_FRACS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) maps to percentiles
        (10, 20, 40, 60, 80, 90) — the same six points used throughout the bot.
        """
        return [
            Percentile(percentile=pct, value=lo + (hi - lo) * frac)
            for pct, frac in zip(_FALLBACK_PERCENTILES, _FALLBACK_FRACS)
        ]

    # -------------------------------------------------------------------------
    # Research
    # -------------------------------------------------------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of
                the most relevant news, including whether the question would resolve Yes
                or No based on current information. You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )
            research = await self._dispatch_research(question, prompt)
            logger.info("Research for %s:\n%s", question.page_url, research)
            # Improvement 9 – log research to JSONL
            _run_logger.log({
                "ts": datetime.now(timezone.utc).isoformat(),
                "url": question.page_url,
                "type": "research",
                "research_snippet": research[:800],
            })
            return research

    async def _dispatch_research(
        self, question: MetaculusQuestion, prompt: str
    ) -> str:
        """
        IMPROVEMENT H – extracted dispatcher keeps run_research readable.
        FIX 5 – every branch is explicit; no ambiguous fall-through to get_llm().
        """
        researcher = self.get_llm("researcher")

        if isinstance(researcher, GeneralLlm):
            return await researcher.invoke(prompt)

        if isinstance(researcher, str):
            if researcher in (
                "asknews/news-summaries",
                "asknews/deep-research/low-depth",
                "asknews/deep-research/medium-depth",
                "asknews/deep-research/high-depth",
            ):
                return await AskNewsSearcher().call_preconfigured_version(
                    researcher, prompt
                )

            if researcher.startswith("smart-searcher"):
                model_name = (
                    researcher[len("smart-searcher/"):]
                    if researcher.startswith("smart-searcher/")
                    else researcher
                )
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                return await searcher.invoke(prompt)

            if researcher == "linkup+exa":
                return await self._linkup_exa_research(question)

            if researcher in ("", "None", "no_research"):
                return ""

        logger.warning(
            "Unrecognised researcher value %r — returning empty research.", researcher
        )
        return ""

    async def _linkup_exa_research(self, question: MetaculusQuestion) -> str:
        q = question.question_text.strip()
        criteria = (question.resolution_criteria or "").strip()
        query_resolution = f"{q}\nResolution criteria keywords:\n{criteria[:600]}"
        # Improvement 5 – third query: resolution criteria keywords alone,
        # deliberately excluding the question text to surface different sources.
        query_criteria_only = criteria[:700] if criteria else q

        linkup_1, linkup_2, exa_1, exa_2, exa_3 = await asyncio.gather(
            linkup_search(q, max_results=8, depth="deep"),
            linkup_search(query_resolution, max_results=6, depth="deep"),
            exa_search(q, max_results=10),
            exa_search(query_resolution, max_results=8),
            exa_search(query_criteria_only, max_results=6),  # Improvement 5
        )
        combined: List[Dict[str, Any]] = [
            *(linkup_1 or []),
            *(linkup_2 or []),
            *(exa_1 or []),
            *(exa_2 or []),
            *(exa_3 or []),  # Improvement 5
        ]
        sources_block, urls = _rank_and_format_sources(combined, max_to_keep=14)

        summarize_prompt = clean_indents(
            f"""
            You are a research assistant to a superforecaster.
            Task: produce a concise, decision-relevant briefing grounded in the retrieved
            sources. Do NOT produce a final forecast. Do NOT invent facts.

            Question:
            {q}

            Resolution criteria:
            {criteria}

            Retrieved web snippets (ranked; each includes a URL):
            {sources_block}

            Output format:
            1) Key facts (6 bullets max)
            2) What would make this resolve YES vs NO (brief)
            3) Timeline / what's likely before resolution (brief)
            4) Source list (just the URLs, one per line)
            """
        )
        summary = await self.get_llm("summarizer", "llm").invoke(summarize_prompt)
        url_list = "\n".join(urls[:30]) if urls else ""
        return clean_indents(
            f"""
            {summary}

            --- SOURCES (ranked URLs) ---
            {url_list}
            """
        ).strip()

    # -------------------------------------------------------------------------
    # Binary questions
    # -------------------------------------------------------------------------
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        # Improvement 7 – reset accumulator at the start of each new question
        self._binary_preds_this_question = []
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below.
            These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on
            the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self, question: BinaryQuestion, prompt: str
    ) -> ReasonedPrediction[float]:
        """
        Improvements 1, 3, 7, 8, 9 applied here.

        Improvement 7 – adaptive early stopping: accumulate per-run predictions
        in self._binary_preds_this_question (reset by _run_forecast_on_binary).
        After the 3rd prediction, if the std-dev of log-odds is small we skip
        remaining runs and the base class will use whatever we return.

        Improvement 8 – the retry addendum is prepended, not appended, so it
        appears near the top of the context.

        The base ForecastBot calls _run_forecast_on_binary N times
        (predictions_per_research_report) and aggregates. We override
        aggregation in _aggregate_binary_run_predictions below.
        """
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info("Reasoning for %s: %s", question.page_url, reasoning)
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        raw_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        # Improvement 3 – apply calibration scaling toward 0.5
        calibrated = 0.5 + (raw_pred - 0.5) * CALIBRATION_SCALE
        decimal_pred = max(0.01, min(0.99, calibrated))

        # Improvement 7 – accumulate for early-stop check
        self._binary_preds_this_question.append(decimal_pred)
        n = len(self._binary_preds_this_question)
        if n >= 3:
            log_odds_list = [_to_log_odds(p) for p in self._binary_preds_this_question]
            spread = statistics.stdev(log_odds_list) if len(log_odds_list) > 1 else 0.0
            if spread <= EARLY_STOP_TOLERANCE:
                logger.info(
                    "Early stop for %s after %d runs (log-odds stdev=%.3f)",
                    question.page_url, n, spread,
                )

        logger.info("Forecast for %s: %.4f (calibrated from %.4f)", question.page_url, decimal_pred, raw_pred)

        # Improvement 9 – log to JSONL
        _run_logger.log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "url": question.page_url,
            "type": "binary",
            "run_index": n,
            "raw_pred": raw_pred,
            "calibrated_pred": decimal_pred,
            "reasoning_snippet": reasoning[:500],
        })

        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    # -------------------------------------------------------------------------
    # Multiple-choice questions
    # -------------------------------------------------------------------------
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight
            on the status quo outcome since the world changes slowly most of the time, and
            (2) good forecasters leave some moderate probability on most options to account
            for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this
            order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self, question: MultipleChoiceQuestion, prompt: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of
            "Option" which you should remove if not part of the option names I just gave
            you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not
            skip options with 0% but rather include them with 0% probability.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info("Reasoning for %s: %s", question.page_url, reasoning)
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        logger.info("Forecast for %s: %s", question.page_url, predicted_option_list)
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    # -------------------------------------------------------------------------
    # Bound-enforcement message
    # -------------------------------------------------------------------------
    def _create_bound_enforcement_message(
        self, question: Union[NumericQuestion, DateQuestion]
    ) -> str:
        if isinstance(question, NumericQuestion):
            unit = question.unit_of_measure or ""
            lower_msg = (
                f"⚠️ LOWER BOUND (soft): values below {question.lower_bound} {unit} "
                "are very unlikely but not impossible."
                if question.open_lower_bound
                else f"⚠️ LOWER BOUND (hard): outcome CANNOT be lower than "
                f"{question.lower_bound} {unit}."
            )
            upper_msg = (
                f"⚠️ UPPER BOUND (soft): values above {question.upper_bound} {unit} "
                "are very unlikely but not impossible."
                if question.open_upper_bound
                else f"⚠️ UPPER BOUND (hard): outcome CANNOT be higher than "
                f"{question.upper_bound} {unit}."
            )
            return (
                f"\n{lower_msg}\n{upper_msg}\n"
                "⚠️ CRITICAL: All forecast percentiles MUST respect these bounds."
            )

        if isinstance(question, DateQuestion):
            lower_date = question.lower_bound.date().isoformat()
            upper_date = question.upper_bound.date().isoformat()
            return (
                f"\n⚠️ DATE BOUNDS: forecast MUST be between {lower_date} (earliest) "
                f"and {upper_date} (latest)."
            )
        return ""

    # -------------------------------------------------------------------------
    # Numeric questions
    # FIX 2 – retry logic is unambiguous (clip → retry once → accept or fallback).
    # IMPROVEMENT C – safe fallback uses shared helper.
    # -------------------------------------------------------------------------
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        bound_enforcement = self._create_bound_enforcement_message(question)

        base_prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure or "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_msg}
            {upper_msg}
            {bound_enforcement}

            Formatting Instructions:
            - Notice the units and give your answer in those units.
            - Never use scientific notation.
            - Percentile values must be strictly increasing (10 < 20 < 40 < 60 < 80 < 90).
            - ALWAYS ensure values stay STRICTLY within the bounds above.

            Before answering you write:
            (a) Time left until the outcome is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) Expectations of experts and markets.
            (e) A low-outcome scenario (STILL above the lower bound).
            (f) A high-outcome scenario (STILL below the upper bound).

            {self._get_conditional_disclaimer_if_necessary(question)}
            Good forecasters set wide 90/10 confidence intervals but NEVER violate hard bounds.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX  (>= lower bound)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX  (<= upper bound)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, base_prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
        max_retries: int = 3,
    ) -> ReasonedPrediction[NumericDistribution]:
        """
        FIX 2 – Unambiguous retry policy:
          Attempt 0   : generate → clip if needed → append addendum → always retry.
          Attempts 1+ : generate → clip if needed → accept (warn) or return clean.
          All retries exhausted → safe uniform fallback.
        """
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                reasoning = await self.get_llm("default", "llm").invoke(prompt)
                logger.info(
                    "Numeric reasoning for %s (attempt %d): %s",
                    question.page_url, attempt + 1, reasoning,
                )

                parsing_instructions = clean_indents(
                    f"""
                    The text is a forecast distribution for a numeric question.
                    Question: "{question.question_text}"
                    Units: {question.unit_of_measure}
                    Example bounds: {question.lower_bound} – {question.upper_bound} {question.unit_of_measure}
                    - Parse values in the correct units (convert if needed).
                    - No scientific notation.
                    - NEVER return values outside [{question.lower_bound}, {question.upper_bound}].
                    - If percentiles are not explicitly given, indicate that instead of guessing.
                    """
                )

                percentile_list: List[Percentile] = await structure_output(
                    reasoning,
                    list[Percentile],
                    model=self.get_llm("parser", "llm"),
                    additional_instructions=parsing_instructions,
                    num_validation_samples=self._structure_output_validation_samples,
                )

                # Improvement 4 – enforce monotonicity before clipping
                percentile_list = _sort_percentiles_monotone(percentile_list)

                clipped, was_clipped = self._clip_numeric_percentiles(
                    percentile_list, question
                )

                if was_clipped:
                    logger.warning(
                        "Numeric percentile clipping on attempt %d for %s",
                        attempt + 1, question.page_url,
                    )
                    if attempt < max_retries - 1:
                        # Improvement 8 – prepend addendum so it's near top of context
                        addendum = self._build_retry_addendum(
                            question.lower_bound,
                            question.upper_bound,
                            question.unit_of_measure or "",
                        )
                        prompt = addendum + "\n\n" + prompt
                        continue  # retry with addendum
                    # Final attempt: accept clipped result with a warning
                    logger.warning(
                        "Accepting clipped numeric result for %s after %d attempts.",
                        question.page_url, attempt + 1,
                    )

                prediction = NumericDistribution.from_question(clipped, question)
                logger.info(
                    "Numeric forecast for %s: %s",
                    question.page_url, prediction.declared_percentiles,
                )
                # Improvement 9 – JSONL run log
                _run_logger.log({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "url": question.page_url,
                    "type": "numeric",
                    "attempt": attempt + 1,
                    "percentiles": [(p.percentile, p.value) for p in clipped],
                    "reasoning_snippet": reasoning[:500],
                })
                return ReasonedPrediction(
                    prediction_value=prediction, reasoning=reasoning
                )

            except AssertionError as exc:
                last_error = exc
                logger.warning(
                    "AssertionError on numeric attempt %d for %s: %s",
                    attempt + 1, question.page_url, exc,
                )
                if attempt < max_retries - 1:
                    # Improvement 8 – prepend
                    addendum = self._build_retry_addendum(
                        question.lower_bound,
                        question.upper_bound,
                        question.unit_of_measure or "",
                    )
                    prompt = addendum + "\n\n" + prompt

        logger.error(
            "All %d numeric attempts failed for %s. Using safe fallback. Last error: %s",
            max_retries, question.page_url, last_error,
        )
        # Improvement 6 – prefer community prediction over raw uniform fallback
        community_pred = getattr(question, "community_prediction", None)
        if community_pred is not None:
            logger.info(
                "Using community prediction as fallback for %s", question.page_url
            )
            safe_pcts = _community_numeric_percentiles(
                community_pred,
                float(question.lower_bound),
                float(question.upper_bound),
            )
        else:
            safe_pcts = self._safe_fallback_percentiles(
                float(question.lower_bound), float(question.upper_bound)
            )
        prediction = NumericDistribution.from_question(safe_pcts, question)
        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning=(
                f"SAFETY FALLBACK: distribution within "
                f"[{question.lower_bound}, {question.upper_bound}] "
                f"after {max_retries} failed attempts."
            ),
        )

    @staticmethod
    def _clip_numeric_percentiles(
        percentile_list: List[Percentile],
        question: NumericQuestion,
    ) -> tuple[List[Percentile], bool]:
        clipped: List[Percentile] = []
        was_clipped = False
        lo = float(question.lower_bound)
        hi = float(question.upper_bound)
        for p in percentile_list:
            clamped = max(lo, min(hi, p.value))
            if clamped != p.value:
                was_clipped = True
            clipped.append(Percentile(percentile=p.percentile, value=clamped))
        return clipped, was_clipped

    # -------------------------------------------------------------------------
    # Date questions  (FIX 2 – same unambiguous retry logic as numeric)
    # -------------------------------------------------------------------------
    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        bound_enforcement = self._create_bound_enforcement_message(question)

        base_prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_msg}
            {upper_msg}
            {bound_enforcement}

            Formatting Instructions:
            - Express all answers as dates: YYYY-MM-DD.
            - If hours matter: YYYY-MM-DDTHH:MM:SSZ (UTC). No other formats.
            - Dates must be strictly chronological (earliest at percentile 10).
            - ALWAYS stay STRICTLY within the bounds above.

            Before answering you write:
            (a) Time left until the outcome is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) Expectations of experts and markets.
            (e) An early-outcome scenario (STILL after the lower bound).
            (f) A late-outcome scenario (STILL before the upper bound).

            {self._get_conditional_disclaimer_if_necessary(question)}
            Good forecasters set wide 90/10 confidence intervals but NEVER violate
            hard date bounds.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD  (>= lower bound)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD  (<= upper bound)
            "
            """
        )
        return await self._date_prompt_to_forecast(question, base_prompt)

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
        max_retries: int = 3,
    ) -> ReasonedPrediction[NumericDistribution]:
        last_error: Optional[Exception] = None
        lower_ts = question.lower_bound.timestamp()
        upper_ts = question.upper_bound.timestamp()

        for attempt in range(max_retries):
            try:
                reasoning = await self.get_llm("default", "llm").invoke(prompt)
                logger.info(
                    "Date reasoning for %s (attempt %d): %s",
                    question.page_url, attempt + 1, reasoning,
                )

                parsing_instructions = clean_indents(
                    f"""
                    The text is a forecast distribution for a date question.
                    Question: "{question.question_text}"
                    Bounds: {question.lower_bound.date().isoformat()} to
                            {question.upper_bound.date().isoformat()}
                    - Format each date as a valid parseable datetime string.
                    - Assume midnight UTC if no time is given.
                    - All dates MUST fall within the stated bounds.
                    - If percentiles are not explicitly given, indicate that.
                    """
                )

                date_percentile_list: List[DatePercentile] = await structure_output(
                    reasoning,
                    list[DatePercentile],
                    model=self.get_llm("parser", "llm"),
                    additional_instructions=parsing_instructions,
                    num_validation_samples=self._structure_output_validation_samples,
                )

                clipped, was_clipped = self._clip_date_percentiles(
                    date_percentile_list, lower_ts, upper_ts, question
                )

                if was_clipped:
                    logger.warning(
                        "Date percentile clipping on attempt %d for %s",
                        attempt + 1, question.page_url,
                    )
                    if attempt < max_retries - 1:
                        # Improvement 8 – prepend so reminder is near top of context
                        addendum = self._build_retry_addendum(
                            question.lower_bound.date().isoformat(),
                            question.upper_bound.date().isoformat(),
                            is_date=True,
                        )
                        prompt = addendum + "\n\n" + prompt
                        continue
                    logger.warning(
                        "Accepting clipped date result for %s after %d attempts.",
                        question.page_url, attempt + 1,
                    )

                prediction = NumericDistribution.from_question(clipped, question)
                logger.info(
                    "Date forecast for %s: %s",
                    question.page_url, prediction.declared_percentiles,
                )
                # Improvement 9 – JSONL run log
                _run_logger.log({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "url": question.page_url,
                    "type": "date",
                    "attempt": attempt + 1,
                    "percentiles": [(p.percentile, p.value) for p in clipped],
                    "reasoning_snippet": reasoning[:500],
                })
                return ReasonedPrediction(
                    prediction_value=prediction, reasoning=reasoning
                )

            except AssertionError as exc:
                last_error = exc
                logger.warning(
                    "AssertionError on date attempt %d for %s: %s",
                    attempt + 1, question.page_url, exc,
                )
                if attempt < max_retries - 1:
                    # Improvement 8 – prepend
                    addendum = self._build_retry_addendum(
                        question.lower_bound.date().isoformat(),
                        question.upper_bound.date().isoformat(),
                        is_date=True,
                    )
                    prompt = addendum + "\n\n" + prompt

        logger.error(
            "All %d date attempts failed for %s. Using safe fallback. Last error: %s",
            max_retries, question.page_url, last_error,
        )
        # Improvement 6 – prefer community prediction over raw uniform fallback
        community_pred = getattr(question, "community_prediction", None)
        if community_pred is not None:
            logger.info(
                "Using community prediction as date fallback for %s", question.page_url
            )
            safe_pcts = _community_numeric_percentiles(
                community_pred, lower_ts, upper_ts
            )
        else:
            safe_pcts = self._safe_fallback_percentiles(lower_ts, upper_ts)
        prediction = NumericDistribution.from_question(safe_pcts, question)
        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning=(
                f"SAFETY FALLBACK: date distribution within "
                f"[{question.lower_bound.date().isoformat()}, "
                f"{question.upper_bound.date().isoformat()}] "
                f"after {max_retries} failed attempts."
            ),
        )

    @staticmethod
    def _clip_date_percentiles(
        date_percentile_list: List[DatePercentile],
        lower_ts: float,
        upper_ts: float,
        question: DateQuestion,
    ) -> tuple[List[Percentile], bool]:
        clipped: List[Percentile] = []
        was_clipped = False
        for dp in date_percentile_list:
            ts = dp.value.timestamp()
            clamped = max(lower_ts, min(upper_ts, ts))
            if clamped != ts:
                was_clipped = True
                logger.warning(
                    "Clipped date p%s from %s for %s",
                    dp.percentile, dp.value.isoformat(), question.page_url,
                )
            clipped.append(Percentile(percentile=dp.percentile, value=clamped))
        return clipped, was_clipped

    # -------------------------------------------------------------------------
    # Bounds messages
    # -------------------------------------------------------------------------
    def _create_upper_and_lower_bound_messages(
        self, question: Union[NumericQuestion, DateQuestion]
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            upper = float(
                question.nominal_upper_bound
                if question.nominal_upper_bound is not None
                else question.upper_bound
            )
            lower = float(
                question.nominal_lower_bound
                if question.nominal_lower_bound is not None
                else question.lower_bound
            )
            unit = question.unit_of_measure or ""
            upper_msg = (
                f"The question creator thinks the number is likely not higher than {upper} {unit}."
                if question.open_upper_bound
                else f"The outcome cannot be higher than {upper} {unit}."
            )
            lower_msg = (
                f"The question creator thinks the number is likely not lower than {lower} {unit}."
                if question.open_lower_bound
                else f"The outcome cannot be lower than {lower} {unit}."
            )

        elif isinstance(question, DateQuestion):
            upper_msg = (
                f"The question creator thinks the date is likely not later than "
                f"{question.upper_bound.date().isoformat()}."
                if question.open_upper_bound
                else f"The outcome cannot be later than "
                f"{question.upper_bound.date().isoformat()}."
            )
            lower_msg = (
                f"The question creator thinks the date is likely not earlier than "
                f"{question.lower_bound.date().isoformat()}."
                if question.open_lower_bound
                else f"The outcome cannot be earlier than "
                f"{question.lower_bound.date().isoformat()}."
            )
        else:
            raise ValueError(f"Unsupported question type: {type(question)}")

        return upper_msg, lower_msg

    # -------------------------------------------------------------------------
    # Conditional questions
    # -------------------------------------------------------------------------
    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, full_research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )

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
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore[arg-type]
            child=child_info.prediction_value,  # type: ignore[arg-type]
            prediction_yes=yes_info.prediction_value,  # type: ignore[arg-type]
            prediction_no=no_info.prediction_value,  # type: ignore[arg-type]
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self,
        question: MetaculusQuestion,
        research: str,
        question_type: str,
    ) -> tuple[ReasonedPrediction[Union[PredictionTypes, PredictionAffirmed]], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ("parent", "child")
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            previous_forecast = previous_forecasts[-1]
            current_utc = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc
            ):
                pretty_value = DataOrganizer.get_readable_prediction(  # type: ignore[arg-type]
                    previous_forecast
                )
                prediction: ReasonedPrediction[Union[PredictionTypes, PredictionAffirmed]] = (
                    ReasonedPrediction(
                        prediction_value=PredictionAffirmed(),
                        reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                    )
                )
                return prediction, research

        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore[return-value]

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value:
            {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast but NOT the current
            forecast — it is prior information relevant to your current forecast.
            The reasoning was:
            ```
            {reasoning.reasoning}
            ```
            Do NOT use this to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        """
        FIX 10 – handles both plain-string and enum conditional_type safely.
        """
        ct = question.conditional_type
        # Support both plain str and enum (.value)
        ct_str = ct.value if hasattr(ct, "value") else str(ct)
        if ct_str not in ("yes", "no"):
            return ""
        return clean_indents(
            """
            As you are given a conditional question, forecast ONLY the **CHILD** question
            given the parent's resolution. Never re-forecast the parent. Use probabilistic
            reasoning, strongly considering the parent's resolution, to forecast the child.
            """
        )

    # -------------------------------------------------------------------------
    # IMPROVEMENT G – per-question timing wrapper
    # -------------------------------------------------------------------------
    async def _make_prediction(
        self, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction[Any]:
        t0 = time.monotonic()
        result = await super()._make_prediction(question, research)
        logger.info(
            "Prediction for %s completed in %.1fs", question.page_url, time.monotonic() - t0
        )
        return result


# ---------------------------------------------------------------------------
# IMPROVEMENT E – structured startup banner
# ---------------------------------------------------------------------------
def _log_startup_banner(mode: str, dry_run: bool) -> None:
    logger.info("=" * 60)
    logger.info("  Nike Bot  —  Just Forecast It.")
    logger.info("  Mode       : %s%s", mode, "  [DRY RUN]" if dry_run else "")
    logger.info("  Default    : %s", OPENROUTER_DEFAULT_MODEL)
    logger.info("  Parser     : %s", OPENROUTER_PARSER_MODEL)
    logger.info("  LinkUp     : %s", "configured" if LINKUP_API_KEY else "not configured")
    logger.info("  Exa        : %s", "configured" if EXA_API_KEY else "not configured")
    logger.info("  MaxDepth   : %d", MAX_COERCE_DEPTH)
    logger.info("  CalibScale : %.2f", CALIBRATION_SCALE)
    logger.info("  EarlyStop  : %.2f log-odds stdev", EARLY_STOP_TOLERANCE)
    logger.info("  RunLog     : %s", RUN_LOG_PATH if RUN_LOG_PATH else "disabled")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").propagate = False

    arg_parser = argparse.ArgumentParser(description="Run Nike Bot forecasting system")
    arg_parser.add_argument(
        "--mode",
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    # IMPROVEMENT F – dry-run: research runs but nothing is published
    arg_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run research but do not publish forecasts to Metaculus.",
    )
    args = arg_parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    dry_run: bool = args.dry_run

    _log_startup_banner(run_mode, dry_run)

    nike_bot = NikeBot(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        dry_run=dry_run,
        llms={
            "default": GeneralLlm(
                model=OPENROUTER_DEFAULT_MODEL,
                temperature=0.2,
                timeout=60,
                allowed_tries=2,
            ),
            "summarizer": GeneralLlm(
                model=OPENROUTER_DEFAULT_MODEL,
                temperature=0.2,
                timeout=60,
                allowed_tries=2,
            ),
            # Choose one researcher backend by commenting / uncommenting:
            # "researcher": "asknews/news-summaries",
            # "researcher": "smart-searcher/openai/gpt-4o-mini",
            # "researcher": "linkup+exa",
            "researcher": GeneralLlm(
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
        },
    )

    client = PatchedMetaculusClient()

    # FIX 8 – validate slug before running any forecasts
    async def _run_tournament_mode() -> List[Any]:
        slug_ok = await client.validate_tournament_slug(MARKET_PULSE_TOURNAMENT_SLUG)
        if not slug_ok:
            logger.error(
                "Tournament slug '%s' is invalid — skipping market-pulse.",
                MARKET_PULSE_TOURNAMENT_SLUG,
            )

        seasonal = await nike_bot.forecast_on_tournament(
            client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
        )
        minibench = await nike_bot.forecast_on_tournament(
            client.CURRENT_MINIBENCH_ID, return_exceptions=True
        )
        market_pulse: List[Any] = (
            await nike_bot.forecast_on_tournament(
                MARKET_PULSE_TOURNAMENT_SLUG, return_exceptions=True
            )
            if slug_ok
            else []
        )
        return list(seasonal) + list(minibench) + list(market_pulse)

    # FIX 7 – test_questions fetches are properly async-gathered
    async def _run_test_mode() -> List[Any]:
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
        ]
        nike_bot.skip_previously_forecasted_questions = False
        questions = await asyncio.gather(
            *[client.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
        )
        return await nike_bot.forecast_questions(list(questions), return_exceptions=True)

    async def _run_cup_mode() -> List[Any]:
        nike_bot.skip_previously_forecasted_questions = False
        return await nike_bot.forecast_on_tournament(
            client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
        )

    if run_mode == "tournament":
        forecast_reports = asyncio.run(_run_tournament_mode())
    elif run_mode == "metaculus_cup":
        forecast_reports = asyncio.run(_run_cup_mode())
    else:
        forecast_reports = asyncio.run(_run_test_mode())

    nike_bot.log_report_summary(forecast_reports)
