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

# Vultr serverless inference config.
# Override via env to swap in any other Vultr-compatible model without code changes.
VULTR_API_KEY = os.getenv(
    "VULTR_SERVERLESS_INFERENCE_API_KEY",
    os.getenv("VULTR_API_KEY", ""),
)
VULTR_API_URL = os.getenv(
    "VULTR_API_URL", "https://api.vultr.com/v2/ai/inference"
)
VULTR_DEFAULT_MODEL = os.getenv("VULTR_DEFAULT_MODEL", "buoyant-3.5")
VULTR_SUMMARIZER_MODEL = os.getenv("VULTR_SUMMARIZER_MODEL", VULTR_DEFAULT_MODEL)
VULTR_PARSER_MODEL = os.getenv("VULTR_PARSER_MODEL", VULTR_DEFAULT_MODEL)
VULTR_MAX_OUTPUT_TOKENS = int(os.getenv("VULTR_MAX_OUTPUT_TOKENS", "1024"))

# ── Research API keys (all optional — missing keys = that source silently skipped)
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID", "")
ASKNEWS_CLIENT_SECRET = os.getenv("ASKNEWS_CLIENT_SECRET", "")

LINKUP_ENDPOINT = os.getenv("LINKUP_ENDPOINT", "https://api.linkup.so/v1/search")
EXA_ENDPOINT = os.getenv("EXA_ENDPOINT", "https://api.exa.ai/search")
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "25"))


class VultrLlm:
    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        timeout: float = 60.0,
        allowed_tries: int = 1,
        max_output_tokens: int = VULTR_MAX_OUTPUT_TOKENS,
        api_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.allowed_tries = allowed_tries
        self.max_output_tokens = max_output_tokens
        self.api_url = api_url or VULTR_API_URL

    async def invoke(self, prompt: str) -> str:
        if not VULTR_API_KEY:
            raise ValueError(
                "VULTR_SERVERLESS_INFERENCE_API_KEY is required for Vultr serverless inference."
            )

        headers = {
            "Authorization": f"Bearer {VULTR_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": prompt,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            last_exc: Optional[Exception] = None
            for attempt in range(max(1, self.allowed_tries)):
                try:
                    response = await client.post(
                        self.api_url,
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt + 1 >= self.allowed_tries:
                        raise
                    await asyncio.sleep(1.0)
            else:
                raise last_exc  # type: ignore

        def _extract_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for key in ("output", "result", "text", "content"):
                    if key in value:
                        extracted = _extract_text(value[key])
                        if extracted:
                            return extracted
                return json.dumps(value)
            if isinstance(value, list):
                return " ".join(_extract_text(item) for item in value if item is not None)
            return str(value)

        output = _extract_text(data.get("output") or data.get("result") or data.get("choices") or data.get("response") or data.get("text"))
        if not output:
            raise ValueError(
                "Vultr inference response did not include usable text output."
            )

        return output.strip()

MAX_COERCE_DEPTH = int(os.getenv("MAX_COERCE_DEPTH", "30"))

AI_TOURNAMENT_ID = "33022"
MARKET_PULSE_TOURNAMENT_SLUG = "market-pulse-26q2"
SPRING_2026_AI_BENCHMARKING_SLUG = "spring-aib-2026"

_FALLBACK_FRACS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
_FALLBACK_PERCENTILES = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)

# No default routing toward 50%; forecasts should follow evidence.
CALIBRATION_SCALE: float = float(os.getenv("CALIBRATION_SCALE", "1.00"))

# Mild aggregation extremization after evidence-based forecasts.
EXTREMIZE_SCALE: float = float(os.getenv("EXTREMIZE_SCALE", "1.15"))

EARLY_STOP_TOLERANCE: float = float(os.getenv("EARLY_STOP_TOLERANCE", "0.15"))

# Extremizing low forecasts – if prediction is <= 35%, push to 8%
LOW_FORECAST_THRESHOLD: float = float(os.getenv("LOW_FORECAST_THRESHOLD", "0.35"))
LOW_FORECAST_FLOOR: float = float(os.getenv("LOW_FORECAST_FLOOR", "0.08"))

# Minibench extremization
MINIBENCH_EXTREMIZE_HIGH_CEILING: float = float(os.getenv("MINIBENCH_EXTREMIZE_HIGH_CEILING", "0.51"))
MINIBENCH_EXTREMIZE_HIGH_ROOF: float = float(os.getenv("MINIBENCH_EXTREMIZE_HIGH_ROOF", "0.99"))
MINIBENCH_EXTREMIZE_LOW_THRESHOLD: float = float(os.getenv("MINIBENCH_EXTREMIZE_LOW_THRESHOLD", "0.49"))
MINIBENCH_EXTREMIZE_LOW_FLOOR: float = float(os.getenv("MINIBENCH_EXTREMIZE_LOW_FLOOR", "0.01"))

# Spring contest extremization — more conservative to avoid overconfidence
SPRING_CONTEST_MIN_CONFIDENCE: float = float(os.getenv("SPRING_CONTEST_MIN_CONFIDENCE", "0.70"))
SPRING_EXTREMIZE_HIGH_CEILING: float = float(os.getenv("SPRING_EXTREMIZE_HIGH_CEILING", "0.60"))
SPRING_EXTREMIZE_HIGH_ROOF: float = float(os.getenv("SPRING_EXTREMIZE_HIGH_ROOF", "0.95"))
SPRING_EXTREMIZE_LOW_THRESHOLD: float = float(os.getenv("SPRING_EXTREMIZE_LOW_THRESHOLD", "0.40"))
SPRING_EXTREMIZE_LOW_FLOOR: float = float(os.getenv("SPRING_EXTREMIZE_LOW_FLOOR", "0.05"))

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
# HARD PATCH 1: NumericQuestion bound coercion
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
# ---------------------------------------------------------------------------
class PatchedMetaculusClient(MetaculusClient):
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
                return await result if inspect.isawaitable(result) else result
        raise AttributeError(
            "Could not find a tournament retrieval method on MetaculusClient."
        )

    async def validate_tournament_slug(self, slug: str) -> bool:
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
    """Query Linkup. Returns [] silently if API key is missing or call fails."""
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
    try:
        async with httpx.AsyncClient() as client:
            data = await _post_json_http(client, LINKUP_ENDPOINT, headers, payload)
        return data.get("results", []) or []
    except Exception as exc:
        logger.warning("Linkup search failed (query=%r): %s", query[:60], exc)
        return []


async def exa_search(
    query: str,
    max_results: int = 8,
    max_age_hours: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Query Exa. Returns [] silently if API key is missing or call fails."""
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
    try:
        async with httpx.AsyncClient() as client:
            data = await _post_json_http(client, EXA_ENDPOINT, headers, payload)
        return data.get("results", []) or []
    except Exception as exc:
        logger.warning("Exa search failed (query=%r): %s", query[:60], exc)
        return []


async def asknews_search(
    query: str,
    mode: str = "asknews/news-summaries",
) -> str:
    """
    Query AskNews. Returns '' silently if credentials are missing or call fails.

    mode can be one of:
      - "asknews/news-summaries"
      - "asknews/deep-research/low-depth"
      - "asknews/deep-research/medium-depth"
      - "asknews/deep-research/high-depth"
    """
    if not (ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET):
        return ""
    try:
        result = await AskNewsSearcher().call_preconfigured_version(mode, query)
        return result or ""
    except Exception as exc:
        logger.warning("AskNews search failed (mode=%s): %s", mode, exc)
        return ""


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
# In-run question cache
# ---------------------------------------------------------------------------
class QuestionCache:
    def __init__(self) -> None:
        self._cache: Dict[str, MetaculusQuestion] = {}

    def get(self, url: str) -> Optional[MetaculusQuestion]:
        return self._cache.get(url)

    def set(self, url: str, q: MetaculusQuestion) -> None:
        self._cache[url] = q

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _to_log_odds(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1.0 - p))


def _from_log_odds(lo: float) -> float:
    return 1.0 / (1.0 + math.exp(-lo))


def _aggregate_binary_predictions(probs: List[float]) -> float:
    """
    Log-odds mean, then optional calibration scale, then extremization.
    CALIBRATION_SCALE=1.0 (default) means no regression toward 0.5.
    EXTREMIZE_SCALE>1.0 pushes the aggregate further from 0.5 after merging.
    """
    if not probs:
        return 0.5
    mean_lo = statistics.mean(_to_log_odds(p) for p in probs)
    calibrated_lo = mean_lo * CALIBRATION_SCALE
    extremized_lo = calibrated_lo * EXTREMIZE_SCALE
    extremized = _from_log_odds(extremized_lo)
    return max(0.01, min(0.99, extremized))


def _trimmed_mean(values: List[float]) -> float:
    if len(values) < 4:
        return statistics.mean(values)
    trimmed = sorted(values)[1:-1]
    return statistics.mean(trimmed)


# ---------------------------------------------------------------------------
# Monotone percentile sort
# ---------------------------------------------------------------------------

def _sort_percentiles_monotone(percentile_list: List[Percentile]) -> List[Percentile]:
    if not percentile_list:
        return percentile_list
    ordered = sorted(percentile_list, key=lambda p: p.percentile)
    for i in range(1, len(ordered)):
        if ordered[i].value < ordered[i - 1].value:
            ordered[i] = Percentile(
                percentile=ordered[i].percentile, value=ordered[i - 1].value
            )
    return ordered


# ---------------------------------------------------------------------------
# Community-prediction fallback helper
# ---------------------------------------------------------------------------

def _community_numeric_percentiles(
    community_pred: Any, lo: float, hi: float
) -> List[Percentile]:
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
# JSONL run logger
# ---------------------------------------------------------------------------

class RunLogger:
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
# Reasoning compressor — trims verbose LLM output for Metaculus comments
# ---------------------------------------------------------------------------

async def _compress_reasoning(
    llm: GeneralLlm,
    full_reasoning: str,
    question_text: str,
    final_prediction_str: str,
) -> str:
    prompt = clean_indents(
        f"""
        You are editing a forecaster's reasoning note for a public comment.

        Question: {question_text}
        Final prediction: {final_prediction_str}

        Full reasoning:
        {full_reasoning[:3000]}

        Write exactly 2-3 sentences in first person that:
        1. State the strongest evidence that supports the forecast.
        2. Briefly name the main remaining risk.
        3. State the conclusion and confidence clearly.

        Rules:
        - Use first person (I / I'm) and be direct.
        - Do not describe the research process, tools, or strategy.
        - No model names, no tool names, no "my research assistant says".
        - No hedging phrases like "it's hard to say" or "I could be wrong".
        - No bullet points, headers, or markdown.
        - Start with evidence or the conclusion, not with "The question asks...".
        """
    )
    try:
        compressed = await llm.invoke(prompt)
        lines = [l.strip() for l in compressed.strip().splitlines() if l.strip()]
        return " ".join(lines[:5])
    except Exception as exc:
        logger.warning("Reasoning compression failed: %s", exc)
        return full_reasoning.strip()[:300]


# ---------------------------------------------------------------------------
# Multi-source research engine
# ---------------------------------------------------------------------------

async def _multi_source_research(
    question: MetaculusQuestion,
    summarizer_llm: GeneralLlm,
) -> str:
    """
    Fan out to Exa, Linkup, and AskNews in parallel. Each source is optional —
    missing API keys cause silent skips. All available results are merged,
    ranked, and summarised. The final block clearly labels which sources
    contributed so the downstream LLM knows how fresh the data is.

    This is intentionally additive: callers can layer additional context on top.
    """
    q = question.question_text.strip()
    criteria = (question.resolution_criteria or "").strip()
    query_resolution = f"{q}\nResolution criteria:\n{criteria[:600]}"
    query_criteria_only = criteria[:700] if criteria else q

    today = datetime.now().strftime("%Y-%m-%d")

    # ── Fan out all sources simultaneously ──────────────────────────────────
    (
        exa_1, exa_2, exa_3,
        lk_1, lk_2,
        asknews_result,
    ) = await asyncio.gather(
        exa_search(q, max_results=10),
        exa_search(query_resolution, max_results=8),
        exa_search(query_criteria_only, max_results=6),
        linkup_search(q, max_results=8, depth="deep"),
        linkup_search(query_resolution, max_results=6, depth="deep"),
        asknews_search(q, mode="asknews/news-summaries"),
        return_exceptions=True,
    )

    # Normalise exceptions to empty
    def _safe_list(v: Any) -> List[Dict[str, Any]]:
        return v if isinstance(v, list) else []

    def _safe_str(v: Any) -> str:
        return v if isinstance(v, str) else ""

    structured_results: List[Dict[str, Any]] = [
        *_safe_list(exa_1),
        *_safe_list(exa_2),
        *_safe_list(exa_3),
        *_safe_list(lk_1),
        *_safe_list(lk_2),
    ]
    asknews_text: str = _safe_str(asknews_result)

    sources_active: List[str] = []
    if any(_safe_list(r) for r in (exa_1, exa_2, exa_3)):
        sources_active.append("Exa")
    if any(_safe_list(r) for r in (lk_1, lk_2)):
        sources_active.append("Linkup")
    if asknews_text:
        sources_active.append("AskNews")

    sources_label = (
        ", ".join(sources_active) if sources_active else "none (no API keys configured)"
    )
    logger.info(
        "Multi-source research for %s — active sources: %s",
        question.page_url, sources_label,
    )

    # ── Rank and format structured snippets ─────────────────────────────────
    sources_block, urls = _rank_and_format_sources(structured_results, max_to_keep=14)
    url_list = "\n".join(urls[:30]) if urls else "(none)"

    # ── Build AskNews section ────────────────────────────────────────────────
    asknews_section = ""
    if asknews_text:
        asknews_section = clean_indents(
            f"""
            === ASKNEWS NEWS SUMMARY (fetched {today}) ===
            {asknews_text[:3000]}
            """
        )

    # ── Synthesise with summariser LLM ──────────────────────────────────────
    synthesise_prompt = clean_indents(
        f"""
        You are a research assistant to a professional superforecaster.
        Today's date: {today}.

        Your job: produce a concise, decision-relevant briefing that helps the
        forecaster make a well-calibrated probability estimate. Do NOT produce
        a final forecast yourself. Do NOT invent facts. Only state what the
        retrieved evidence says.

        Question:
        {q}

        Resolution criteria:
        {criteria}

        === STRUCTURED WEB SNIPPETS (ranked by source credibility) ===
        {sources_block}

        {asknews_section}

        Output format (use these exact headers):
        ## Key facts
        (4-6 bullets of the most decision-relevant facts from the sources above)

        ## YES scenario
        (What evidence / conditions would lead to a YES resolution)

        ## NO scenario
        (What evidence / conditions would lead to a NO resolution)

        ## Timeline signal
        (Most important near-term event or data point that would shift the answer)

        ## Recency note
        (State the date range of the evidence you found and flag any gaps)

        ## Sources
        {url_list}
        """
    )

    try:
        summary = await summarizer_llm.invoke(synthesise_prompt)
    except Exception as exc:
        logger.warning("Synthesis LLM failed for %s: %s", question.page_url, exc)
        summary = (
            f"(Synthesis failed: {exc})\n\n"
            f"Raw snippets:\n{sources_block}\n\n{asknews_section}"
        )

    return clean_indents(
        f"""
        {summary}

        ---
        RESEARCH SOURCES USED: {sources_label}
        RETRIEVED: {today}
        ---
        """
    ).strip()


# ---------------------------------------------------------------------------
# NikeBot
# ---------------------------------------------------------------------------
class NikeBot(ForecastBot):
    """
    Nike Bot — Evidence-first forecast mode.

    Research strategy (all sources additive and optional via API keys):
      • Exa         — semantic web search, high recall
      • Linkup      — deep web crawl with highlighted snippets
      • AskNews     — curated news summaries with freshness scores
    All three run in parallel; missing keys are silently skipped so the bot
    always runs even with zero third-party keys (falls back to LLM priors only,
    which is the worst case and clearly flagged in the research block).

    Forecast model: Vultr serverless inference model by default.
    Every forecast prompt includes the full research block so the model's
    knowledge cutoff cannot silently contaminate the answer — the research
    always carries a retrieval date.
    """

    _max_concurrent_questions: int = 1

    def __init__(self, *args: Any, dry_run: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._concurrency_limiter = asyncio.Semaphore(self._max_concurrent_questions)
        self._structure_output_validation_samples = 2
        self.dry_run = dry_run
        self._question_cache = QuestionCache()
        self._binary_preds_this_question: List[float] = []

    # -------------------------------------------------------------------------
    # Retry addendum builder
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
    # Safe fallback percentile helper
    # -------------------------------------------------------------------------
    @staticmethod
    def _safe_fallback_percentiles(lo: float, hi: float) -> List[Percentile]:
        return [
            Percentile(percentile=pct, value=lo + (hi - lo) * frac)
            for pct, frac in zip(_FALLBACK_PERCENTILES, _FALLBACK_FRACS)
        ]

    # -------------------------------------------------------------------------
    # Research
    # -------------------------------------------------------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            # ── 1. Multi-source live research (Exa + Linkup + AskNews) ──────
            live_research = await _multi_source_research(
                question,
                self.get_llm("summarizer", "llm"),
            )

            # ── 2. Optional extra research via configured "researcher" key ──
            extra_research = await self._dispatch_extra_research(question)

            # ── 3. Merge ────────────────────────────────────────────────────
            if extra_research.strip():
                research = clean_indents(
                    f"""
                    {live_research}

                    === ADDITIONAL RESEARCH ===
                    {extra_research}
                    """
                ).strip()
            else:
                research = live_research

            logger.info("Research for %s:\n%s", question.page_url, research[:1200])
            _run_logger.log({
                "ts": datetime.now(timezone.utc).isoformat(),
                "url": question.page_url,
                "type": "research",
                "research_snippet": research[:800],
            })
            return research

    async def _dispatch_extra_research(self, question: MetaculusQuestion) -> str:
        """
        Runs any researcher backend registered under the "researcher" LLM key.
        Returns '' if no researcher key is configured or the call fails.
        This is additive — it supplements, not replaces, the multi-source research.
        """
        researcher = self.get_llm("researcher")
        if researcher is None:
            return ""

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of
            the most relevant news, including whether the question would resolve Yes
            or No based on current information. You do not produce forecasts yourself.

            Question:
            {question.question_text}

            Resolution criteria:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )

        try:
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

                if researcher == "linkup":
                    return await self._linkup_research(question)

                if researcher == "exa":
                    return await self._exa_research(question)

                if researcher == "linkup+exa":
                    return await self._linkup_exa_research(question)

                if researcher in ("", "None", "no_research"):
                    return ""

        except Exception as exc:
            logger.warning("Extra researcher failed for %s: %s", question.page_url, exc)

        return ""

    async def _linkup_research(self, question: MetaculusQuestion) -> str:
        q = question.question_text.strip()
        criteria = (question.resolution_criteria or "").strip()
        query_resolution = f"{q}\nResolution criteria keywords:\n{criteria[:600]}"

        linkup_1, linkup_2 = await asyncio.gather(
            linkup_search(q, max_results=8, depth="deep"),
            linkup_search(query_resolution, max_results=6, depth="deep"),
        )
        combined: List[Dict[str, Any]] = [*(linkup_1 or []), *(linkup_2 or [])]
        sources_block, urls = _rank_and_format_sources(combined, max_to_keep=10)

        summarize_prompt = clean_indents(
            f"""
            You are a research assistant to a superforecaster.
            Task: produce a concise, decision-relevant briefing grounded in the retrieved
            sources. Do NOT produce a final forecast. Do NOT invent facts.

            Question: {q}
            Resolution criteria: {criteria}

            Retrieved web snippets (ranked; each includes a URL):
            {sources_block}

            Output format:
            1) Key facts (4-5 bullets max)
            2) What would make this resolve YES vs NO (brief)
            3) Timeline / what's likely before resolution (brief)
            4) Source list (just the URLs, one per line)
            """
        )
        summary = await self.get_llm("summarizer", "llm").invoke(summarize_prompt)
        url_list = "\n".join(urls[:20]) if urls else ""
        return clean_indents(
            f"""
            {summary}

            --- LINKUP SOURCES (ranked URLs) ---
            {url_list}
            """
        ).strip()

    async def _exa_research(self, question: MetaculusQuestion) -> str:
        q = question.question_text.strip()
        criteria = (question.resolution_criteria or "").strip()
        query_resolution = f"{q}\nResolution criteria keywords:\n{criteria[:600]}"
        query_criteria_only = criteria[:700] if criteria else q

        exa_1, exa_2, exa_3 = await asyncio.gather(
            exa_search(q, max_results=10),
            exa_search(query_resolution, max_results=8),
            exa_search(query_criteria_only, max_results=6),
        )
        combined: List[Dict[str, Any]] = [*(exa_1 or []), *(exa_2 or []), *(exa_3 or [])]
        sources_block, urls = _rank_and_format_sources(combined, max_to_keep=10)

        summarize_prompt = clean_indents(
            f"""
            You are a research assistant to a superforecaster.
            Task: produce a concise, decision-relevant briefing grounded in the retrieved
            sources. Do NOT produce a final forecast. Do NOT invent facts.

            Question: {q}
            Resolution criteria: {criteria}

            Retrieved web snippets (ranked; each includes a URL):
            {sources_block}

            Output format:
            1) Key facts (4-5 bullets max)
            2) What would make this resolve YES vs NO (brief)
            3) Timeline / what's likely before resolution (brief)
            4) Source list (just the URLs, one per line)
            """
        )
        summary = await self.get_llm("summarizer", "llm").invoke(summarize_prompt)
        url_list = "\n".join(urls[:20]) if urls else ""
        return clean_indents(
            f"""
            {summary}

            --- EXA SOURCES (ranked URLs) ---
            {url_list}
            """
        ).strip()

    async def _linkup_exa_research(self, question: MetaculusQuestion) -> str:
        q = question.question_text.strip()
        criteria = (question.resolution_criteria or "").strip()
        query_resolution = f"{q}\nResolution criteria keywords:\n{criteria[:600]}"
        query_criteria_only = criteria[:700] if criteria else q

        linkup_1, linkup_2, exa_1, exa_2, exa_3 = await asyncio.gather(
            linkup_search(q, max_results=8, depth="deep"),
            linkup_search(query_resolution, max_results=6, depth="deep"),
            exa_search(q, max_results=10),
            exa_search(query_resolution, max_results=8),
            exa_search(query_criteria_only, max_results=6),
        )
        combined: List[Dict[str, Any]] = [
            *(linkup_1 or []), *(linkup_2 or []),
            *(exa_1 or []), *(exa_2 or []), *(exa_3 or []),
        ]
        sources_block, urls = _rank_and_format_sources(combined, max_to_keep=14)

        summarize_prompt = clean_indents(
            f"""
            You are a research assistant to a superforecaster.
            Task: produce a concise, decision-relevant briefing grounded in the retrieved
            sources. Do NOT produce a final forecast. Do NOT invent facts.

            Question: {q}
            Resolution criteria: {criteria}

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
        self._binary_preds_this_question = []
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = clean_indents(
            f"""
            You are a professional forecaster with a strong track record.
            Your job is to produce a well-calibrated but decisive probability estimate.

            ⚠️  IMPORTANT — KNOWLEDGE CUTOFF WARNING ⚠️
            Your training data has a cutoff. The research block below was retrieved
            on {today} from live web sources. You MUST rely on the research block
            for any recent facts, not on your training memory. Where the research
            contradicts your training data, the research is more recent and takes
            precedence.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria (not yet satisfied):
            {question.resolution_criteria}

            {question.fine_print}

            ── LIVE RESEARCH (retrieved {today}) ──────────────────────────────
            {research}
            ────────────────────────────────────────────────────────────────────

            Today is {today}.

            Before stating your probability, write briefly:
            (a) Time remaining until resolution.
            (b) The status quo outcome if nothing changes.
            (c) The strongest evidence from the RESEARCH pointing toward YES.
            (d) The strongest evidence from the RESEARCH pointing toward NO.
            (e) Your overall read: which way does the evidence lean, and how strongly?

            IMPORTANT INSTRUCTIONS:
            - Forecast based on the LIVE RESEARCH above, not on your training priors.
            - Do not bias the forecast toward 50% unless the evidence is genuinely balanced.
            - If you assess >75% likelihood, say so. If <25%, say so.
            - Good forecasters put extra weight on the status quo, but follow strong
              evidence when it exists.
            - Avoid artificial centering; 50% should only appear in genuinely ambiguous cases.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self, question: BinaryQuestion, prompt: str
    ) -> ReasonedPrediction[float]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info("Reasoning for %s: %s", question.page_url, reasoning)
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        raw_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
        calibrated = 0.5 + (raw_pred - 0.5) * CALIBRATION_SCALE
        decimal_pred = max(0.01, min(0.99, calibrated))

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

        compressed = await _compress_reasoning(
            self.get_llm("default", "llm"),
            reasoning,
            question.question_text,
            f"{decimal_pred*100:.1f}%",
        )

        logger.info(
            "Forecast for %s: %.4f (from raw %.4f)", question.page_url, decimal_pred, raw_pred
        )
        _run_logger.log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "url": question.page_url,
            "type": "binary",
            "run_index": n,
            "raw_pred": raw_pred,
            "calibrated_pred": decimal_pred,
            "reasoning_snippet": reasoning[:500],
            "compressed_reasoning": compressed,
        })

        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=compressed)

    # -------------------------------------------------------------------------
    # Multiple-choice questions
    # -------------------------------------------------------------------------
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = clean_indents(
            f"""
            You are a professional forecaster with a strong track record.

            ⚠️  KNOWLEDGE CUTOFF WARNING: Your training data has a cutoff. The research
            below was retrieved on {today} from live sources and takes precedence over
            your training memory for any recent facts.

            Question:
            {question.question_text}

            Options: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            ── LIVE RESEARCH (retrieved {today}) ──────────────────────────────
            {research}
            ────────────────────────────────────────────────────────────────────

            Today is {today}.

            Before stating probabilities, write briefly:
            (a) Time remaining until resolution.
            (b) The status quo outcome if nothing changes.
            (c) One scenario that would produce a surprising outcome.

            IMPORTANT INSTRUCTIONS:
            - If evidence strongly favours one option, reflect that in the probabilities.
            - Avoid artificially spreading mass across all options when evidence is clear.
            - Keep some residual probability on each option for genuine uncertainty.
            - Rely on the LIVE RESEARCH above, not on training priors.
            {self._get_conditional_disclaimer_if_necessary(question)}

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
        compressed = await _compress_reasoning(
            self.get_llm("default", "llm"),
            reasoning,
            question.question_text,
            str(predicted_option_list),
        )
        logger.info("Forecast for %s: %s", question.page_url, predicted_option_list)
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=compressed
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
    # -------------------------------------------------------------------------
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        bound_enforcement = self._create_bound_enforcement_message(question)
        today = datetime.now().strftime("%Y-%m-%d")

        base_prompt = clean_indents(
            f"""
            You are a professional forecaster with a strong track record.

            ⚠️  KNOWLEDGE CUTOFF WARNING: Your training data has a cutoff. The research
            below was retrieved on {today} from live sources and takes precedence over
            your training memory for any recent facts.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units: {question.unit_of_measure or "Not stated (please infer this)"}

            ── LIVE RESEARCH (retrieved {today}) ──────────────────────────────
            {research}
            ────────────────────────────────────────────────────────────────────

            Today is {today}.

            {lower_msg}
            {upper_msg}
            {bound_enforcement}

            Formatting Instructions:
            - Give your answer in the stated units.
            - Never use scientific notation.
            - Percentile values must be strictly increasing (10 < 20 < 40 < 60 < 80 < 90).
            - ALWAYS ensure values stay STRICTLY within the bounds above.

            Before stating percentiles, write briefly:
            (a) Time remaining.
            (b) The outcome if nothing changes.
            (c) The outcome if the current trend continues.
            (d) Expert / market expectations per the LIVE RESEARCH.
            (e) A plausible low outcome (still above lower bound).
            (f) A plausible high outcome (still below upper bound).

            IMPORTANT: If research clearly points to a specific range, your percentiles
            should reflect that. Rely on the LIVE RESEARCH above, not on training priors.

            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
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
                    Bounds: {question.lower_bound} – {question.upper_bound} {question.unit_of_measure}
                    - Parse values in the correct units.
                    - No scientific notation.
                    - NEVER return values outside [{question.lower_bound}, {question.upper_bound}].
                    """
                )

                percentile_list: List[Percentile] = await structure_output(
                    reasoning,
                    list[Percentile],
                    model=self.get_llm("parser", "llm"),
                    additional_instructions=parsing_instructions,
                    num_validation_samples=self._structure_output_validation_samples,
                )

                percentile_list = _sort_percentiles_monotone(percentile_list)
                clipped, was_clipped = self._clip_numeric_percentiles(
                    percentile_list, question
                )

                if was_clipped:
                    logger.warning(
                        "Numeric clipping on attempt %d for %s", attempt + 1, question.page_url
                    )
                    if attempt < max_retries - 1:
                        addendum = self._build_retry_addendum(
                            question.lower_bound,
                            question.upper_bound,
                            question.unit_of_measure or "",
                        )
                        prompt = addendum + "\n\n" + prompt
                        continue
                    logger.warning(
                        "Accepting clipped numeric result for %s after %d attempts.",
                        question.page_url, attempt + 1,
                    )

                prediction = NumericDistribution.from_question(clipped, question)
                compressed = await _compress_reasoning(
                    self.get_llm("default", "llm"),
                    reasoning,
                    question.question_text,
                    f"P10={clipped[0].value if clipped else '?'}, P90={clipped[-1].value if clipped else '?'}",
                )
                logger.info("Numeric forecast for %s: %s", question.page_url, prediction.declared_percentiles)
                _run_logger.log({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "url": question.page_url,
                    "type": "numeric",
                    "attempt": attempt + 1,
                    "percentiles": [(p.percentile, p.value) for p in clipped],
                    "reasoning_snippet": reasoning[:500],
                })
                return ReasonedPrediction(
                    prediction_value=prediction, reasoning=compressed
                )

            except AssertionError as exc:
                last_error = exc
                logger.warning(
                    "AssertionError on numeric attempt %d for %s: %s",
                    attempt + 1, question.page_url, exc,
                )
                if attempt < max_retries - 1:
                    addendum = self._build_retry_addendum(
                        question.lower_bound,
                        question.upper_bound,
                        question.unit_of_measure or "",
                    )
                    prompt = addendum + "\n\n" + prompt

        logger.error(
            "All %d numeric attempts failed for %s. Last error: %s",
            max_retries, question.page_url, last_error,
        )
        community_pred = getattr(question, "community_prediction", None)
        if community_pred is not None:
            safe_pcts = _community_numeric_percentiles(
                community_pred, float(question.lower_bound), float(question.upper_bound)
            )
        else:
            safe_pcts = self._safe_fallback_percentiles(
                float(question.lower_bound), float(question.upper_bound)
            )
        prediction = NumericDistribution.from_question(safe_pcts, question)
        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning=f"Uniform fallback within [{question.lower_bound}, {question.upper_bound}] after {max_retries} failed attempts.",
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
    # Date questions
    # -------------------------------------------------------------------------
    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        bound_enforcement = self._create_bound_enforcement_message(question)
        today = datetime.now().strftime("%Y-%m-%d")

        base_prompt = clean_indents(
            f"""
            You are a professional forecaster with a strong track record.

            ⚠️  KNOWLEDGE CUTOFF WARNING: Your training data has a cutoff. The research
            below was retrieved on {today} from live sources and takes precedence over
            your training memory for any recent facts.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            ── LIVE RESEARCH (retrieved {today}) ──────────────────────────────
            {research}
            ────────────────────────────────────────────────────────────────────

            Today is {today}.

            {lower_msg}
            {upper_msg}
            {bound_enforcement}

            Formatting Instructions:
            - All answers as dates: YYYY-MM-DD.
            - Dates must be strictly chronological (earliest at percentile 10).
            - ALWAYS stay STRICTLY within the bounds above.

            Before stating percentiles, write briefly:
            (a) Time remaining.
            (b) The outcome if nothing changes.
            (c) The outcome if the current trend continues.
            (d) Expert / market expectations per the LIVE RESEARCH.
            (e) A plausible early outcome (still after lower bound).
            (f) A plausible late outcome (still before upper bound).

            IMPORTANT: If research clearly narrows the likely date, your percentiles
            should reflect that. Rely on the LIVE RESEARCH above, not on training priors.

            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD
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
                        "Date clipping on attempt %d for %s", attempt + 1, question.page_url
                    )
                    if attempt < max_retries - 1:
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
                compressed = await _compress_reasoning(
                    self.get_llm("default", "llm"),
                    reasoning,
                    question.question_text,
                    f"P10={clipped[0].value if clipped else '?'}, P90={clipped[-1].value if clipped else '?'}",
                )
                logger.info("Date forecast for %s: %s", question.page_url, prediction.declared_percentiles)
                _run_logger.log({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "url": question.page_url,
                    "type": "date",
                    "attempt": attempt + 1,
                    "percentiles": [(p.percentile, p.value) for p in clipped],
                    "reasoning_snippet": reasoning[:500],
                })
                return ReasonedPrediction(
                    prediction_value=prediction, reasoning=compressed
                )

            except AssertionError as exc:
                last_error = exc
                logger.warning(
                    "AssertionError on date attempt %d for %s: %s",
                    attempt + 1, question.page_url, exc,
                )
                if attempt < max_retries - 1:
                    addendum = self._build_retry_addendum(
                        question.lower_bound.date().isoformat(),
                        question.upper_bound.date().isoformat(),
                        is_date=True,
                    )
                    prompt = addendum + "\n\n" + prompt

        logger.error(
            "All %d date attempts failed for %s. Last error: %s",
            max_retries, question.page_url, last_error,
        )
        community_pred = getattr(question, "community_prediction", None)
        if community_pred is not None:
            safe_pcts = _community_numeric_percentiles(community_pred, lower_ts, upper_ts)
        else:
            safe_pcts = self._safe_fallback_percentiles(lower_ts, upper_ts)
        prediction = NumericDistribution.from_question(safe_pcts, question)
        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning=f"Uniform fallback within date bounds after {max_retries} failed attempts.",
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
        ct = question.conditional_type
        ct_str = ct.value if hasattr(ct, "value") else str(ct)
        if ct_str not in ("yes", "no"):
            return ""
        return clean_indents(
            """
            As you are given a conditional question, forecast ONLY the **CHILD** question
            given the parent's resolution. Never re-forecast the parent.
            """
        )

    # -------------------------------------------------------------------------
    # Per-question timing wrapper
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
# Startup banner
# ---------------------------------------------------------------------------
def _log_startup_banner(mode: str, dry_run: bool) -> None:
    sources: List[str] = []
    if EXA_API_KEY:
        sources.append("Exa")
    if LINKUP_API_KEY:
        sources.append("Linkup")
    if ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET:
        sources.append("AskNews")
    sources_str = ", ".join(sources) if sources else "none (LLM priors only)"

    logger.info("=" * 60)
    logger.info("  Nike Bot  —  Just Forecast It.")
    logger.info("  Mode          : %s%s", mode, "  [DRY RUN]" if dry_run else "")
    logger.info("  Default model : %s", VULTR_DEFAULT_MODEL)
    logger.info("  Live sources  : %s", sources_str)
    logger.info("  CalibScale    : %.2f (1.0 = no regression)", CALIBRATION_SCALE)
    logger.info("  ExtremizeScale: %.2f (>1.0 = push from 0.5)", EXTREMIZE_SCALE)
    logger.info("  EarlyStop     : %.2f log-odds stdev", EARLY_STOP_TOLERANCE)
    logger.info("  RunLog        : %s", RUN_LOG_PATH if RUN_LOG_PATH else "disabled")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Minibench and Spring Contest Extremization Helpers
# ---------------------------------------------------------------------------

def _evidence_suggests_extremization(forecast: dict) -> bool:
    if not isinstance(forecast, dict) or 'explanation' not in forecast:
        return False
    explanation = forecast.get('explanation', '').lower()
    strong_evidence_keywords = [
        'strong evidence', 'highly confident', 'clear indication', 'overwhelming',
        'compelling evidence', 'definitive', 'certain', 'conclusive',
        'robust evidence', 'solid foundation', 'high confidence'
    ]
    has_keywords = any(keyword in explanation for keyword in strong_evidence_keywords)
    is_detailed = len(explanation) > 500
    return has_keywords or is_detailed


def _extremize_minibench_forecasts(forecasts: List[Any]) -> List[Any]:
    extremized = []
    for forecast in forecasts:
        if isinstance(forecast, Exception):
            extremized.append(forecast)
            continue
        try:
            if isinstance(forecast, dict):
                forecast_copy = forecast.copy()
                if "decimal_pred" in forecast_copy:
                    pred = forecast_copy["decimal_pred"]
                    if pred >= MINIBENCH_EXTREMIZE_HIGH_CEILING:
                        forecast_copy["decimal_pred"] = MINIBENCH_EXTREMIZE_HIGH_ROOF
                        logger.info("Minibench: Extremized high %.2f → %.2f", pred, MINIBENCH_EXTREMIZE_HIGH_ROOF)
                    elif pred <= MINIBENCH_EXTREMIZE_LOW_THRESHOLD:
                        forecast_copy["decimal_pred"] = MINIBENCH_EXTREMIZE_LOW_FLOOR
                        logger.info("Minibench: Extremized low %.2f → %.2f", pred, MINIBENCH_EXTREMIZE_LOW_FLOOR)
                extremized.append(forecast_copy)
            else:
                extremized.append(forecast)
        except Exception as e:
            logger.warning("Error extremizing minibench forecast: %s", e)
            extremized.append(forecast)
    return extremized


def _extremize_spring_forecasts(forecasts: List[Any]) -> List[Any]:
    extremized = []
    for forecast in forecasts:
        if isinstance(forecast, Exception):
            extremized.append(forecast)
            continue
        try:
            if isinstance(forecast, dict):
                forecast_copy = forecast.copy()
                if "decimal_pred" in forecast_copy:
                    pred = forecast_copy["decimal_pred"]
                    evidence_strong = _evidence_suggests_extremization(forecast)
                    if evidence_strong:
                        if pred >= SPRING_EXTREMIZE_HIGH_CEILING:
                            forecast_copy["decimal_pred"] = SPRING_EXTREMIZE_HIGH_ROOF
                            logger.info("Spring: Extremized high %.2f → %.2f", pred, SPRING_EXTREMIZE_HIGH_ROOF)
                        elif pred <= SPRING_EXTREMIZE_LOW_THRESHOLD:
                            forecast_copy["decimal_pred"] = SPRING_EXTREMIZE_LOW_FLOOR
                            logger.info("Spring: Extremized low %.2f → %.2f", pred, SPRING_EXTREMIZE_LOW_FLOOR)
                    else:
                        logger.info("Spring: Forecast %.2f not extremized (weak evidence)", pred)
                extremized.append(forecast_copy)
            else:
                extremized.append(forecast)
        except Exception as e:
            logger.warning("Error extremizing spring forecast: %s", e)
            extremized.append(forecast)
    return extremized


async def _conditionally_forecast_spring(client: Any, bot: Any) -> List[Any]:
    logger.info("Spring contest: Forecasting with conservative extremization...")
    try:
        spring_results = await bot.forecast_on_tournament(
            SPRING_2026_AI_BENCHMARKING_SLUG, return_exceptions=True
        )
        return list(_extremize_spring_forecasts(spring_results))
    except Exception as e:
        logger.warning("Error forecasting on spring contest: %s", e)
        return []


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
        skip_previously_forecasted_questions=False,
        extra_metadata_in_explanation=False,
        dry_run=dry_run,
        llms={
            # ── Forecast & parse ─────────────────────────────────────────────
            # Vultr serverless inference model. Override VULTR_DEFAULT_MODEL env var
            # to swap in any other Vultr-compatible model without code changes.
            "default": VultrLlm(
                model=VULTR_DEFAULT_MODEL,
                temperature=0.2,
                timeout=60,
                allowed_tries=2,
                max_output_tokens=VULTR_MAX_OUTPUT_TOKENS,
            ),
            "summarizer": VultrLlm(
                model=VULTR_SUMMARIZER_MODEL,
                temperature=0.2,
                timeout=60,
                allowed_tries=2,
                max_output_tokens=VULTR_MAX_OUTPUT_TOKENS,
            ),
            "parser": VultrLlm(
                model=VULTR_PARSER_MODEL,
                temperature=0.0,
                timeout=60,
                allowed_tries=2,
                max_output_tokens=VULTR_MAX_OUTPUT_TOKENS,
            ),
            # ── Optional extra researcher ─────────────────────────────────────
            # The multi-source engine (Exa + Linkup + AskNews) always runs first.
            # Uncomment ONE of the lines below to add an extra research pass on
            # top of the base sources. Leave all commented out if not needed.
            #
            # "researcher": "asknews/deep-research/medium-depth",
            # "researcher": "linkup+exa",
        },
    )

    client = PatchedMetaculusClient()

    async def _run_tournament_mode() -> List[Any]:
        slug_ok = await client.validate_tournament_slug(MARKET_PULSE_TOURNAMENT_SLUG)
        if not slug_ok:
            logger.error(
                "Tournament slug '%s' is invalid — skipping market-pulse.",
                MARKET_PULSE_TOURNAMENT_SLUG,
            )

        seasonal = await nike_bot.forecast_on_tournament(
            AI_TOURNAMENT_ID, return_exceptions=True
        )

        minibench_ok = (
            await client.validate_tournament_slug(client.CURRENT_MINIBENCH_ID)
            if hasattr(client, "CURRENT_MINIBENCH_ID")
               and isinstance(client.CURRENT_MINIBENCH_ID, str)
            else True
        )
        if not minibench_ok:
            logger.error("Minibench tournament not available — skipping.")
            minibench: List[Any] = []
        else:
            minibench_raw = await nike_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
            minibench = _extremize_minibench_forecasts(minibench_raw)

        market_pulse: List[Any] = (
            await nike_bot.forecast_on_tournament(
                MARKET_PULSE_TOURNAMENT_SLUG, return_exceptions=True
            )
            if slug_ok
            else []
        )
        spring_results = await _conditionally_forecast_spring(client, nike_bot)
        return list(seasonal) + list(minibench) + list(market_pulse) + spring_results

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

