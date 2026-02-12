import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
import dotenv
from typing import Any, Dict, List, Literal, Tuple, Optional, Union
from urllib.parse import urlparse

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

# -----------------------------
# Robust OpenRouter model defaults for LiteLLM
# -----------------------------
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "openrouter/openrouter/free")
OPENROUTER_PARSER_MODEL = os.getenv("OPENROUTER_PARSER_MODEL", OPENROUTER_DEFAULT_MODEL)

# Optional web search providers
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
LINKUP_ENDPOINT = os.getenv("LINKUP_ENDPOINT", "https://api.linkup.so/v1/search")
EXA_ENDPOINT = os.getenv("EXA_ENDPOINT", "https://api.exa.ai/search")
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "25"))

# Tournament slug you added
MARKET_PULSE_TOURNAMENT_SLUG = "market-pulse-26q1"

# Whitespace helper
_WS_RE = re.compile(r"\s+")

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

# More robust: match any key that looks like *upper*bound* or *lower*bound* case-insensitively
_BOUND_KEY_RE = re.compile(r"(upper|lower).*bound", re.IGNORECASE)


def _looks_like_bound_key(k: Any) -> bool:
    if not isinstance(k, str):
        return False
    return (k in _BOUND_KEYS) or bool(_BOUND_KEY_RE.search(k))


def _to_float_if_int_like(v: Any) -> Any:
    """
    Convert int -> float and numeric strings like "11" -> 11.0.
    """
    if isinstance(v, int):
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


def _coerce_int_bounds_to_float(obj: Any) -> Any:
    """
    Recursively convert int bounds -> float bounds in dict/list payloads.
    Fixes upstream assertions that upper/lower bounds are floats.
    """
    if isinstance(obj, dict):
        out: Dict[Any, Any] = {}
        for k, v in obj.items():
            if _looks_like_bound_key(k):
                out[k] = _to_float_if_int_like(v)
            else:
                out[k] = _coerce_int_bounds_to_float(v)
        return out
    if isinstance(obj, list):
        return [_coerce_int_bounds_to_float(x) for x in obj]
    return obj


# -----------------------------
# GLOBAL PATCH (FIXED): coerce bounds without breaking classmethod/staticmethod binding
# -----------------------------
from forecasting_tools.data_models.data_organizer import DataOrganizer  # noqa: E402

# IMPORTANT:
# - If the original is a @classmethod, DataOrganizer.get_question_from_post_json is a *bound* method when accessed
#   and will NOT expect you to pass cls again.
# - If you wrap it incorrectly, you'll see: "takes 2 positional arguments but 3 were given"
# So: capture the *descriptor* from __dict__, and call the underlying function correctly.
_orig_descriptor = DataOrganizer.__dict__.get("get_question_from_post_json")

if isinstance(_orig_descriptor, classmethod):
    _ORIG_KIND = "classmethod"
    _ORIG_FUNC = _orig_descriptor.__func__
elif isinstance(_orig_descriptor, staticmethod):
    _ORIG_KIND = "staticmethod"
    _ORIG_FUNC = _orig_descriptor.__func__
else:
    # could be a plain function (instance method) or already-unwrapped function
    _ORIG_KIND = "function"
    _ORIG_FUNC = _orig_descriptor  # type: ignore


def _patched_get_question_from_post_json(*args: Any, **kwargs: Any):
    """
    Coerce bounds in the post_json argument and forward to the original implementation,
    preserving the original binding semantics (classmethod/staticmethod/instance method).
    """
    args_list = list(args)

    # Coerce the first dict/list argument (post_json)
    for i, a in enumerate(args_list):
        if isinstance(a, (dict, list)):
            args_list[i] = _coerce_int_bounds_to_float(a)
            break

    if _ORIG_KIND == "classmethod":
        # args_list[0] is cls
        cls = args_list[0]
        return _ORIG_FUNC(cls, *args_list[1:], **kwargs)
    if _ORIG_KIND == "function":
        # instance method (unlikely here, but safe)
        self_obj = args_list[0]
        return _ORIG_FUNC(self_obj, *args_list[1:], **kwargs)
    # staticmethod
    return _ORIG_FUNC(*args_list, **kwargs)


# Re-assign in the same "shape" as originally declared
if _ORIG_KIND == "classmethod":
    DataOrganizer.get_question_from_post_json = classmethod(_patched_get_question_from_post_json)
elif _ORIG_KIND == "staticmethod":
    DataOrganizer.get_question_from_post_json = staticmethod(_patched_get_question_from_post_json)
else:
    DataOrganizer.get_question_from_post_json = _patched_get_question_from_post_json


class PatchedMetaculusClient(MetaculusClient):
    """
    Extra safety: coerce bounds in any post-json->question conversion path.
    Also includes tournament retrieval aliases to fit different versions.
    """

    def _post_json_to_questions_while_handling_groups(self, post_json_from_api, group_question_mode=None):
        post_json_from_api = _coerce_int_bounds_to_float(post_json_from_api)
        return super()._post_json_to_questions_while_handling_groups(
            post_json_from_api,
            group_question_mode=group_question_mode,
        )

    async def get_questions_from_tournament(self, tournament_id_or_slug: Union[str, int]):
        return await self._get_open_tournament_questions(tournament_id_or_slug)

    async def get_questions_in_tournament(self, tournament_id_or_slug: Union[str, int]):
        return await self._get_open_tournament_questions(tournament_id_or_slug)

    async def get_tournament_questions(self, tournament_id_or_slug: Union[str, int]):
        return await self._get_open_tournament_questions(tournament_id_or_slug)

    async def _get_open_tournament_questions(self, tournament_id_or_slug: Union[str, int]):
        """
        Try common upstream method names across forecasting_tools versions.
        """
        for name in (
            "get_all_open_questions_from_tournament",
            "get_open_questions_from_tournament",
        ):
            fn = getattr(super(), name, None)
            if callable(fn):
                out = fn(tournament_id_or_slug)
                return await out if asyncio.iscoroutine(out) else out
        raise AttributeError(
            "Could not find a tournament retrieval method on MetaculusClient. "
            "Tried: get_all_open_questions_from_tournament, get_open_questions_from_tournament."
        )


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    payload: Dict,
) -> Dict:
    r = await client.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


async def linkup_search(query: str, max_results: int = 8, depth: str = "deep") -> List[Dict]:
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
    async with httpx.AsyncClient() as client:
        data = await _post_json(client, LINKUP_ENDPOINT, headers, payload)
    return data.get("results", []) or []


async def exa_search(query: str, max_results: int = 8, max_age_hours: Optional[int] = None) -> List[Dict]:
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
    async with httpx.AsyncClient() as client:
        data = await _post_json(client, EXA_ENDPOINT, headers, payload)
    return data.get("results", []) or []


# -----------------------------
# Source quality ranking for Linkup/Exa snippets (optional)
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

_MED_TRUST_HINTS = ("investor", "ir.", "investors.", "press", "newsroom", "docs.", "github.com")
_LOW_TRUST_HINTS = ("pinterest.", "quora.", "medium.com", "substack.com", "blogspot.", "wordpress.", "tumblr.", "tiktok.", "facebook.", "x.com", "twitter.com")


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


class SpringTemplateBot2026(ForecastBot):
    """
    Template bot structure preserved.
    Changes embedded:
    - Robust OpenRouter model defaults (OPENROUTER_DEFAULT_MODEL)
    - Fixed DataOrganizer patch that preserves classmethod/staticmethod call semantics
    - Robust bound coercion (ints + numeric strings, regex key matching)
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)

            elif (
                researcher == "asknews/news-summaries"
                or researcher == "asknews/deep-research/low-depth"
                or researcher == "asknews/deep-research/medium-depth"
                or researcher == "asknews/deep-research/high-depth"
            ):
                research = await AskNewsSearcher().call_preconfigured_version(researcher, prompt)

            elif isinstance(researcher, str) and researcher.startswith("smart-searcher"):
                model_name = researcher[len("smart-searcher/") :] if researcher.startswith("smart-searcher/") else researcher
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)

            elif isinstance(researcher, str) and researcher == "linkup+exa":
                q = question.question_text.strip()
                criteria = (question.resolution_criteria or "").strip()
                query_resolution = f"{q}\nResolution criteria keywords:\n{criteria[:600]}"

                linkup_1, linkup_2, exa_1, exa_2 = await asyncio.gather(
                    linkup_search(q, max_results=8, depth="deep"),
                    linkup_search(query_resolution, max_results=6, depth="deep"),
                    exa_search(q, max_results=10),
                    exa_search(query_resolution, max_results=8),
                    return_exceptions=False,
                )
                combined: List[Dict] = []
                combined.extend(linkup_1 or [])
                combined.extend(linkup_2 or [])
                combined.extend(exa_1 or [])
                combined.extend(exa_2 or [])

                sources_block, urls = _rank_and_format_sources(combined, max_to_keep=14)

                summarize_prompt = clean_indents(
                    f"""
                    You are a research assistant to a superforecaster.
                    Task: produce a concise, decision-relevant briefing grounded in the retrieved sources.
                    Do NOT produce a final forecast. Do NOT invent facts.

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
                research = clean_indents(
                    f"""
                    {summary}

                    --- SOURCES (ranked URLs) ---
                    {url_list}
                    """
                ).strip()

            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)

            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
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

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self, question: BinaryQuestion, prompt: str
    ) -> ReasonedPrediction[float]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
        logger.info(f"Forecasted URL {question.page_url} with prediction: {decimal_pred}.")
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

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
            (c) A description of an scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
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

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        logger.info(f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}.")
        return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self, question: NumericQuestion, prompt: str
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}.")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
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

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        return await self._date_prompt_to_forecast(question, prompt)

    async def _date_prompt_to_forecast(
        self, question: DateQuestion, prompt: str
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(percentile=percentile.percentile, value=percentile.value.timestamp())
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}.")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: Union[NumericQuestion, DateQuestion]
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            upper_bound_number = float(question.nominal_upper_bound) if question.nominal_upper_bound is not None else float(question.upper_bound)
            lower_bound_number = float(question.nominal_lower_bound) if question.nominal_lower_bound is not None else float(question.lower_bound)
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(question.parent, research, "parent")
        child_info, full_research = await self._get_question_prediction_info(question.child, full_research, "child")
        yes_info, full_research = await self._get_question_prediction_info(question.question_yes, full_research, "yes")
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
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(reasoning=full_reasoning, prediction_value=full_prediction)

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
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
                return (prediction, research)  # type: ignore

        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(self, question: MetaculusQuestion) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run the TemplateBot forecasting system")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in ["tournament", "metaculus_cup", "test_questions"], "Invalid run mode"

    template_bot = SpringTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
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
            # choose one:
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

    # IMPORTANT: patched client to avoid bound assertion failures
    client = PatchedMetaculusClient()

    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True)
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True)
        )
        market_pulse_reports = asyncio.run(
            template_bot.forecast_on_tournament(MARKET_PULSE_TOURNAMENT_SLUG, return_exceptions=True)
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports + market_pulse_reports

    elif run_mode == "metaculus_cup":
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True)
        )

    else:
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [client.get_question_by_url(question_url) for question_url in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(template_bot.forecast_questions(questions, return_exceptions=True))

    template_bot.log_report_summary(forecast_reports)
