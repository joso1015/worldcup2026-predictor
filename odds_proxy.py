"""
Local The Odds API proxy for correct-score bookmaker prices.

Run from this folder after creating .env with THE_ODDS_API_KEY:
    py odds_proxy.py

Then open outputs/dashboard.html. The static dashboard calls this localhost
proxy so the API key never needs to be embedded in the generated HTML.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import requests
from urllib3.exceptions import InsecureRequestWarning

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://api.the-odds-api.com/v4"
TEAM_ALIASES = {
    "Bosnia and Herzegovina": ["Bosnia & Herzegovina", "Bosnia-Herzegovina", "Bosnia"],
    "Czechia": ["Czech Republic"],
    "Ivory Coast": ["Cote d'Ivoire", "Cote D Ivoire"],
    "South Korea": ["Korea Republic", "Republic of Korea"],
    "Turkiye": ["Turkey", "Türkiye"],
    "United States": ["USA", "U.S.A.", "USMNT"],
}
CACHE: dict[tuple[str, ...], tuple[float, Any]] = {}


class OddsProxyError(Exception):
    """Error shown to the browser without leaking the API key."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name and name not in os.environ:
                os.environ[name] = value


def setting(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def normalize_team(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value or "")
    ascii_value = "".join(character for character in ascii_value if not unicodedata.combining(character))
    ascii_value = ascii_value.lower().replace("&", " and ")
    ascii_value = re.sub(r"[^a-z0-9]+", " ", ascii_value)
    return re.sub(r"\s+", " ", ascii_value).strip()


def team_variants(team: str) -> set[str]:
    normalized = normalize_team(team)
    variants = {normalized}
    for canonical, aliases in TEAM_ALIASES.items():
        all_names = [canonical] + aliases
        normalized_names = {normalize_team(name) for name in all_names}
        if normalized in normalized_names:
            variants.update(normalized_names)
    return {variant for variant in variants if variant}


def teams_match(expected: str, observed: str) -> bool:
    observed_normalized = normalize_team(observed)
    if not observed_normalized:
        return False
    for variant in team_variants(expected):
        if observed_normalized == variant:
            return True
        if len(variant) >= 5 and (variant in observed_normalized or observed_normalized in variant):
            return True
    return False


def mask_secret(text: str) -> str:
    api_key = setting("THE_ODDS_API_KEY")
    return text.replace(api_key, "<redacted>") if api_key else text


def ssl_verification_enabled() -> bool:
    value = setting("THE_ODDS_API_VERIFY_SSL", "true").lower()
    return value not in {"0", "false", "no", "off"}


def api_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return str(payload.get("message") or payload.get("error") or payload)
    except ValueError:
        pass

    text = response.text[:1000]
    title_match = re.search(r"<title>\s*(.*?)\s*</title>", text, re.I | re.S)
    if title_match:
        return re.sub(r"\s+", " ", title_match.group(1)).strip()
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:500] or response.reason


def odds_api_get(path: str, params: dict[str, str], cache_key: tuple[str, ...], cache_seconds: int) -> Any:
    api_key = setting("THE_ODDS_API_KEY")
    if not api_key:
        raise OddsProxyError("THE_ODDS_API_KEY is missing. Add it to .env or the process environment.", 503)

    cached = CACHE.get(cache_key)
    now = time.time()
    if cached and now - cached[0] < cache_seconds:
        return cached[1]

    endpoint = f"{setting('THE_ODDS_API_BASE_URL', BASE_URL).rstrip('/')}/{path.lstrip('/')}"
    request_params = {"apiKey": api_key, **params}
    headers = {
        "Accept": "application/json",
        "User-Agent": "worldcup2026-predictor/1.0",
    }
    verify_ssl = ssl_verification_enabled()
    if not verify_ssl:
        requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    try:
        response = requests.get(endpoint, params=request_params, headers=headers, timeout=20, verify=verify_ssl)
    except requests.RequestException as request_error:
        raise OddsProxyError(f"Could not reach The Odds API: {mask_secret(str(request_error))}") from request_error

    if response.status_code != 200:
        message = api_error_message(response)
        raise OddsProxyError(f"The Odds API returned HTTP {response.status_code}: {mask_secret(message)}", response.status_code)

    try:
        payload = response.json()
    except ValueError as json_error:
        raise OddsProxyError("The Odds API returned invalid JSON.") from json_error

    CACHE[cache_key] = (now, payload)
    return payload


def fetch_events() -> list[dict[str, Any]]:
    sport = setting("THE_ODDS_API_SPORT", "soccer_fifa_world_cup")
    cache_seconds = int(setting("ODDS_PROXY_CACHE_SECONDS", "300") or "300")
    events = odds_api_get(
        f"/sports/{sport}/events",
        {"dateFormat": "iso"},
        ("events", sport),
        cache_seconds,
    )
    if not isinstance(events, list):
        raise OddsProxyError("The Odds API response was not a list of events.")
    return events


def fetch_event_odds(event_id: str, markets: str) -> dict[str, Any]:
    sport = setting("THE_ODDS_API_SPORT", "soccer_fifa_world_cup")
    regions = setting("THE_ODDS_API_REGIONS", "uk,eu,us,au")
    cache_seconds = int(setting("ODDS_PROXY_CACHE_SECONDS", "300") or "300")
    payload = odds_api_get(
        f"/sports/{sport}/events/{event_id}/odds",
        {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        },
        ("event_odds", sport, event_id, regions, markets),
        cache_seconds,
    )
    if not isinstance(payload, dict):
        raise OddsProxyError("The Odds API event odds response was not an object.")
    return payload


def find_event(events: list[dict[str, Any]], requested_home: str, requested_away: str) -> tuple[dict[str, Any] | None, bool]:
    for event in events:
        api_home = str(event.get("home_team") or "")
        api_away = str(event.get("away_team") or "")
        direct_match = teams_match(requested_home, api_home) and teams_match(requested_away, api_away)
        if direct_match:
            return event, False
        reversed_match = teams_match(requested_home, api_away) and teams_match(requested_away, api_home)
        if reversed_match:
            return event, True
    return None, False


def parse_score_text(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    score_match = re.search(r"(?<!\d)(\d{1,2})\s*[-:]\s*(\d{1,2})(?!\d)", str(value))
    if not score_match:
        return None
    return int(score_match.group(1)), int(score_match.group(2))


def outcome_score(outcome: dict[str, Any], reversed_order: bool) -> str | None:
    for field_name in ("point", "name", "description"):
        parsed = parse_score_text(outcome.get(field_name))
        if parsed is None:
            continue
        home_goals, away_goals = parsed
        if reversed_order:
            home_goals, away_goals = away_goals, home_goals
        return f"{home_goals}-{away_goals}"
    return None


def outcome_price(outcome: dict[str, Any]) -> float | None:
    for field_name in ("price", "odds", "decimal"):
        if field_name not in outcome:
            continue
        try:
            return float(outcome[field_name])
        except (TypeError, ValueError):
            return None
    return None


def append_quote(target: dict[str, list[dict[str, Any]]], key: str,
                 bookmaker: dict[str, Any], price: float) -> None:
    book_name = bookmaker.get("title") or bookmaker.get("key") or "Bookmaker"
    target.setdefault(key, []).append({
        "book": str(book_name),
        "odds": price,
        "last_update": bookmaker.get("last_update"),
    })


def extract_correct_scores(event_odds: dict[str, Any], reversed_order: bool) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    correct_score: dict[str, list[dict[str, Any]]] = {}
    market_keys = set()
    for bookmaker in event_odds.get("bookmakers", []) or []:
        for market in bookmaker.get("markets", []) or []:
            market_key = str(market.get("key") or "")
            market_keys.add(market_key)
            if market_key != "correct_score":
                continue
            for outcome in market.get("outcomes", []) or []:
                if not isinstance(outcome, dict):
                    continue
                score = outcome_score(outcome, reversed_order)
                price = outcome_price(outcome)
                if score is not None and price is not None:
                    append_quote(correct_score, score, bookmaker, price)
    return correct_score, market_keys


def extract_match_result(event_odds: dict[str, Any], requested_home: str,
                         requested_away: str) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    match_result: dict[str, list[dict[str, Any]]] = {"home": [], "draw": [], "away": []}
    market_keys = set()
    for bookmaker in event_odds.get("bookmakers", []) or []:
        for market in bookmaker.get("markets", []) or []:
            market_key = str(market.get("key") or "")
            market_keys.add(market_key)
            if market_key != "h2h":
                continue
            for outcome in market.get("outcomes", []) or []:
                if not isinstance(outcome, dict):
                    continue
                name = str(outcome.get("name") or outcome.get("description") or "")
                price = outcome_price(outcome)
                if price is None:
                    continue
                normalized_name = normalize_team(name)
                if teams_match(requested_home, name):
                    append_quote(match_result, "home", bookmaker, price)
                elif teams_match(requested_away, name):
                    append_quote(match_result, "away", bookmaker, price)
                elif normalized_name in {"draw", "tie"}:
                    append_quote(match_result, "draw", bookmaker, price)
    return {key: value for key, value in match_result.items() if value}, market_keys


def correct_score_payload(requested_home: str, requested_away: str) -> tuple[dict[str, Any], int]:
    events = fetch_events()
    event, reversed_order = find_event(events, requested_home, requested_away)
    if event is None:
        return {
            "source": "The Odds API",
            "correct_score": {},
            "message": f"No The Odds API event matched {requested_home} vs {requested_away} for the configured sport.",
        }, 404
    event_id = event.get("id")
    if not event_id:
        return {
            "source": "The Odds API",
            "correct_score": {},
            "message": "The matched The Odds API event did not include an event id.",
        }, 502

    market_keys = set()
    market_errors: dict[str, str] = {}
    event_payload = event
    correct_score: dict[str, list[dict[str, Any]]] = {}
    match_result: dict[str, list[dict[str, Any]]] = {}

    try:
        event_odds = fetch_event_odds(str(event_id), setting("THE_ODDS_API_MARKETS", "correct_score"))
        event_payload = event_odds
        correct_score, keys = extract_correct_scores(event_odds, reversed_order)
        market_keys.update(keys)
    except OddsProxyError as market_error:
        market_errors["correct_score"] = str(market_error)

    try:
        h2h_odds = fetch_event_odds(str(event_id), "h2h")
        if event_payload is event:
            event_payload = h2h_odds
        match_result, keys = extract_match_result(h2h_odds, requested_home, requested_away)
        market_keys.update(keys)
    except OddsProxyError as market_error:
        market_errors["h2h"] = str(market_error)

    message = None
    if not correct_score and market_errors.get("correct_score"):
        message = market_errors["correct_score"]
    elif not correct_score:
        message = "The matched event did not include a correct_score market."
    if not correct_score and not match_result and market_errors:
        message = "; ".join(market_errors.values())

    return {
        "source": "The Odds API",
        "event": {
            "home": event_payload.get("home_team") or event.get("home_team"),
            "away": event_payload.get("away_team") or event.get("away_team"),
            "commence_time": event_payload.get("commence_time") or event.get("commence_time"),
        },
        "market_keys": sorted(market_keys),
        "correct_score": correct_score,
        "match_result": match_result,
        "market_errors": market_errors,
        "message": message,
    }, 200 if (correct_score or match_result) else 502


class OddsProxyHandler(BaseHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path == "/health":
            self.send_json({"ok": True, "source": "The Odds API proxy"})
            return
        if parsed_url.path != "/correct-score":
            self.send_json({"error": "Use /correct-score?home=Team&away=Team"}, 404)
            return

        query = urllib.parse.parse_qs(parsed_url.query)
        requested_home = (query.get("home") or [""])[0].strip()
        requested_away = (query.get("away") or [""])[0].strip()
        if not requested_home or not requested_away:
            self.send_json({"error": "Both home and away query parameters are required."}, 400)
            return

        try:
            payload, status = correct_score_payload(requested_home, requested_away)
        except OddsProxyError as proxy_error:
            payload = {"source": "The Odds API", "correct_score": {}, "message": str(proxy_error)}
            status = proxy_error.status
        self.send_json(payload, status)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_text: str, *args: Any) -> None:
        print("odds_proxy:", format_text % args)


def main() -> None:
    load_dotenv(os.path.join(HERE, ".env"))
    host = setting("ODDS_PROXY_HOST", "127.0.0.1")
    port = int(setting("ODDS_PROXY_PORT", "8787") or "8787")
    server = ThreadingHTTPServer((host, port), OddsProxyHandler)
    print(f"The Odds API proxy listening on http://{host}:{port}")
    print("Dashboard endpoint: http://127.0.0.1:8787/correct-score?home={home}&away={away}")
    server.serve_forever()


if __name__ == "__main__":
    main()
