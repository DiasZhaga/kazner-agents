"""Tool: fetch the plain text of a Kazakh Wikipedia article (MediaWiki TextExtracts API)."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

import requests

from kazner_agents.errors import AgentError, TransientError

API_URL = "https://kk.wikipedia.org/w/api.php"
ARTICLE_URL = "https://kk.wikipedia.org/wiki/"
LICENCE = "CC BY-SA 4.0"


@dataclass
class Article:
    title: str
    url: str
    text: str


def fetch_article(title: str, user_agent: str, timeout: float = 20.0) -> Article:
    """Download one article as plain text. Network problems raise TransientError (retried)."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        "format": "json",
        "formatversion": 2,
    }
    try:
        response = requests.get(
            API_URL, params=params, headers={"User-Agent": user_agent}, timeout=timeout
        )
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise TransientError(f"Wikipedia API not reachable: {type(exc).__name__}") from exc
    if response.status_code == 429 or response.status_code >= 500:
        raise TransientError(f"Wikipedia API answered HTTP {response.status_code}")
    if response.status_code != 200:
        raise AgentError(f"Wikipedia API answered HTTP {response.status_code}")

    page = response.json()["query"]["pages"][0]
    if page.get("missing") or page.get("invalid"):
        raise AgentError(f"Kazakh Wikipedia has no article titled {title!r}")
    text = page.get("extract") or ""
    if not text.strip():
        raise AgentError(f"the article {title!r} has no text")
    real_title = page["title"]  # after redirects
    url = ARTICLE_URL + quote(real_title.replace(" ", "_"))
    return Article(title=real_title, url=url, text=text)
