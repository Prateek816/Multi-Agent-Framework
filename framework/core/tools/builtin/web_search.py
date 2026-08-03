"""
General-purpose web research tools, implemented as LangChain tools
(langchain_core.tools.tool). They are not bound to an MCP server here —
wrap ALL_TOOLS with your MCP server layer (e.g. langchain-mcp-adapters)
separately.
"""

import asyncio
import re
from typing import Optional
from urllib.parse import quote

import httpx
from langchain_core.tools import tool

USER_AGENT = "GeneralResearchAgent/1.0"

# ─── WEB SEARCH ───────────────────────────────────────────────────────────────

@tool
async def web_search(query: str, num_results: int = 10) -> list[dict]:
    """Search the web using DuckDuckGo. No API key required.
    Use for general research on any topic, person, organization, or event."""
    from ddgs import DDGS

    def _search():
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=num_results))
        except Exception as e:
            raise RuntimeError(f"DuckDuckGo search failed: {str(e)}") from e

    results = await asyncio.to_thread(_search)
    return [
        {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
        for r in results
    ]


@tool
async def scrape_page(url: str) -> str:
    """Scrape the full text content from any webpage via Jina Reader. No API key required.
    Handles JS-rendered pages, news sites, and most public webpages."""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            jina_url = f"https://r.jina.ai/{url}"
            r = await client.get(
                jina_url,
                headers={"User-Agent": USER_AGENT, "Accept": "text/plain"},
                follow_redirects=True,
            )
            r.raise_for_status()
            return r.text
        except Exception:
            # Fallback: basic HTTP fetch
            try:
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                return r.text[:8000]
            except Exception as e:
                raise RuntimeError(f"Failed to scrape {url}: {str(e)}") from e

# ─── YOUTUBE ──────────────────────────────────────────────────────────────────

@tool
async def get_youtube_transcript(url: str, lang: str = "en") -> str:
    """Extract the full transcript from a YouTube video. No API key required.
    Works with videos that have captions (manual or auto-generated).
    Use for researching talks, interviews, tutorials, or any video content."""
    from youtube_transcript_api import YouTubeTranscriptApi

    def _extract():
        match = re.search(r"(?:v=|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})", url)
        if not match:
            raise ValueError(f"Could not extract video ID from URL: {url}")
        video_id = match.group(1)
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        except Exception as first_err:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
            except Exception as e:
                return f"No transcript available for {url}: {str(first_err)}; fallback error: {str(e)}"
        return " ".join(item["text"] for item in transcript)

    return await asyncio.to_thread(_extract)

# ─── RSS ──────────────────────────────────────────────────────────────────────

@tool
async def read_rss_feed(url: str, max_items: int = 20) -> dict:
    """Read any RSS or Atom feed and return its latest entries. No API key required.
    Use for tracking news, blogs, publications, or any site that publishes a feed."""
    import feedparser

    def _parse():
        feed = feedparser.parse(url)
        if feed.bozo and not feed.entries:
            exc = feed.get("bozo_exception", "unknown error")
            raise RuntimeError(f"Failed to parse RSS feed {url}: {exc}")
        return {
            "title": feed.feed.get("title", ""),
            "description": feed.feed.get("description", ""),
            "items": [
                {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "summary": entry.get("summary", "")[:500],
                    "published": entry.get("published", ""),
                }
                for entry in feed.entries[:max_items]
            ],
        }

    return await asyncio.to_thread(_parse)

# ─── REDDIT ───────────────────────────────────────────────────────────────────

@tool
async def search_reddit(query: str, subreddit: Optional[str] = None, limit: int = 10) -> list[dict]:
    """Search Reddit posts and threads. No API key required.
    Use for gathering public opinions, discussions, and community feedback on any topic."""
    params: dict = {"q": query, "limit": limit, "sort": "relevance", "t": "all"}
    if subreddit:
        search_url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params["restrict_sr"] = 1
    else:
        search_url = "https://www.reddit.com/search.json"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            search_url,
            params=params,
            headers={"User-Agent": USER_AGENT},
        )
        r.raise_for_status()
        posts = r.json().get("data", {}).get("children", [])
        return [
            {
                "title": p["data"].get("title"),
                "url": p["data"].get("url"),
                "score": p["data"].get("score"),
                "subreddit": p["data"].get("subreddit"),
                "permalink": "https://www.reddit.com" + p["data"].get("permalink", ""),
                "text": p["data"].get("selftext", "")[:500],
                "num_comments": p["data"].get("num_comments"),
            }
            for p in posts
        ]


@tool
async def read_reddit_post(url: str) -> dict:
    """Read a Reddit post and its top comments in full. No API key required.
    Use for deep-diving into a specific discussion thread."""
    clean = url.split("?")[0].rstrip("/")
    json_url = clean + ".json?limit=20"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            json_url,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        r.raise_for_status()
        data = r.json()
    post_data = data[0]["data"]["children"][0]["data"]
    comments_raw = data[1]["data"]["children"] if len(data) > 1 else []
    comments = [
        {
            "author": c["data"].get("author"),
            "text": c["data"].get("body", "")[:500],
            "score": c["data"].get("score"),
        }
        for c in comments_raw
        if c.get("kind") == "t1"
    ][:15]
    return {
        "title": post_data.get("title"),
        "author": post_data.get("author"),
        "subreddit": post_data.get("subreddit"),
        "score": post_data.get("score"),
        "text": post_data.get("selftext", ""),
        "url": post_data.get("url"),
        "num_comments": post_data.get("num_comments"),
        "comments": comments,
    }

# ─── GITHUB ───────────────────────────────────────────────────────────────────

@tool
async def search_github(query: str, limit: int = 10) -> list[dict]:
    """Search GitHub repositories. No API key required (unauthenticated rate limit: 60/hr).
    Use for researching open-source projects, libraries, or a person's or organization's codebase."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "per_page": limit, "sort": "stars"},
            headers={"Accept": "application/vnd.github.v3+json", "User-Agent": USER_AGENT},
        )
        r.raise_for_status()
        return [
            {
                "name": item.get("full_name"),
                "description": item.get("description"),
                "stars": item.get("stargazers_count"),
                "language": item.get("language"),
                "url": item.get("html_url"),
                "topics": item.get("topics", []),
            }
            for item in r.json().get("items", [])
        ]


@tool
async def read_github_repo(owner: str, repo: str, path: str = "") -> str:
    """Read a GitHub repository's README or any file within it. No API key required for public repos.
    Use for researching a project's documentation, code, or structure."""
    safe_owner = quote(owner, safe="")
    safe_repo = quote(repo, safe="")
    api_path = f"contents/{quote(path, safe='/')}" if path else "readme"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"https://api.github.com/repos/{safe_owner}/{safe_repo}/{api_path}",
            headers={
                "Accept": "application/vnd.github.v3.raw",
                "User-Agent": USER_AGENT,
            },
        )
        r.raise_for_status()
        return r.text[:10000]


# Convenience list for wiring these into an agent or, later, an MCP server layer
ALL_TOOLS = [
    web_search,
    scrape_page,
    get_youtube_transcript,
    read_rss_feed,
    search_reddit,
    read_reddit_post,
    search_github,
    read_github_repo,
]