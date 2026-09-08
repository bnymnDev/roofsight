"""Wikimedia Commons as an image source.

Commons has whole-building photos in the framing a PV planner sees: the house fills the frame,
shot from the pavement, at 3000 px. What it does not have much of is ordinary post-war housing;
uploads skew to listed and otherwise notable buildings. It is therefore a source of roof and
obstacle variety next to own photos, not a replacement for them.

Transport note: Wikimedia rejects some HTTP clients by TLS fingerprint with a 403 and their
robot-policy notice — ``httpx``, which the rest of the package uses, is among them. The
standard library's ``urllib`` is accepted, so this module uses it and paces itself.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class CommonsThrottledError(RuntimeError):
    """Wikimedia is rate-limiting this network, not just this request.

    Their media hosts throttle by client address. A data-centre address is often limited to a
    trickle no matter how slowly the client asks, while an ordinary connection downloads
    thousands of files without a single 429. Collect from the machine you actually work on.
    """


API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "roofsight/0.1 (https://github.com/bnymnDev/roofsight)"
FILE_PAGE = "https://commons.wikimedia.org/wiki/"

#: Commons serves pre-rendered thumbnails at these widths; asking for other sizes makes the
#: media host render on the fly, which it rate-limits.
THUMB_WIDTHS: tuple[int, ...] = (320, 640, 800, 1024, 1280, 1920, 2560)

#: Commons ``LicenseShortName`` → SPDX-ish id. Everything not listed here is refused: the
#: dataset is CC-BY-SA 4.0 and may only contain material that can be redistributed under it.
LICENSE_MAP: dict[str, str] = {
    "cc by-sa 4.0": "CC-BY-SA-4.0",
    "cc by-sa 3.0": "CC-BY-SA-3.0",
    "cc by-sa 2.5": "CC-BY-SA-2.5",
    "cc by-sa 2.0": "CC-BY-SA-2.0",
    "cc by 4.0": "CC-BY-4.0",
    "cc by 3.0": "CC-BY-3.0",
    "cc by 2.5": "CC-BY-2.5",
    "cc by 2.0": "CC-BY-2.0",
    "cc0": "CC0-1.0",
    "public domain": "public-domain",
}

_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def normalize_license(short_name: str | None) -> str | None:
    """Map a Commons license label to an SPDX-ish id, or None if it is not redistributable.

    Country ports ("CC BY-SA 3.0 de") map to their base license; the port only localises the
    legal text.
    """
    if not short_name:
        return None
    s = _WS.sub(" ", short_name.strip().lower())
    if s in LICENSE_MAP:
        return LICENSE_MAP[s]
    # country ports: "cc by-sa 3.0 de", "cc by-sa 2.0 fr"
    m = re.match(r"^(cc by(?:-sa)? \d\.\d)(?: [a-z]{2}(?:-\w+)?)?$", s)
    if m and m.group(1) in LICENSE_MAP:
        return LICENSE_MAP[m.group(1)]
    if s.startswith("public domain") or s == "pd":
        return "public-domain"
    return None


def plain_text(html: str | None) -> str:
    """Commons returns author and credit as HTML fragments."""
    if not html:
        return ""
    return _WS.sub(" ", _TAGS.sub(" ", html)).strip()


@dataclass(frozen=True, slots=True)
class CommonsImage:
    title: str  # "File:Foo.jpg"
    url: str  # thumbnail at the requested width
    width: int
    height: int
    license: str  # normalized
    author: str
    page_url: str

    @property
    def file_name(self) -> str:
        """A stable, filesystem-safe name derived from the Commons title."""
        stem = self.title.removeprefix("File:")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
        return f"commons_{safe[:120]}"

    @property
    def attribution(self) -> str:
        author = self.author or "unknown author"
        return f"{author}, Wikimedia Commons, {self.license}, {self.page_url}"


class CommonsClient:
    """A paced, retrying Commons API client.

    ``min_interval`` seconds pass between requests; 429 and 503 are retried with the
    ``Retry-After`` the server asks for, doubling otherwise.
    """

    def __init__(
        self,
        user_agent: str = USER_AGENT,
        min_interval: float = 1.0,
        retries: int = 4,
        opener: Any = None,
        max_retry_wait: float = 120.0,
        throttle_giveup: int = 4,
    ) -> None:
        self.user_agent = user_agent
        self.min_interval = min_interval
        self.retries = retries
        self.max_retry_wait = max_retry_wait
        self.throttle_giveup = throttle_giveup
        self._opener = opener or urllib.request.build_opener()
        self._last = 0.0
        self._throttled = 0

    def _wait(self) -> None:
        gap = self.min_interval - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()

    def _fetch(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        for attempt in range(self.retries + 1):
            self._wait()
            try:
                with self._opener.open(req, timeout=120) as r:
                    data: bytes = r.read()
                    self._throttled = 0
                    return data
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    self._throttled += 1
                    if self._throttled >= self.throttle_giveup:
                        raise CommonsThrottledError(
                            f"Wikimedia returned 429 on {self._throttled} requests in a row even "
                            f"at {self.min_interval:.0f}s between them. This network is rate "
                            "limited; run the collection from an ordinary connection."
                        ) from e
                if e.code not in (429, 500, 502, 503) or attempt == self.retries:
                    raise
                after = e.headers.get("Retry-After") if e.headers else None
                delay = float(after) if after and after.isdigit() else 2.0 * (2**attempt)
                delay = min(delay, self.max_retry_wait)
                log.warning("commons: HTTP %s, waiting %.0fs", e.code, delay)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.retries:
                    raise
                log.warning("commons: %s, retrying", e)
                time.sleep(2.0 * (2**attempt))
        raise RuntimeError("unreachable")

    def api(self, **params: Any) -> dict[str, Any]:
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")
        url = f"{API}?{urllib.parse.urlencode(params)}"
        payload: dict[str, Any] = json.loads(self._fetch(url))
        if "error" in payload:
            raise RuntimeError(f"commons api error: {payload['error']}")
        return payload

    def search(self, query: str, limit: int) -> Iterator[str]:
        """File titles matching a CirrusSearch query, e.g. ``deepcat:"Houses in Germany"``."""
        offset, seen = 0, 0
        while seen < limit:
            batch = min(50, limit - seen)
            d = self.api(
                action="query",
                list="search",
                srsearch=query,
                srnamespace=6,
                srlimit=batch,
                sroffset=offset,
            )
            hits = d.get("query", {}).get("search", [])
            if not hits:
                return
            for h in hits:
                yield h["title"]
                seen += 1
            offset += len(hits)
            if "continue" not in d:
                return

    def category_members(self, category: str, limit: int) -> Iterator[str]:
        """File titles directly in a category (no recursion; use ``deepcat:`` in search)."""
        title = category if category.startswith("Category:") else f"Category:{category}"
        cont: str | None = None
        seen = 0
        while seen < limit:
            params: dict[str, Any] = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": title,
                "cmtype": "file",
                "cmlimit": min(500, limit - seen),
            }
            if cont:
                params["cmcontinue"] = cont
            d = self.api(**params)
            for m in d.get("query", {}).get("categorymembers", []):
                yield m["title"]
                seen += 1
            cont = d.get("continue", {}).get("cmcontinue")
            if not cont:
                return

    def image_info(self, titles: list[str], width: int = 1280) -> list[CommonsImage]:
        """Metadata for up to 50 titles. Files without a redistributable license are dropped."""
        if width not in THUMB_WIDTHS:
            raise ValueError(f"width must be one of {THUMB_WIDTHS}, got {width}")
        out: list[CommonsImage] = []
        for i in range(0, len(titles), 50):
            d = self.api(
                action="query",
                titles="|".join(titles[i : i + 50]),
                prop="imageinfo",
                iiprop="url|size|extmetadata",
                iiurlwidth=width,
            )
            for page in d.get("query", {}).get("pages", []):
                img = self._parse(page, width)
                if img is not None:
                    out.append(img)
        return out

    def _parse(self, page: dict[str, Any], width: int) -> CommonsImage | None:
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url")
        if not url or not info.get("width"):
            return None
        meta = info.get("extmetadata") or {}
        lic = normalize_license((meta.get("LicenseShortName") or {}).get("value"))
        if lic is None:
            log.debug("commons: skipping %s, license not redistributable", page.get("title"))
            return None
        w = min(width, int(info["width"]))
        h = round(int(info["height"]) * w / int(info["width"]))
        return CommonsImage(
            title=page["title"],
            url=url,
            width=int(info.get("thumbwidth") or w),
            height=int(info.get("thumbheight") or h),
            license=lic,
            author=plain_text((meta.get("Artist") or {}).get("value")),
            page_url=FILE_PAGE + urllib.parse.quote(page["title"].replace(" ", "_")),
        )

    def download(self, img: CommonsImage, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            return dest
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(self._fetch(img.url))
        tmp.replace(dest)
        return dest


def collect(
    client: CommonsClient,
    queries: list[str],
    categories: list[str],
    per_source_limit: int,
    width: int = 1280,
) -> list[CommonsImage]:
    """Titles from every query and category, deduped, with metadata resolved."""
    titles: list[str] = []
    seen: set[str] = set()
    for q in queries:
        for t in client.search(q, per_source_limit):
            if t not in seen:
                seen.add(t)
                titles.append(t)
        log.info("commons: %d titles after query %r", len(titles), q)
    for c in categories:
        for t in client.category_members(c, per_source_limit):
            if t not in seen:
                seen.add(t)
                titles.append(t)
        log.info("commons: %d titles after category %r", len(titles), c)
    images = client.image_info(titles, width)
    log.info("commons: %d of %d titles usable (license, size)", len(images), len(titles))
    return images
