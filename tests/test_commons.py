"""Commons client: licenses, paging, attribution. No network."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from roofsight.data.commons import (
    CommonsClient,
    CommonsImage,
    collect,
    normalize_license,
    plain_text,
)


class FakeOpener:
    """Answers with queued payloads and records the urls it was asked for."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def open(self, req: Any, timeout: float = 0) -> Any:
        self.urls.append(req.full_url)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        body = r if isinstance(r, bytes) else json.dumps(r).encode()
        return _Resp(body)


class _Resp(io.BytesIO):
    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *a: object) -> None:
        self.close()


def client(responses: list[Any]) -> CommonsClient:
    return CommonsClient(min_interval=0.0, opener=FakeOpener(responses))


def test_normalize_license() -> None:
    assert normalize_license("CC BY-SA 4.0") == "CC-BY-SA-4.0"
    assert normalize_license("CC BY-SA 3.0 de") == "CC-BY-SA-3.0"  # country port
    assert normalize_license("cc by 2.0") == "CC-BY-2.0"
    assert normalize_license("CC0") == "CC0-1.0"
    assert normalize_license("Public domain") == "public-domain"
    assert normalize_license("GFDL") is None  # not redistributable under CC-BY-SA
    assert normalize_license("Fair use") is None
    assert normalize_license(None) is None


def test_plain_text_strips_commons_html() -> None:
    assert plain_text('<a href="/wiki/User:X" title="X">Jane  Doe</a>') == "Jane Doe"
    assert plain_text(None) == ""


def _page(title: str, lic: str, w: int = 3000, h: int = 2000) -> dict[str, Any]:
    return {
        "title": title,
        "imageinfo": [
            {
                "thumburl": f"https://upload.wikimedia.org/{title}/1280px.jpg",
                "url": f"https://upload.wikimedia.org/{title}.jpg",
                "width": w,
                "height": h,
                "thumbwidth": 1280,
                "thumbheight": round(h * 1280 / w),
                "extmetadata": {
                    "LicenseShortName": {"value": lic},
                    "Artist": {"value": "<a>Jane Doe</a>"},
                },
            }
        ],
    }


def test_image_info_drops_unusable_licenses() -> None:
    c = client(
        [{"query": {"pages": [_page("File:A.jpg", "CC BY-SA 4.0"), _page("File:B.jpg", "GFDL")]}}]
    )
    imgs = c.image_info(["File:A.jpg", "File:B.jpg"])
    assert [i.title for i in imgs] == ["File:A.jpg"]
    img = imgs[0]
    assert img.license == "CC-BY-SA-4.0"
    assert img.author == "Jane Doe"
    assert img.attribution == (
        "Jane Doe, Wikimedia Commons, CC-BY-SA-4.0, https://commons.wikimedia.org/wiki/File%3AA.jpg"
    )
    assert img.file_name == "commons_A.jpg"


def test_image_info_rejects_non_bucket_widths() -> None:
    with pytest.raises(ValueError, match="width must be one of"):
        client([]).image_info(["File:A.jpg"], width=999)


def test_file_name_is_filesystem_safe() -> None:
    img = CommonsImage(
        title="File:Haus, Köln – Nr. 5 (2019).jpg",
        url="u",
        width=1,
        height=1,
        license="CC-BY-SA-4.0",
        author="",
        page_url="p",
    )
    assert img.file_name == "commons_Haus_K_ln_Nr._5_2019_.jpg"
    assert img.attribution.startswith("unknown author, Wikimedia Commons")


def test_search_pages_until_the_limit() -> None:
    page1 = {"query": {"search": [{"title": f"File:{i}.jpg"} for i in range(50)]}, "continue": {}}
    page2 = {"query": {"search": [{"title": "File:50.jpg"}]}}
    c = client([page1, page2])
    assert len(list(c.search("q", 51))) == 51
    assert "sroffset=50" in c._opener.urls[1]  # type: ignore[attr-defined]


def test_category_members_follows_continue() -> None:
    c = client(
        [
            {
                "query": {"categorymembers": [{"title": "File:A.jpg"}]},
                "continue": {"cmcontinue": "X"},
            },
            {"query": {"categorymembers": [{"title": "File:B.jpg"}]}},
        ]
    )
    assert list(c.category_members("Houses in Germany", 10)) == ["File:A.jpg", "File:B.jpg"]
    assert "Category%3AHouses+in+Germany" in c._opener.urls[0]  # type: ignore[attr-defined]


def test_retries_on_429_then_succeeds() -> None:
    err = urllib.error.HTTPError("u", 429, "slow down", {"Retry-After": "0"}, None)  # type: ignore[arg-type]
    c = client([err, {"query": {"search": [{"title": "File:A.jpg"}]}}])
    assert list(c.search("q", 1)) == ["File:A.jpg"]


def test_gives_up_on_404() -> None:
    err = urllib.error.HTTPError("u", 404, "gone", {}, None)  # type: ignore[arg-type]
    with pytest.raises(urllib.error.HTTPError):
        list(client([err]).search("q", 1))


def test_collect_dedupes_across_queries_and_categories() -> None:
    c = client(
        [
            {"query": {"search": [{"title": "File:A.jpg"}, {"title": "File:B.jpg"}]}},
            {"query": {"categorymembers": [{"title": "File:B.jpg"}, {"title": "File:C.jpg"}]}},
            {"query": {"pages": [_page(f"File:{x}.jpg", "CC BY-SA 4.0") for x in "ABC"]}},
        ]
    )
    imgs = collect(c, ["q"], ["Cat"], per_source_limit=2)
    assert [i.title for i in imgs] == ["File:A.jpg", "File:B.jpg", "File:C.jpg"]


def test_download_is_idempotent(tmp_path: Path) -> None:
    c = client([b"\xff\xd8jpegbytes"])
    img = CommonsImage("File:A.jpg", "https://u/x.jpg", 10, 10, "CC-BY-SA-4.0", "J", "p")
    dest = tmp_path / "a.jpg"
    assert c.download(img, dest).read_bytes() == b"\xff\xd8jpegbytes"
    assert c.download(img, dest) == dest  # no second request queued, so this would raise
    assert not list(tmp_path.glob("*.part"))
