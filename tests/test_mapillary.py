import httpx

from roofsight.data.mapillary import MapillaryClient, keep, parse_image


def test_parse_and_filter() -> None:
    img = parse_image(
        {
            "id": 1,
            "thumb_2048_url": "u",
            "camera_type": "perspective",
            "quality_score": 0.9,
            "creator": {"username": "me"},
            "geometry": {"coordinates": [6.9, 50.9]},
        }
    )
    assert img is not None
    assert keep(img, "perspective", 0.6)
    assert "CC BY-SA 4.0" in img.attribution
    assert not keep(img, "spherical", 0.6)
    assert parse_image({"id": 2}) is None


def test_search_paginates() -> None:
    pages = {
        "https://graph.mapillary.com/images": {
            "data": [
                {
                    "id": 1,
                    "thumb_2048_url": "u1",
                    "camera_type": "perspective",
                    "quality_score": 0.9,
                },
                {"id": 2, "thumb_2048_url": "u2", "camera_type": "spherical", "quality_score": 0.9},
            ],
            "paging": {"next": "https://graph.mapillary.com/next"},
        },
        "https://graph.mapillary.com/next": {
            "data": [
                {
                    "id": 3,
                    "thumb_2048_url": "u3",
                    "camera_type": "perspective",
                    "quality_score": 0.2,
                }
            ],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=pages[str(request.url).split("?")[0]])

    client = MapillaryClient(token="t", client=httpx.Client(transport=httpx.MockTransport(handler)))
    ids = [i.id for i in client.search("0,0,1,1", limit=10)]
    assert ids == ["1"]


def test_grid_cells() -> None:
    from roofsight.data.mapillary import grid_cells

    cells = grid_cells("0,0,1,1", 2)
    assert len(cells) == 4
    assert cells[0] == "0.000000,0.000000,0.500000,0.500000"
    assert cells[-1] == "0.500000,0.500000,1.000000,1.000000"


def test_search_grid_retries_and_skips(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import roofsight.data.mapillary as m

    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bbox = request.url.params["bbox"]
        calls.append(bbox)
        if bbox.startswith("0.000000,0.000000"):
            return httpx.Response(500)  # this cell always fails → skipped
        item = {
            "id": bbox,
            "thumb_2048_url": "u",
            "camera_type": "perspective",
            "quality_score": 0.9,
        }
        return httpx.Response(200, json={"data": [item, item]})  # duplicate ids inside a cell

    client = MapillaryClient(token="t", client=httpx.Client(transport=httpx.MockTransport(handler)))
    ids = [i.id for i in client.search("0,0,1,1", limit=10, grid=2)]
    assert len(ids) == 3
    assert calls.count("0.000000,0.000000,0.500000,0.500000") == 4  # 1 try + 3 retries
