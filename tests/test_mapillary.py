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
