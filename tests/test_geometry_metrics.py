import math

from roofsight.eval.geometry import PlaneEstimate, azimuth_error, geometry_mae


def test_azimuth_is_circular() -> None:
    assert azimuth_error(359.0, 1.0) == 2.0
    assert azimuth_error(10.0, 190.0) == 180.0


def test_geometry_mae() -> None:
    truth = [
        PlaneEstimate(image_id=1, plane_id=1, pitch_deg=35, azimuth_deg=180),
        PlaneEstimate(image_id=1, plane_id=2, pitch_deg=35, azimuth_deg=0),
        PlaneEstimate(image_id=2, plane_id=1, pitch_deg=40, azimuth_deg=90),
    ]
    est = [
        PlaneEstimate(image_id=1, plane_id=1, pitch_deg=37, azimuth_deg=175),
        PlaneEstimate(image_id=1, plane_id=2, pitch_deg=31, azimuth_deg=357),
    ]
    r = geometry_mae(truth, est)
    assert r.n_planes == 2 and r.n_missing == 1
    assert math.isclose(r.pitch_mae, 3.0)
    assert math.isclose(r.azimuth_mae, 4.0)
