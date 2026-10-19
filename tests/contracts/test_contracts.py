from autonomy.contracts import MarkerCounts, Point2D, ModuleStatus, DetectedSite


def test_marker_counts_reject_negative_values():
    try:
        MarkerCounts(red=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative counts must be rejected")


def test_point_uses_team_local_frame_by_default():
    point = Point2D(x_m=1.0, y_m=2.0)
    assert point.frame_id == "team_local"


def test_detected_site_confidence_range():
    site = DetectedSite(
        site_id="site-1",
        position=Point2D(0.0, 0.0),
        counts=MarkerCounts(red=2),
        confidence=0.9,
    )
    assert site.counts.red == 2
    assert ModuleStatus.OK.value == "ok"
