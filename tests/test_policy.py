from verify.policy import classify  # type: ignore[import-not-found]
from verify.types import Claim, Evidence, Stance  # type: ignore[import-not-found]

_TH = {"T_support_ok": 0.7, "T_contra_veto": 0.7, "min_domain_class_weight": 0.2}


def _ev(eid, source_class):
    return Evidence(eid, f"https://{eid}.example", f"{eid}.example", source_class, None, "t", "c")


def _claim():
    return Claim("some claim", [])


def test_verified_when_strong_support_low_contra():
    # encyclopedia supports (1.0), forum contradicts (0.3 < 0.7 veto)
    ev = {1: _ev(1, "encyclopedia"), 2: _ev(2, "forum")}
    stance = {1: Stance.SUPPORTS, 2: Stance.CONTRADICTS}
    v = classify(_claim(), stance, ev, _TH)
    assert v.verdict == "verified"


def test_contested_when_support_and_contra_both_strong():
    ev = {1: _ev(1, "encyclopedia"), 2: _ev(2, "official")}
    stance = {1: Stance.SUPPORTS, 2: Stance.CONTRADICTS}  # 1.0 vs 1.0 → veto trips
    v = classify(_claim(), stance, ev, _TH)
    assert v.verdict == "contested"


def test_unsupported_when_no_support():
    ev = {1: _ev(1, "forum")}
    stance = {1: Stance.NEUTRAL}
    v = classify(_claim(), stance, ev, _TH)
    assert v.verdict == "unsupported"


def test_unsupported_when_support_below_threshold_and_no_contra():
    ev = {1: _ev(1, "forum")}  # 0.3 < 0.7
    stance = {1: Stance.SUPPORTS}
    v = classify(_claim(), stance, ev, _TH)
    assert v.verdict == "unsupported"
    assert v.support_weight == 0.3
    assert v.contra_weight == 0.0


def test_support_boundary_meeting_threshold_is_verified():
    # blog 0.4 + forum 0.3 meets T_support_ok (0.7), no contradiction
    ev = {1: _ev(1, "blog"), 2: _ev(2, "forum")}
    stance = {1: Stance.SUPPORTS, 2: Stance.SUPPORTS}
    v = classify(_claim(), stance, ev, _TH)
    assert v.verdict == "verified"


def test_veto_boundary_contra_at_threshold_is_contested():
    # support 1.0, contra = blog 0.4 + forum 0.3 = 0.7 → NOT < veto → not verified
    ev = {1: _ev(1, "encyclopedia"), 2: _ev(2, "blog"), 3: _ev(3, "forum")}
    stance = {1: Stance.SUPPORTS, 2: Stance.CONTRADICTS, 3: Stance.CONTRADICTS}
    v = classify(_claim(), stance, ev, _TH)
    assert v.verdict == "contested"


def test_only_contradiction_is_unsupported():
    # support 0, contra 1.0 → else-branch (support>0 is False) → unsupported
    ev = {1: _ev(1, "official")}
    stance = {1: Stance.CONTRADICTS}
    v = classify(_claim(), stance, ev, _TH)
    assert v.verdict == "unsupported"
    assert v.support_weight == 0.0


def test_min_domain_class_weight_floor_excludes_low_trust():
    # unknown class (0.2) supports; a floor above 0.2 drops it from the sum
    ev = {1: _ev(1, "unknown")}
    stance = {1: Stance.SUPPORTS}
    th = {"T_support_ok": 0.7, "T_contra_veto": 0.7, "min_domain_class_weight": 0.5}
    v = classify(_claim(), stance, ev, th)
    assert v.support_weight == 0.0
    assert v.verdict == "unsupported"


def test_min_domain_class_weight_floor_includes_at_boundary():
    # default floor 0.2 keeps unknown (0.2) evidence: 0.2 >= 0.2
    ev = {1: _ev(1, "unknown")}
    stance = {1: Stance.SUPPORTS}
    v = classify(_claim(), stance, ev, _TH)
    assert v.support_weight == 0.2
