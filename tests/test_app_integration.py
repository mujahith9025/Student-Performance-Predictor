import pytest
from app import compute_grade_and_status, get_interval_bounds, load_all_subject_artifacts

def test_compute_grade_and_status():
    """Verify grade tier boundaries and CSS badge mapping."""
    g_ap, css_ap, _ = compute_grade_and_status(95.0)
    assert g_ap == "A+"
    assert css_ap == "grade-A"
    
    g_a, css_a, _ = compute_grade_and_status(85.0)
    assert g_a == "A"
    
    g_b, css_b, _ = compute_grade_and_status(75.0)
    assert g_b == "B"
    assert css_b == "grade-B"
    
    g_f, css_f, _ = compute_grade_and_status(45.0)
    assert g_f == "F"
    assert css_f == "grade-F"

def test_get_interval_bounds():
    """Verify standard confidence interval calculation formulas."""
    lower, upper, margin = get_interval_bounds(pred_score=75.0, conf_str="95%", residual_std=2.0)
    # Z for 95% is 1.960 -> margin = 1.960 * 2.0 = 3.92
    assert pytest.approx(margin, 0.01) == 3.92
    assert pytest.approx(lower, 0.01) == 71.08
    assert pytest.approx(upper, 0.01) == 78.92

def test_load_all_subject_artifacts():
    """Verify app cached artifacts load properly."""
    models_bundle, metrics_bundle = load_all_subject_artifacts()
    assert isinstance(models_bundle, dict)
    assert isinstance(metrics_bundle, dict)
    assert len(models_bundle) > 0
