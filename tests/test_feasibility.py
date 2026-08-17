from validation.feasibility import compute_gate_funnel, evaluate_feasibility


def test_funnel_applies_stages_in_sequence_on_survivors_only():
    candidates = list(range(100))  # 0..99
    stages = [
        ("even", lambda x: x % 2 == 0),          # 100 -> 50
        ("divisible_by_4", lambda x: x % 4 == 0),  # 50 -> 25
        ("gt_10", lambda x: x > 10),                 # 25 -> 22
    ]
    report = compute_gate_funnel(candidates, stages)
    assert [s.count_before for s in report.stages] == [100, 50, 25]
    assert [s.count_after for s in report.stages] == [50, 25, 22]
    assert report.initial_count == 100
    assert report.final_count == 22


def test_dropped_and_survival_rate_computed_correctly():
    candidates = list(range(100))
    stages = [("half", lambda x: x % 2 == 0)]
    report = compute_gate_funnel(candidates, stages)
    stage = report.stages[0]
    assert stage.dropped == 50
    assert stage.survival_rate == 0.5


def test_biggest_drop_stage_identifies_the_worst_gate():
    candidates = list(range(1000))
    stages = [
        ("mild", lambda x: x < 900),      # drops 100
        ("severe", lambda x: x % 10 == 0),  # drops most of the remaining 900 -> 90 survive, drops 810
        ("mild2", lambda x: x < 800),         # drops a handful more
    ]
    report = compute_gate_funnel(candidates, stages)
    worst = report.biggest_drop_stage()
    assert worst.name == "severe"


def test_empty_candidates_yields_zero_counts_without_crash():
    report = compute_gate_funnel([], [("stage", lambda x: True)])
    assert report.stages[0].count_before == 0
    assert report.stages[0].count_after == 0
    import math
    assert math.isnan(report.stages[0].survival_rate)


def test_no_stages_yields_empty_report():
    report = compute_gate_funnel(list(range(10)), [])
    assert report.stages == []
    assert report.final_count == 0
    assert report.biggest_drop_stage() is None


def test_feasibility_ok_status_when_above_warn_threshold():
    candidates = list(range(500))
    report = compute_gate_funnel(candidates, [("keep_half", lambda x: x % 2 == 0)])
    verdict = evaluate_feasibility(report, warn_below=100, reject_below=50)
    assert verdict.status == "ok"
    assert verdict.final_count == 250


def test_feasibility_warn_status_between_reject_and_warn_thresholds():
    candidates = list(range(160))
    report = compute_gate_funnel(candidates, [("keep_half", lambda x: x % 2 == 0)])  # -> 80
    verdict = evaluate_feasibility(report, warn_below=100, reject_below=50)
    assert verdict.status == "warn"
    assert "Do NOT add filters" in verdict.recommendation


def test_feasibility_reject_status_below_reject_threshold():
    candidates = list(range(60))
    report = compute_gate_funnel(candidates, [("keep_tenth", lambda x: x % 10 == 0)])  # -> 6
    verdict = evaluate_feasibility(report, warn_below=100, reject_below=50)
    assert verdict.status == "reject"
    assert "do not proceed to tuning" in verdict.recommendation


def test_feasibility_recommendation_names_the_worst_gate():
    candidates = list(range(1000))
    stages = [
        ("gentle", lambda x: x < 950),        # drops 50
        ("brutal", lambda x: x % 50 == 0),      # drops nearly everything remaining
    ]
    report = compute_gate_funnel(candidates, stages)
    verdict = evaluate_feasibility(report, warn_below=100, reject_below=50)
    assert "'brutal'" in verdict.recommendation
