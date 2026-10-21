import math

import numpy as np

from phonetics.evaluation.phonetic_matches import classify_pairs, normalize_final, summarize


def _columns(left_i, right_i, left_f, right_f, distance=None, mask=None):
    n = len(left_i)
    return {
        "left_gold_initial": np.asarray(left_i), "right_gold_initial": np.asarray(right_i),
        "left_gold_final": np.asarray(left_f), "right_gold_final": np.asarray(right_f),
        "distance": np.asarray(distance if distance is not None else np.arange(n), dtype=float),
        "overlap_mask": np.asarray(mask if mask is not None else ["000"] * n),
    }


def test_contextual_palatal_u_normalization_does_not_affect_g():
    assert normalize_final("q", "u") == "ü"
    assert normalize_final("q", "uan") == "üan"
    assert normalize_final("q", "un") == "ün"
    assert normalize_final("q", "ue") == "üe"
    assert normalize_final("g", "u") == "u"
    result = classify_pairs(_columns(["g"], ["q"], ["u"], ["u"]))
    assert not result["normalized_equal_final"][0]
    assert not result["close_final"][0]
    assert classify_pairs(_columns(["g"], ["q"], ["u"], ["u"]), "extended")["close_final"][0]


def test_apical_i_is_contextual_and_not_ordinary_i():
    result = classify_pairs(_columns(["z", "zh", "b"], ["zh", "b", "p"], ["i"] * 3, ["i"] * 3))
    assert result["relation_final"].tolist() == ["apical_place", "", ""]
    assert result["normalized_equal_final"].tolist() == [False, False, True]


def test_relations_are_symmetric_and_not_transitively_expanded():
    forward = classify_pairs(_columns(["j", "q"], ["q", "c"], ["a", "a"], ["o", "o"]))
    reverse = classify_pairs(_columns(["q", "c"], ["j", "q"], ["o", "o"], ["a", "a"]))
    assert forward["close_initial"].tolist() == reverse["close_initial"].tolist() == [True, True]
    # j-q and q-c being listed must not imply j-c through a graph closure.
    assert not classify_pairs(_columns(["j"], ["c"], ["a"], ["o"]))["close_initial"][0]


def test_extended_profile_keeps_sensitivity_additions_separate():
    columns = _columns(["n"], ["l"], ["a"], ["o"])
    assert not classify_pairs(columns, "core")["close_initial"][0]
    assert classify_pairs(columns, "extended")["relation_initial"][0] == "dialect_learner_looser"


def test_summary_empty_primary_denominator_and_exclusive_sum():
    columns = _columns(
        ["b", "b", "m"], ["p", "p", "f"], ["an", "an", "a"], ["ang", "ang", "o"],
        distance=[5, 15, 15], mask=["001", "000", "000"],
    )
    rows = summarize(columns)["bins"]
    assert rows[0]["count"] == 1 and rows[0]["mask000_count"] == 0
    assert math.isnan(rows[0]["either_fraction_mask000"])
    assert rows[0]["exclusive_sum_ok"]
    assert rows[1]["mask000_count"] == 2
    assert rows[1]["both_count"] == 1
    assert rows[1]["either_count"] == 1
    assert rows[1]["neither_count"] == 1
    assert rows[1]["exclusive_sum_ok"]
    assert sum(rows[1][f"exclusive_{name}_count"] for name in ("both", "initial_only", "final_only", "neither")) == 2


def test_empty_input_and_negative_distance_validation():
    empty = _columns([], [], [], [], distance=[], mask=[])
    assert classify_pairs(empty)["close_initial"].shape == (0,)
    assert all(row["exclusive_sum_ok"] for row in summarize(empty)["bins"])
    bad = _columns(["b"], ["p"], ["a"], ["o"], distance=[-1])
    try:
        summarize(bad)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative distance was accepted")
