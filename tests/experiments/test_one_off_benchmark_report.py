"""Check that the one-off report labels the fastest median as the winner."""

from experiments.one_off_benchmark import _case_label, _print_report, _winner_report


def test_vectorized_case_labels_keep_convention_and_backend():
    assert (
        _case_label("test_hamilton_numpy_rotate_10k_quatops", "NumPy rotate (10k)")
        == "Hamilton quatops"
    )
    assert (
        _case_label(
            "test_jpl_numpy_multiply_10k_numpy_quaternion_checked",
            "NumPy multiply (10k, checked)",
        )
        == "JPL numpy_quaternion_checked"
    )


def test_winner_report_selects_lowest_median():
    plugin_data = {
        "benchmarks": [
            {
                "group": "hot multiply",
                "name": "test_hot_multiply_scipy",
                "stats": {
                    "min": 3e-6,
                    "median": 4e-6,
                    "mean": 5e-6,
                    "stddev": 1e-6,
                    "rounds": 9,
                },
            },
            {
                "group": "hot multiply",
                "name": "test_hot_multiply_namedtuple",
                "stats": {
                    "min": 2e-6,
                    "median": 2.01e-6,
                    "mean": 2.1e-6,
                    "stddev": 0.2e-6,
                    "rounds": 9,
                },
            },
            {
                "group": "hot multiply",
                "name": "test_hot_multiply_tuple",
                "stats": {
                    "min": 1e-6,
                    "median": 2e-6,
                    "mean": 2e-6,
                    "stddev": 0.2e-6,
                    "rounds": 9,
                },
            },
        ]
    }

    result = _winner_report(plugin_data)

    assert result["hot multiply"]["winner"] == "tuple"
    assert result["hot multiply"]["co_winners"] == ["tuple", "namedtuple"]
    assert result["hot multiply"]["results"][0]["relative_to_winner"] == 1.0


def test_report_prints_winner_label(capsys):
    report = {
        "environment": {
            "python": "3.x",
            "numpy": "2.x",
            "numpy_quaternion": "2024.x",
            "scipy": "1.x",
            "pytest_benchmark": "5.x",
            "platform": "test machine",
        },
        "settings": {"min_rounds": 9, "max_time": 0.2},
        "groups": {
            "hot rotate": {
                "winner": "tuple",
                "co_winners": ["tuple"],
                "results": [
                    {
                        "backend": "tuple",
                        "median_us": 2.0,
                        "relative_to_winner": 1.0,
                        "within_one_percent_of_winner": True,
                        "rounds": 9,
                    }
                ],
            }
        },
    }

    _print_report(report)

    output = capsys.readouterr().out
    assert "hot rotate — WINNER: tuple" in output
    assert "Backends within 1%" in output
