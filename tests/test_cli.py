from __future__ import annotations

from diagnostic_suite.cli import parse_args


def test_parse_default_args() -> None:
    args = parse_args([])
    assert args.tasks == ["toy2d"]
    assert args.models == ["fno", "unet", "resnet"]


def test_parse_make_toy() -> None:
    args = parse_args(["--make-toy", "--data-root", "mydata", "--toy-grid", "16"])
    assert args.make_toy is True
    assert str(args.data_root) == "mydata"
    assert args.toy_grid == 16


def test_parse_reproducibility_controls() -> None:
    args = parse_args(
        [
            "--seeds",
            "1",
            "2",
            "3",
            "--eval-horizons",
            "1",
            "4",
            "--scaling-factor",
            "1.5",
            "--resume",
        ]
    )
    assert args.seeds == [1, 2, 3]
    assert args.eval_horizons == [1, 4]
    assert args.scaling_factor == 1.5
    assert args.resume is True
