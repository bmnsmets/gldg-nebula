"""Runs a small simulation through the nebula command on the CPU."""

import numpy as np
import pytest
from nebula import runner
from nebula.cli import main
from nebula.geometry import triangulate_zmap
from nebula.io.triangles import write_tri
from PIL import Image
from safetensors import safe_open
from safetensors.numpy import load_file


def have_nebula():
    try:
        runner.find_binary(gpu=False)
        runner.find_material("silicon")
    except FileNotFoundError:
        return False
    return True


pytestmark = [
    pytest.mark.requires_nebula,
    pytest.mark.skipif(
        not have_nebula(), reason="nebula_cpu_mt or silicon.mat not built"
    ),
]

RUN = """
name = "small"
seed = 3

[geometry.zmap]
file = "line.png"
z_max = 20
z_terminator = -100
materials = ["silicon"]

[beam]
energy = 500
z = 25
avg_electrons = 20
beam_std = 0.5
x = [-12, 12]
y = [-4, 4]
step = 4

[detectors.se]
energy = [0, 50]

[detectors.bse]
energy = [50, inf]

[detectors.bse_left]
energy = [50, inf]
azimuth = [90, 270]
"""


def test_small_run(tmp_path, capsys):
    pixels = np.zeros((32, 32), dtype=np.uint8)
    pixels[:, 12:20] = 255  # a line along y, 8 nm wide
    Image.fromarray(pixels).save(tmp_path / "line.png")
    (tmp_path / "small.toml").write_text(RUN)

    assert main([str(tmp_path / "small.toml"), "--cpu", "--plot"]) == 0
    captured = capsys.readouterr()
    out = captured.out
    assert "nebula_cpu_mt" in out
    # What is run is printed on stderr before it runs; no progress bar, as
    # stderr is not a terminal
    assert "small: 6 x 2 pixels of 4 nm" in captured.err
    assert "geometry:" in captured.err and "1 material" in captured.err
    assert "running nebula_cpu_mt" in captured.err
    assert "%|" not in captured.err
    # The results are plotted next to them
    figure = tmp_path / "small-plot.png"
    assert figure.stat().st_size > 0 and f"saved {figure}" in out

    output = tmp_path / "small.safetensors"
    tensors = load_file(output)
    assert set(tensors) == {"se", "bse", "bse_left", "primaries", "x", "y", "zmap"}
    # x = [-12, 12], y = [-4, 4] are the edges of the scan; pixels of 4 nm
    np.testing.assert_allclose(tensors["x"], [-10, -6, -2, 2, 6, 10])
    np.testing.assert_allclose(tensors["y"], [-2, 2])
    for name in ["se", "bse", "bse_left"]:
        assert tensors[name].shape == (6, 2)
        assert np.all(np.isfinite(tensors[name]))
    assert 0.1 < np.mean(tensors["se"]) < 3
    assert 0.05 < np.mean(tensors["bse"]) < 1
    assert np.all(tensors["bse_left"] <= tensors["bse"])
    # Height of the surface at the beam positions: the 8 nm wide, 20 nm high
    # line at |x| < 4 nm (within the 0.2 nm tolerance), 0 beside it
    zmap = tensors["zmap"]
    assert zmap.shape == (6, 2)
    on_line = np.abs(tensors["x"]) < 4
    np.testing.assert_allclose(zmap[on_line], 20, atol=0.2 + 1e-4)
    np.testing.assert_allclose(zmap[~on_line], 0, atol=0.2 + 1e-4)
    with safe_open(output, "numpy") as f:
        metadata = f.metadata()
    assert metadata["config"] == RUN and metadata["binary"] == "nebula_cpu_mt"


def test_default_scan(tmp_path):
    # Without x, y and step, the beam scans the z-map's own pixels
    pixels = np.zeros((32, 32), dtype=np.uint8)
    pixels[:, 12:20] = 255
    Image.fromarray(pixels).save(tmp_path / "line.png")
    run = "\n".join(
        line
        for line in RUN.splitlines()
        if not line.startswith(("x =", "y =", "step ="))
    )
    run = run.replace("avg_electrons = 20", "avg_electrons = 1")
    (tmp_path / "small.toml").write_text(run)
    output = tmp_path / "out" / "default.safetensors"
    output.parent.mkdir()
    assert (
        main([str(tmp_path / "small.toml"), "--cpu", "-o", str(output), "--plot"]) == 0
    )
    assert (tmp_path / "out" / "default-plot.png").is_file()
    assert main([str(tmp_path / "small.toml"), "--cpu"]) == 0
    assert not (tmp_path / "small-plot.png").exists()
    tensors = load_file(output)
    assert tensors["se"].shape == tensors["zmap"].shape == (32, 32)
    np.testing.assert_allclose(tensors["x"], np.arange(32) - 15.5)
    np.testing.assert_allclose(tensors["y"], np.arange(32) - 15.5)


def test_tri_geometry(tmp_path, capsys):
    # The same line, from a triangle file instead of a z-map
    heights = np.zeros((32, 32))  # [ix, iy]
    heights[12:20, :] = 20
    write_tri(
        tmp_path / "line.tri",
        triangulate_zmap(heights, terminator_z=-100, detector_z=30),
    )
    run = RUN.replace(
        """[geometry.zmap]
file = "line.png"
z_max = 20
z_terminator = -100
materials = ["silicon"]""",
        """[geometry]
file = "line.tri"
materials = ["silicon"]""",
    )
    (tmp_path / "small.toml").write_text(run)
    assert main([str(tmp_path / "small.toml"), "--cpu"]) == 0
    assert "from line.tri" in capsys.readouterr().err
    tensors = load_file(tmp_path / "small.safetensors")
    assert tensors["se"].shape == tensors["zmap"].shape == (6, 2)
    on_line = np.abs(tensors["x"]) < 4
    np.testing.assert_allclose(tensors["zmap"][on_line], 20, atol=1e-4)
    np.testing.assert_allclose(tensors["zmap"][~on_line], 0, atol=1e-4)


def test_invalid_run(tmp_path, capsys):
    (tmp_path / "bad.toml").write_text('name = "bad"')
    assert main([str(tmp_path / "bad.toml")]) == 1
    assert "geometry" in capsys.readouterr().err
