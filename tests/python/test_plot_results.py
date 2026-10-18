"""Tests for nebula.imaging.plot_results and the nebula plot command."""

import numpy as np
import pytest
from nebula.cli import main
from nebula.imaging import plot_results
from safetensors.numpy import save_file

RUN = """
name = "test"

[beam]
energy = 1000

[detectors.beta]
energy = [50, inf]
azimuth = [0, 90]

[detectors.alpha]
energy = [0, 50]
"""


@pytest.fixture(autouse=True)
def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def write_results(path, *, zmap=True, metadata=True):
    rng = np.random.default_rng(0)
    alpha = rng.uniform(0, 1, (4, 3)).astype(np.float32)
    alpha[0, 0] = np.nan  # a pixel without primaries
    alpha[1, 1] = 100  # an outlier
    tensors = {
        "alpha": alpha,
        "beta": rng.uniform(0, 0.1, (4, 3)).astype(np.float32),
        "primaries": np.full((4, 3), 10, dtype=np.int32),
        "x": np.array([-3, -1, 1, 3], dtype=np.float32),
        "y": np.array([-2, 0, 2], dtype=np.float32),
    }
    if zmap:
        tensors["zmap"] = np.linspace(0, 20, 12, dtype=np.float32).reshape(4, 3)
    save_file(
        tensors,
        str(path),
        metadata={"name": "test", "config": RUN} if metadata else None,
    )
    return path


def images(fig):
    return [ax.images[0] for ax in fig.axes if ax.images]


def test_panels_in_run_file_order(tmp_path):
    fig = plot_results(write_results(tmp_path / "r.safetensors"))
    shown = images(fig)
    assert len(shown) == 3
    titles = [image.axes.get_title() for image in shown]
    assert titles[0] == "surface height"
    assert titles[1].startswith("beta: mean") and "azimuth 0–90°" in titles[1]
    assert titles[2].startswith("alpha: mean") and "energy 0–50 eV" in titles[2]
    assert shown[0].get_cmap().name.startswith(plot_results.__kwdefaults__["zmap_cmap"])
    assert shown[1].get_cmap().name.startswith("gray")
    # Extent: pixel edges from the pixel centres
    assert shown[1].get_extent() == [-4, 4, -3, 3]
    assert (
        fig._suptitle.get_text()
        == "test, 1000 eV, 4 x 3 pixels of 2 nm, ≈10 primaries per pixel"
    )


def test_colour_limits(tmp_path):
    fig = plot_results(write_results(tmp_path / "r.safetensors"), percentile=90)
    zmap, beta, alpha = images(fig)
    assert zmap.get_clim() == (0, 20)  # full range for the z-map
    # Yield scales start at 0 (beta's values are all above 0) ...
    assert alpha.get_clim()[0] == 0 and beta.get_clim()[0] == 0
    # ... and end at the percentile, so the outlier does not set the scale
    assert alpha.get_clim()[1] < 100
    full = images(plot_results(tmp_path / "r.safetensors", percentile=100))
    assert full[2].get_clim() == (0, pytest.approx(100))  # alpha, up to the outlier
    assert alpha.get_array().mask[0, 0]  # NaN is masked (shown in the bad colour)


def test_without_zmap_and_metadata(tmp_path):
    fig = plot_results(
        write_results(tmp_path / "r.safetensors", zmap=False, metadata=False)
    )
    titles = [image.axes.get_title() for image in images(fig)]
    assert [t.split(":")[0] for t in titles] == ["alpha", "beta"]  # alphabetical


def test_select_detectors(tmp_path):
    path = write_results(tmp_path / "r.safetensors")
    titles = [
        i.axes.get_title() for i in images(plot_results(path, detectors=["beta"]))
    ]
    assert len(titles) == 2 and titles[1].startswith("beta")
    with pytest.raises(ValueError, match="alpha, beta"):
        plot_results(path, detectors=["se"])


def test_plot_command(tmp_path, capsys):
    path = write_results(tmp_path / "r.safetensors")
    assert main(["plot", str(path)]) == 0
    assert (tmp_path / "r-plot.png").stat().st_size > 0
    assert not (tmp_path / "r.png").exists()
    assert (
        main(
            ["plot", str(path), "-o", str(tmp_path / "other.png"), "--cmap", "viridis"]
        )
        == 0
    )
    assert (tmp_path / "other.png").is_file()
    assert main(["plot", str(tmp_path / "missing.safetensors")]) == 1
    assert "nebula plot" in capsys.readouterr().err
