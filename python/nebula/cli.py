"""The nebula command: run a simulation defined in a TOML file, or plot its results.

Usage:

- ``nebula RUN.toml [-o OUTPUT] [--seed N] [--cpu | --gpu] [--keep DIR] [-q]``
  runs the simulation, printing what it runs and its progress on stderr. See `nebula.config` for the file format; the yields per
  detector per pixel, and the height of the highest surface at each beam
  position, are written to a safetensors file (see
  `nebula.simulate.save_results`) and plotted as NAME-plot.png next to it,
  with NAME the name of the results file without extension (``--no-plot``
  skips the plot).
- ``nebula plot RESULTS.safetensors [-o FIGURE] [--detectors NAME ...]
  [--cmap NAME] [--percentile HIGH] [--dpi N]``
  plots such a file (see `nebula.imaging.plot_results`), by default as
  RESULTS-plot.png next to it.
"""

import argparse
import dataclasses
import shutil
import sys
from pathlib import Path

import numpy as np

from .config import ConfigError, load_config
from .simulate import run, save_results

PLOT_DESCRIPTION = """\
Plot the results of a run: the height of the highest surface at each beam
position (if the file has it) and the yield map of every detector in the
file, whatever their names.

Detectors are shown in the order of the run file stored in the safetensors
file, each titled with its mean yield and filters. Yield maps are greyscale
(black low, white high, like SEM images), each with its own colour bar that
starts at 0 and ends at the 99th percentile of its values, so a few noisy
pixels do not set the scale. The z-map uses the viridis colour map and its full range.
Pixels without primary electrons are red."""

PLOT_EPILOG = """\
examples:
  nebula plot examples/z-map/cell65.safetensors
      saves examples/z-map/cell65-plot.png
  nebula plot cell65.safetensors -o figure.pdf --detectors se bse
  nebula plot cell65.safetensors --cmap viridis --percentile 100"""

MAIN_USAGE = """\
nebula RUN.toml [-o OUTPUT] [--seed SEED] [--cpu | --gpu] [--keep DIR] [--no-plot] [-q]
       nebula plot RESULTS.safetensors [-o FIGURE] [options]"""

MAIN_DESCRIPTION = """\
Run a Nebula simulation defined in a TOML file: make (or read) the geometry
and primary electrons, run nebula_gpu (or nebula_cpu_mt without a GPU), apply the
detectors of the run file, save the yield per detector per pixel (and the
height of the highest surface at each beam position) as a safetensors file,
and plot them as NAME-plot.png next to it.

commands:
  nebula RUN.toml                    run a simulation (options below)
  nebula plot RESULTS.safetensors    plot its results; see 'nebula plot -h'"""

MAIN_EPILOG = """\
examples:
  nebula examples/z-map/cell65.toml
      saves examples/z-map/cell65.safetensors and examples/z-map/cell65-plot.png
  nebula examples/line-space/line-space-00.toml
      the same with a ready-made triangle file as geometry
  nebula examples/z-map/cell65.toml --cpu --seed 2 -o run2.safetensors
      saves run2.safetensors and run2-plot.png
  nebula plot examples/z-map/cell65.safetensors --detectors se bse
      plots again, with other options

See the documentation of nebula.config for the format of run files."""


class _Console:
    """Information lines and a progress bar for Nebula on stderr, for 'nebula RUN.toml'.

    The bar (tqdm) appears when Nebula first reports progress and is closed
    before the next information line. tqdm leaves it out when stderr is not a
    terminal; with ``quiet`` nothing is printed at all.
    """

    def __init__(self, *, quiet: bool = False, file=None, disable: bool | None = None):
        self.quiet = quiet
        self.file = file if file is not None else sys.stderr
        self.disable = disable
        self.bar = None

    def report(self, text: str) -> None:
        self.close()
        if not self.quiet:
            print(text, file=self.file, flush=True)

    def progress(self, fraction: float | None) -> None:
        if self.quiet:
            return
        # Nebula counts the primaries it has started; the last ones still run
        description = (
            "pre-scan"
            if fraction is None
            else "simulating"
            if fraction < 1
            else "finishing"
        )
        if self.bar is None:
            from tqdm import tqdm

            size = shutil.get_terminal_size()
            self.bar = tqdm(
                total=100,
                desc=description,
                file=self.file,
                disable=self.disable,
                # Terminals that report no size get 80 x 24, else tqdm hides the
                # bar; on wide terminals the bar stays 80 columns at most
                ncols=min(size.columns, 80),
                nrows=size.lines,
                bar_format="{l_bar}{bar}| [{elapsed}<{remaining}]",
            )
        # Nebula reports once a second, so show every report
        self.bar.set_description_str(description, refresh=False)
        if fraction is not None:
            self.bar.update(100 * fraction - self.bar.n)
        self.bar.refresh()

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def __enter__(self):
        return self

    def __exit__(self, *exception) -> None:
        self.close()


def _save_plot(
    results: Path, output: Path | None = None, dpi: int = 100, **options
) -> Path:
    """Plot a results file with `plot_results` and save the figure.

    By default the figure is NAME-plot.png next to the results, with NAME
    their file name without extension.
    """
    import matplotlib

    matplotlib.use("Agg")  # no window, and no display needed
    import matplotlib.pyplot as plt

    from .imaging import plot_results

    figure = plot_results(results, **options)
    output = output or results.with_name(f"{results.stem}-plot.png")
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def plot(argv: list[str]) -> int:
    """The 'nebula plot' command."""
    parser = argparse.ArgumentParser(
        prog="nebula plot",
        description=PLOT_DESCRIPTION,
        epilog=PLOT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results",
        type=Path,
        metavar="RESULTS.safetensors",
        help="file written by 'nebula RUN.toml'",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FIGURE",
        help="figure file; its extension sets the format, e.g. .png, .pdf, .svg "
        "(default: NAME-plot.png next to the results, with NAME their file "
        "name without extension)",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        metavar="NAME",
        help="detectors to show, in this order (default: all)",
    )
    parser.add_argument(
        "--cmap",
        default="gray",
        metavar="NAME",
        help="matplotlib colour map of the yield maps (default: gray)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        metavar="HIGH",
        help="percentile of each yield map at which its colour scale ends; it "
        "starts at 0 (default: 99; 100 for the maximum)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="resolution of raster figures (default: 100)",
    )
    args = parser.parse_args(argv)

    try:
        output = _save_plot(
            args.results,
            args.output,
            dpi=args.dpi,
            detectors=args.detectors,
            cmap=args.cmap,
            percentile=args.percentile,
        )
    except (ValueError, FileNotFoundError, OSError) as error:
        print(f"nebula plot: {error}", file=sys.stderr)
        return 1
    print(f"saved {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["plot"]:
        return plot(argv[1:])

    parser = argparse.ArgumentParser(
        prog="nebula",
        usage=MAIN_USAGE,
        description=MAIN_DESCRIPTION,
        epilog=MAIN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config", type=Path, metavar="RUN.toml", help="TOML file describing the run"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="results file (default: NAME.safetensors next to the run file, "
        "with NAME from the run file)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed, overriding the one in the file (default: the file's, "
        "or a random one if it has none)",
    )
    device = parser.add_mutually_exclusive_group()
    device.add_argument(
        "--cpu", action="store_true", help="use nebula_cpu_mt, even if there is a GPU"
    )
    device.add_argument(
        "--gpu",
        action="store_true",
        help="use nebula_gpu (default: when there is a GPU)",
    )
    parser.add_argument(
        "--keep",
        type=Path,
        metavar="DIR",
        help="keep Nebula's input files, raw output and log in DIR",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="do not plot the results (default: plot them as NAME-plot.png next "
        "to the results file)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="do not print information and progress while running, only the "
        "summary at the end",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.seed is not None:
            config = dataclasses.replace(config, seed=args.seed, random_seed=False)
        gpu = True if args.gpu else False if args.cpu else None
        with _Console(quiet=args.quiet) as console:
            result = run(
                config,
                gpu=gpu,
                workdir=args.keep,
                report=console.report,
                progress=console.progress,
            )
    except (ConfigError, ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"nebula: {error}", file=sys.stderr)
        return 1

    output = args.output or args.config.parent / f"{config.name}.safetensors"
    save_results(output, config, result)

    print(
        f"{result.binary.name}: {result.primaries.sum()} primaries, "
        f"{result.detected} detected, {result.seconds:.1f} s"
    )
    for name, image in result.images.items():
        print(f"  {name}: mean yield {np.nanmean(image):.4f}")
    print(f"saved {output}")

    if not args.no_plot:
        # The results are saved, so a failing plot only gets a warning
        try:
            print(f"saved {_save_plot(output)}")
        except Exception as error:  # noqa: BLE001 - any plotting problem
            print(
                f"nebula: warning: could not plot the results: {error}", file=sys.stderr
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
