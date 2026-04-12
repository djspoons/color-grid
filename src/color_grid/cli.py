from pathlib import Path

import click
from PIL import Image

from .grid import image_to_cell_colors
from .palette import load_palette
from .quantize import quantize_cells
from .render import PAPER_SIZES_INCHES, PageSpec, render_page, render_solution, save_page


@click.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--width", "-w", type=int, required=True, help="Grid width in cells.")
@click.option("--height", "-h", type=int, required=True, help="Grid height in cells.")
@click.option(
    "--colors",
    "-c",
    type=int,
    default=None,
    help="Max number of colors. Required without --palette; with --palette defaults to all.",
)
@click.option(
    "--palette",
    "palette_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Use a fixed palette JSON file (see palettes/). Use --colors to limit.",
)
@click.option(
    "--color-space",
    type=click.Choice(["rgb", "lab", "ciecam16"], case_sensitive=False),
    default="lab",
    show_default=True,
    help=(
        "Color space used for clustering and palette matching. "
        "lab is a good default; ciecam16 (CIECAM16-UCS) is the most "
        "perceptually accurate but slower and requires colour-science."
    ),
)
@click.option(
    "--method",
    type=click.Choice(["kmeans", "maxcoverage"], case_sensitive=False),
    default="maxcoverage",
    show_default=True,
    help=(
        "kmeans minimizes average error but loses rare vivid colors; "
        "maxcoverage uses farthest-first selection to preserve the full color range."
    ),
)
@click.option(
    "--paper",
    type=click.Choice(sorted(PAPER_SIZES_INCHES.keys()), case_sensitive=False),
    default="letter",
    show_default=True,
    help="Paper size.",
)
@click.option(
    "--margin",
    type=float,
    default=0.5,
    show_default=True,
    help="Page margin in inches.",
)
@click.option(
    "--line-width",
    type=float,
    default=None,
    help="Grid line width in points. Defaults to auto based on cell size.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output path (default: <image>_grid.pdf next to the input). Use .svg for SVG.",
)
@click.option(
    "--solution/--no-solution",
    default=False,
    help="Also write a filled-in preview alongside the page.",
)
def main(
    image_path: Path,
    width: int,
    height: int,
    colors: int | None,
    palette_path: Path | None,
    color_space: str,
    method: str,
    paper: str,
    margin: float,
    line_width: float | None,
    output: Path | None,
    solution: bool,
) -> None:
    """Generate a printable color-by-number grid page from IMAGE_PATH.

    Specify --colors for the number of colors, and/or --palette to snap to a
    fixed palette. With --palette, --colors limits how many palette entries to
    use (defaults to all).
    """
    if colors is None and palette_path is None:
        raise click.UsageError("Provide --colors/-c, --palette, or both.")

    if output is None:
        output = image_path.with_name(f"{image_path.stem}_grid.pdf")

    fmt = "svg" if output.suffix.lower() == ".svg" else "pdf"
    page_spec = PageSpec(paper=paper.lower(), margin_in=margin)

    fixed_palette = None
    palette_codes: list[str] | None = None
    if palette_path is not None:
        pal = load_palette(palette_path)
        fixed_palette = pal.rgb
        palette_codes = pal.codes
        if colors is None:
            colors = len(fixed_palette)
        elif colors > len(fixed_palette):
            raise click.BadParameter(
                f"palette has {len(fixed_palette)} colors but --colors={colors}"
            )

    image = Image.open(image_path)
    cells = image_to_cell_colors(image, width, height)
    labels, palette, chosen_indices = quantize_cells(
        cells,
        colors,
        color_space=color_space.lower(),
        method=method.lower(),
        fixed_palette=fixed_palette,
    )

    entry_labels = None
    if chosen_indices is not None and palette_codes is not None:
        entry_labels = [palette_codes[int(i)] for i in chosen_indices]

    data = render_page(labels, palette, page_spec, entry_labels=entry_labels, fmt=fmt, line_width=line_width)
    save_page(data, output)
    click.echo(f"wrote {output}")

    if solution:
        sol_path = output.with_name(f"{output.stem}_solution{output.suffix}")
        sol_data = render_solution(labels, palette, page_spec, fmt=fmt, line_width=line_width)
        save_page(sol_data, sol_path)
        click.echo(f"wrote {sol_path}")


if __name__ == "__main__":
    main()
