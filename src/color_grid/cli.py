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
@click.option("--colors", "-c", type=int, required=True, help="Number of palette colors.")
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
    "--palette",
    "palette_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Snap output colors to entries in a fixed palette JSON file (see color-sets/).",
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
@click.option("--dpi", type=int, default=300, show_default=True, help="Output resolution.")
@click.option(
    "--margin",
    type=float,
    default=0.5,
    show_default=True,
    help="Page margin in inches.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output path (default: <image>_grid.pdf next to the input). Use .png for an image.",
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
    colors: int,
    color_space: str,
    method: str,
    palette_path: Path | None,
    paper: str,
    dpi: int,
    margin: float,
    output: Path | None,
    solution: bool,
) -> None:
    """Generate a printable color-by-number grid page from IMAGE_PATH."""
    if output is None:
        output = image_path.with_name(f"{image_path.stem}_grid.pdf")

    page_spec = PageSpec(paper=paper.lower(), dpi=dpi, margin_in=margin)

    fixed_palette = None
    palette_codes: list[str] | None = None
    if palette_path is not None:
        pal = load_palette(palette_path)
        fixed_palette = pal.rgb
        palette_codes = pal.codes
        if len(fixed_palette) < colors:
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

    page = render_page(labels, palette, page_spec, entry_labels=entry_labels)
    save_page(page, output, page_spec)
    click.echo(f"wrote {output}")

    if solution:
        sol_path = output.with_name(f"{output.stem}_solution{output.suffix}")
        save_page(render_solution(labels, palette, page_spec), sol_path, page_spec)
        click.echo(f"wrote {sol_path}")


if __name__ == "__main__":
    main()
