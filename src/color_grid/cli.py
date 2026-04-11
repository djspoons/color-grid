from pathlib import Path

import click
from PIL import Image

from .grid import image_to_cell_colors
from .quantize import quantize_cells
from .render import PAPER_SIZES_INCHES, PageSpec, render_page, render_solution, save_page


@click.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--width", "-w", type=int, required=True, help="Grid width in cells.")
@click.option("--height", "-h", type=int, required=True, help="Grid height in cells.")
@click.option("--colors", "-c", type=int, required=True, help="Number of palette colors.")
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

    image = Image.open(image_path)
    cells = image_to_cell_colors(image, width, height)
    labels, palette = quantize_cells(cells, colors)

    page = render_page(labels, palette, page_spec)
    save_page(page, output, page_spec)
    click.echo(f"wrote {output}")

    if solution:
        sol_path = output.with_name(f"{output.stem}_solution{output.suffix}")
        save_page(render_solution(labels, palette, page_spec), sol_path, page_spec)
        click.echo(f"wrote {sol_path}")


if __name__ == "__main__":
    main()
