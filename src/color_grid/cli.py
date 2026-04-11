from pathlib import Path

import click
from PIL import Image

from .grid import image_to_cell_colors
from .quantize import quantize_cells
from .render import render_page, render_solution


@click.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--width", "-w", type=int, required=True, help="Grid width in cells.")
@click.option("--height", "-h", type=int, required=True, help="Grid height in cells.")
@click.option("--colors", "-c", type=int, required=True, help="Number of palette colors.")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output PNG path (default: <image>_grid.png next to the input).",
)
@click.option(
    "--solution/--no-solution",
    default=False,
    help="Also write a filled-in preview image alongside the page.",
)
def main(
    image_path: Path,
    width: int,
    height: int,
    colors: int,
    output: Path | None,
    solution: bool,
) -> None:
    """Generate a color-by-number grid page from IMAGE_PATH."""
    if output is None:
        output = image_path.with_name(f"{image_path.stem}_grid.png")

    image = Image.open(image_path)
    cells = image_to_cell_colors(image, width, height)
    labels, palette = quantize_cells(cells, colors)

    page = render_page(labels, palette)
    page.save(output)
    click.echo(f"wrote {output}")

    if solution:
        sol_path = output.with_name(f"{output.stem}_solution.png")
        render_solution(labels, palette).save(sol_path)
        click.echo(f"wrote {sol_path}")


if __name__ == "__main__":
    main()
