# Color Grid

A tool for generating color-by-number grid pages from images.

## Install

```
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Usage

```
colorgrid path/to/photo.jpg --width 24 --height 24 --colors 12
colorgrid photo.jpg -w 24 -h 24 -c 12 --paper a4 --solution
colorgrid photo.jpg -w 24 -h 24 -c 12 -o out.png
```

Outputs a printable PDF (default letter, 300 dpi) with a numbered grid and color legend. Paper options: letter, legal, a4, a5. Use a `.png` output path for a raster image instead of PDF. Pass `--solution` to also emit a filled-in preview.

## Develop

```
.venv/bin/pip install pytest
.venv/bin/pytest
```
