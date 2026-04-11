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
colorgrid photo.jpg -w 24 -h 24 -c 12 -o out.png --solution
```

Outputs a PNG with a numbered grid and color legend. Pass `--solution` to also emit a filled-in preview.

## Develop

```
.venv/bin/pip install pytest
.venv/bin/pytest
```
