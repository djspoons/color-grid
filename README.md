# Color Grid

A tool for generating color-by-number grid pages from images.

## Install

```
python3 -m venv .venv
.venv/bin/pip install -e .
```

## CLI Usage

```
colorgrid photo.jpg -w 24 -h 24 -c 12
colorgrid photo.jpg -w 24 -h 24 -c 12 --paper a4 --solution
colorgrid photo.jpg -w 24 -h 24 --palette palettes/crayola-sample.json
colorgrid photo.jpg -w 24 -h 24 -c 12 -o out.svg
```

Outputs a vector PDF (default letter size) with a numbered grid and color legend. Use `--colors`/`-c` for auto-quantized colors, or `--palette` to use all colors from a palette file (mutually exclusive). Paper options: letter, legal, a4, a5. Use a `.svg` output path for SVG. Pass `--solution` to also emit a filled-in preview.

## Web App

### Development

Install with web dependencies and start the dev server:

```
.venv/bin/pip install -e '.[web]'
colorgrid-web
```

The app runs at http://127.0.0.1:8000. For auto-reload during development:

```
python -m uvicorn color_grid.web.app:app --host 127.0.0.1 --port 8000 --reload
```

Stop with Ctrl-C.

### Docker

Build and run:

```
docker compose up -d --build
```

The app runs at http://localhost:8000. To stop:

```
docker compose down
```

## Tests

```
.venv/bin/pip install pytest
.venv/bin/pytest
```
