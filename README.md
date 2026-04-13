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

Image tags are auto-generated: `YYYYMMDD-<git-hash>` if the working tree is
clean, `YYYYMMDD-development` if it has uncommitted changes. The version is
displayed in the app footer.

Build and run locally (native platform):

```
docker compose up -d --build
```

The app runs at http://localhost:8000. To stop:

```
docker compose down
```

Cross-compile for a linux/amd64 server:

```
TAG=$(date +%Y%m%d)-$(git diff --quiet && git rev-parse --short HEAD || echo development)
docker buildx build --platform linux/amd64 --build-arg APP_VERSION=$TAG -t colorgrid:$TAG .
```

Then push/transfer to your server and run:

```
docker run -d -p 8000:8000 --restart unless-stopped colorgrid:$TAG
```

## Tests

```
.venv/bin/pip install pytest
.venv/bin/pytest
```
