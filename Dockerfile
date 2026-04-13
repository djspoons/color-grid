FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY palettes/ palettes/

RUN pip install --no-cache-dir '.[web]'

ARG APP_VERSION=development
ENV COLORGRID_PALETTES_DIR=/app/palettes
ENV COLORGRID_HOST=0.0.0.0
ENV COLORGRID_PORT=8000
ENV APP_VERSION=${APP_VERSION}

EXPOSE 8000

CMD ["colorgrid-web"]
