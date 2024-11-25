FROM golang:1.23-bullseye AS builder

WORKDIR /app

COPY go.mod go.sum ./

RUN go mod download

COPY . .

RUN CGO_ENABLED=1 GOOS=linux go build -o /app/server

FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false appuser

WORKDIR /app

RUN mkdir -p /app/models /app/static /app/cache && \
    chown -R appuser:appuser /app

COPY --from=builder /app/server /app/

RUN curl -L -o /app/models/yolo11x.onnx https://github.com/joeychilson/detect/raw/refs/heads/master/models/yolo11x.onnx

COPY static/ /app/static/

ENV ONNX_CACHE_PATH=/app/cache

USER appuser

EXPOSE 8080

CMD ["/app/server"]
