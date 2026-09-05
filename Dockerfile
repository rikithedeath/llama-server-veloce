# Immagine sottile sopra a llama.cpp: cambia solo il modo in cui scende il modello.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# serve a collegare il package GHCR al repository, altrimenti la visibilita' resta grigia
LABEL org.opencontainers.image.source="https://github.com/rikithedeath/llama-server-veloce"
LABEL org.opencontainers.image.description="llama.cpp server-cuda con download del modello a connessioni parallele"

COPY scarica-e-avvia.py /app/scarica-e-avvia.py

# python3 c'e' gia' nell'immagine di partenza: non si installa niente
ENTRYPOINT ["/usr/bin/python3", "/app/scarica-e-avvia.py"]
