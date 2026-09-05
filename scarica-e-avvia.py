#!/usr/bin/env python3
"""Scarica il GGUF a connessioni parallele, poi cede il posto a llama-server.

Il downloader interno di llama.cpp ne apre una sola: su un host Runpod sono ~11 MB/s
contro i ~95 MB/s misurati con otto range-request, cioè quaranta minuti di GPU pagata
invece di tre. Qui il file scende in parallelo e poi si fa `exec`, così il processo
resta uno solo e llama-server continua a essere il PID 1 del container.

Senza MODELLO_REPO/MODELLO_URL l'immagine si comporta esattamente come quella di
partenza, argomenti e variabili LLAMA_ARG_* comprese.
"""
import os
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request

SERVER = "/app/llama-server"

REPO = os.environ.get("MODELLO_REPO")
FILE = os.environ.get("MODELLO_FILE")
URL = os.environ.get("MODELLO_URL")
DEST = os.environ.get("MODELLO_PATH", "/root/modello.gguf")
REVISIONE = os.environ.get("MODELLO_REVISIONE", "main")
CONNESSIONI = int(os.environ.get("MODELLO_CONNESSIONI", "8"))
PEZZO = int(os.environ.get("MODELLO_PEZZO_MB", "64")) * 1024 * 1024
TENTATIVI = int(os.environ.get("MODELLO_TENTATIVI", "5"))


def log(messaggio):
    # su una riga sola e con flush: i log di Runpod non mostrano il progresso scritto con \r
    print(f"[scarica] {messaggio}", flush=True)


def avvia():
    """Passa il controllo a llama-server, senza restare in mezzo come processo padre."""
    os.execve(SERVER, [SERVER] + sys.argv[1:], os.environ)


def intestazioni():
    testa = {"User-Agent": "scarica-e-avvia/1"}
    gettone = os.environ.get("HF_TOKEN")
    if gettone:
        testa["Authorization"] = f"Bearer {gettone}"
    return testa


def dimensione_e_url(url):
    """Chiede il primo byte: risponde la dimensione totale e l'URL finale del CDN."""
    richiesta = urllib.request.Request(url, headers={**intestazioni(), "Range": "bytes=0-0"})
    with urllib.request.urlopen(richiesta, timeout=60) as r:
        intervallo = r.headers.get("Content-Range")
        if not intervallo:
            raise RuntimeError(f"il server non accetta le range-request: {r.headers}")
        return int(intervallo.split("/")[-1]), r.url


def operaio(url, coda, fd, stato, blocco):
    while True:
        try:
            inizio, lunghezza = coda.get_nowait()
        except queue.Empty:
            return
        fine = inizio + lunghezza - 1
        for tentativo in range(TENTATIVI):
            posizione = inizio
            try:
                richiesta = urllib.request.Request(
                    url, headers={**intestazioni(), "Range": f"bytes={inizio}-{fine}"}
                )
                with urllib.request.urlopen(richiesta, timeout=120) as r:
                    while posizione <= fine:
                        dati = r.read(1 << 20)
                        if not dati:
                            break
                        os.pwrite(fd, dati, posizione)
                        posizione += len(dati)
                        with blocco:
                            stato["byte"] += len(dati)
                if posizione != fine + 1:
                    raise RuntimeError(f"pezzo incompleto: {posizione} invece di {fine + 1}")
                with blocco:
                    stato["pezzi"] += 1
                break
            except Exception as errore:
                # una connessione caduta su sedici giga e' normale: si ripete solo quel pezzo
                with blocco:
                    stato["byte"] -= posizione - inizio
                if tentativo == TENTATIVI - 1:
                    log(f"pezzo a {inizio} fallito dopo {TENTATIVI} tentativi: {errore}")
                    return
                time.sleep(2 * (tentativo + 1))


def scarica(url, totale, dest):
    coda = queue.Queue()
    for inizio in range(0, totale, PEZZO):
        coda.put((inizio, min(PEZZO, totale - inizio)))
    attesi = coda.qsize()

    fd = os.open(dest + ".parziale", os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        # il file nasce gia' della dimensione finale: i thread scrivono ognuno al proprio offset
        os.ftruncate(fd, totale)
        stato, blocco = {"byte": 0, "pezzi": 0}, threading.Lock()
        partenza = time.time()
        operai = [
            threading.Thread(target=operaio, args=(url, coda, fd, stato, blocco), daemon=True)
            for _ in range(CONNESSIONI)
        ]
        for o in operai:
            o.start()

        while any(o.is_alive() for o in operai):
            time.sleep(10)
            with blocco:
                fatti = stato["byte"]
            trascorso = max(time.time() - partenza, 1e-6)
            log(
                f"{fatti / 1e9:.1f}/{totale / 1e9:.1f} GB "
                f"({100 * fatti / totale:.0f}%) a {fatti / trascorso / 1e6:.0f} MB/s"
            )

        for o in operai:
            o.join()
    finally:
        os.close(fd)

    # la dimensione non prova niente, il file e' preallocato: contano i pezzi completati
    if stato["pezzi"] != attesi:
        raise RuntimeError(f"completati {stato['pezzi']} pezzi su {attesi}")
    os.rename(dest + ".parziale", dest)


def parti(nome):
    """I GGUF grossi arrivano in piu' file: da -00001-of-00003 si ricavano tutti gli altri."""
    trovato = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", nome)
    if not trovato:
        return [nome]
    quante = int(trovato.group(2))
    radice = nome[: trovato.start()]
    return [f"{radice}-{i:05d}-of-{quante:05d}.gguf" for i in range(1, quante + 1)]


def porta_giu(url, dest):
    totale, url_finale = dimensione_e_url(url)
    if os.path.exists(dest) and os.path.getsize(dest) == totale:
        log(f"{dest} c'e' gia' ed e' della dimensione giusta, non riscarico")
        return
    log(f"{totale / 1e9:.1f} GB da {url} con {CONNESSIONI} connessioni")
    partenza = time.time()
    scarica(url_finale, totale, dest)
    durata = time.time() - partenza
    log(f"finito in {durata / 60:.1f} minuti, media {totale / durata / 1e6:.0f} MB/s")


def main():
    if not REPO and not URL:
        avvia()

    if URL:
        porta_giu(URL, DEST)
        modello = DEST
    else:
        pezzi = parti(FILE)
        if len(pezzi) == 1:
            porta_giu(f"https://huggingface.co/{REPO}/resolve/{REVISIONE}/{FILE}", DEST)
            modello = DEST
        else:
            # llama-server carica da solo le parti successive: basta puntargli la prima,
            # e i nomi originali devono restare, perche' e' da quelli che le trova
            cartella = os.path.join(os.path.dirname(DEST) or "/root", "modello")
            os.makedirs(cartella, exist_ok=True)
            log(f"{len(pezzi)} parti da mettere in {cartella}")
            for pezzo in pezzi:
                porta_giu(
                    f"https://huggingface.co/{REPO}/resolve/{REVISIONE}/{pezzo}",
                    os.path.join(cartella, os.path.basename(pezzo)),
                )
            modello = os.path.join(cartella, os.path.basename(pezzi[0]))

    # llama-server deve leggere il file locale, non rimettersi a scaricare da HuggingFace
    os.environ["LLAMA_ARG_MODEL"] = modello
    for chiave in ("LLAMA_ARG_HF_REPO", "LLAMA_ARG_HF_FILE", "LLAMA_ARG_MODEL_URL"):
        os.environ.pop(chiave, None)
    avvia()


if __name__ == "__main__":
    main()
