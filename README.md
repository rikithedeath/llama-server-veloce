# llama-server-veloce

`ghcr.io/ggml-org/llama.cpp:server-cuda` con una sola aggiunta: **il modello scende a connessioni
parallele** invece che a una sola.

## Perché

Il downloader interno di llama.cpp apre una connessione sola. Misurato su un host Runpod in EU-RO-1,
sullo stesso file e nello stesso momento:

| connessioni | banda |
|---|---|
| 1 | 10,9 MB/s |
| 8 | 94,5 MB/s |

Non è HuggingFace a limitare. Su un GGUF da 15,6 GiB sono **quaranta minuti invece di tre**, tutti a
GPU noleggiata e accesa: circa mezzo dollaro buttato a ogni creazione di pod.

Su Runpod non si può rimediare dall'esterno: la creazione del pod espone solo `args`, *«Arguments
passed to the container entrypoint»*, e l'entrypoint non è sostituibile. L'unico posto dove infilare
un downloader è dentro l'immagine.

## Come si usa

Identica all'immagine di partenza, con tre variabili in più. **Senza `MODELLO_REPO` o `MODELLO_URL`
si comporta esattamente come l'originale**, argomenti e `LLAMA_ARG_*` compresi.

| variabile | default | cosa fa |
|---|---|---|
| `MODELLO_REPO` | — | repo HuggingFace, es. `<utente>/<modello>-GGUF` |
| `MODELLO_FILE` | — | nome del file dentro il repo; se finisce per `-00001-of-000NN.gguf` scende **tutta la serie** |
| `MODELLO_URL` | — | URL diretto, in alternativa alle due sopra |
| `MODELLO_PATH` | `/root/modello.gguf` | dove salvarlo |
| `MODELLO_REVISIONE` | `main` | branch o tag del repo |
| `MODELLO_CONNESSIONI` | `8` | quante range-request in parallelo |
| `MODELLO_PEZZO_MB` | `64` | dimensione di un pezzo |
| `MODELLO_TENTATIVI` | `5` | ritentativi per pezzo |
| `HF_TOKEN` | — | per i repo che lo richiedono |

I GGUF oltre i ~50 GB sono pubblicati a pezzi (`-00001-of-00002.gguf`). Basta indicare il primo: gli
altri si ricavano dal nome, finiscono in `modello/` accanto a `MODELLO_PATH` con il nome originale — è
da quello che llama-server trova i successivi — e `LLAMA_ARG_MODEL` punta al primo.

Finito il download imposta `LLAMA_ARG_MODEL` sul file locale, toglie `LLAMA_ARG_HF_REPO` e
`LLAMA_ARG_HF_FILE` perché llama-server non si rimetta a scaricare, e fa `exec`: il server resta il
PID 1 del container.

Il progresso è stampato su righe intere ogni dieci secondi, così **si vede nei log di Runpod** — il
downloader originale scrive con `\r` e resta invisibile.

## Esempio su Runpod

```bash
runpodctl pod create --name llama \
  --image ghcr.io/rikithedeath/llama-server-veloce:latest \
  --gpu-id "NVIDIA GeForce RTX 4090" --cloud-type SECURE \
  --data-center-ids EU-RO-1,EU-CZ-1,EUR-IS-1,EUR-IS-2 \
  --container-disk-in-gb 40 --ports "8080/http" \
  --env '{"MODELLO_REPO":"...","MODELLO_FILE":"...","LLAMA_ARG_ALIAS":"...", ...}'
```
