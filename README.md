# local-pdf-rag-chatbot

Chat with a PDF, entirely on your own machine. Upload a document, ask questions, and get
answers grounded in what the document actually says — with a built-in refusal when the answer
isn't in there.

**Nothing leaves your computer.** The language model runs locally through
[Ollama](https://ollama.com), the vector store is a local [ChromaDB](https://www.trychroma.com)
directory, and the app makes no outbound calls. No API keys, no cloud costs, no telemetry.

### Project history

The vector-base workflow was originally prototyped in [Langflow](https://www.langflow.org/).
What ships here is plain Python (Streamlit + ChromaDB + the Ollama client) — you don't need
Langflow to run it. If you come across this project under the older name
`rag-chatbot-langflow`, that's why.

---

## How it works

```
your browser  ──►  Streamlit app (Docker)  ──►  Ollama (on your host)
                          │
                          └──►  ChromaDB (Docker volume)
```

The app is containerized; **Ollama stays on your host machine**. That's deliberate — it keeps
using the models you've already pulled and whatever GPU acceleration you already have, instead
of downloading gigabytes again into a container.

On Docker Desktop (Windows and macOS) that split needs no extra setup — `host.docker.internal`
is forwarded to your host automatically, including to a default `127.0.0.1`-only Ollama. On
Linux it needs one extra step; see [Linux: let Ollama accept connections from Docker](#linux-let-ollama-accept-connections-from-docker).

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose)
- [Ollama](https://ollama.com/download), installed and running

## Setup

### 1. Pull the models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Roughly 2.3 GB in total. (If you skip this, the app will offer to download them for you on
first run.)

### 2. Start the app

```bash
docker compose up --build
```

The first build takes a few minutes — ChromaDB pulls in `onnxruntime` and friends. Later starts
are fast.

Then open **http://localhost:8501**.

On Docker Desktop that's it — you should land straight on the upload screen.

### Linux: let Ollama accept connections from Docker

Skip this on Windows and macOS. On Linux, the container reaches your machine over Docker's
bridge network rather than loopback, so a default Ollama — which listens on `127.0.0.1` only —
won't answer it. Tell Ollama to listen on all interfaces:

```bash
sudo systemctl edit ollama.service
```

Add under `[Service]`:

```
Environment="OLLAMA_HOST=0.0.0.0"
```

Then reload and restart:

```bash
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

> **Worth knowing:** this makes Ollama reachable from your local network, not just from your own
> machine. On a home network behind a router that's normally fine. On a shared or public network
> (café, hotel, office wifi), consider setting it back to `127.0.0.1` when you're done.

## Privacy

- **Nothing is sent anywhere.** The models run locally via Ollama; there are no API keys and
  no outbound requests. Streamlit's usage telemetry is switched off in the Dockerfile.
- **The app is bound to `127.0.0.1`**, so it's reachable only from your own computer. It has
  **no login**, so if you change the port mapping in `docker-compose.yml` to a bare
  `"8501:8501"`, everyone on your network can read the documents you upload. Only do that on a
  network you trust.
- **Uploaded documents persist** as embedded text in the `chroma_data` Docker volume until you
  remove it with `docker compose down -v`.

## Configuration

All optional. Copy `.env.example` to `.env` and edit, or set them in your shell — Compose picks
them up automatically.

| Variable | Default | What it does |
| --- | --- | --- |
| `CHAT_MODEL` | `llama3.2` | Ollama model used to answer questions |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model used to embed chunks |
| `RELEVANCE_MAX_DISTANCE` | `0.55` | Relevance cut-off (cosine distance; lower = stricter). Also adjustable live in the sidebar |
| `CHROMA_PATH` | `/data/chroma` in Docker, `./chroma_data` otherwise | Where the vector store lives |
| `OLLAMA_URL` | `http://localhost:11434` | Where the app looks for Ollama. Compose sets this to `http://host.docker.internal:11434` |

> `OLLAMA_URL` is intentionally not named `OLLAMA_HOST`. On your host machine, `OLLAMA_HOST` is
> the address Ollama *binds to* (step 2 above); here it's the URL the app *connects to*. They're
> different things, and reusing the name causes real confusion. `OLLAMA_HOST` is still read as a
> fallback if `OLLAMA_URL` isn't set.

## Troubleshooting

**"I can't reach Ollama"** — first confirm Ollama is actually running on your host:

```bash
ollama list
```

If that works and the app still can't connect:

- **On Linux**, apply [the bind change above](#linux-let-ollama-accept-connections-from-docker).
- **On Windows/macOS**, this is unusual — Docker Desktop normally forwards to your host's
  loopback. The same `OLLAMA_HOST=0.0.0.0` fix applies (`setx OLLAMA_HOST 0.0.0.0` on Windows,
  `launchctl setenv OLLAMA_HOST "0.0.0.0"` on macOS). Restart Ollama **completely** afterwards —
  `setx` only affects newly started processes, so quit it from the system tray and reopen it.

To check what Ollama is bound to:

```bash
# Windows
netstat -ano | findstr 11434
# macOS / Linux
lsof -iTCP:11434 -sTCP:LISTEN
```

**The first build is slow / looks stuck** — that's `pip install chromadb` fetching large wheels.
Give it a few minutes.

**Answers say "I couldn't find that in the uploaded document" too often** — lower the relevance
threshold in the sidebar, or raise `RELEVANCE_MAX_DISTANCE`.

**Starting over with a clean document store:**

```bash
docker compose down -v
```

The `-v` removes the `chroma_data` volume along with every embedded document.

## Running without Docker

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

No Ollama networking changes are needed in this case — the app talks to
`http://localhost:11434` directly.

## License

MIT — see [LICENSE](LICENSE).
