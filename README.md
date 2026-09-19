# local-pdf-rag-chatbot

Chat with a PDF, entirely on your own machine. Upload a document, ask questions, and get
answers grounded in what the document actually says — with a built-in refusal when the answer
isn't in there.

**Nothing leaves your computer.** The language model runs locally through
[Ollama](https://ollama.com), the vector store is a local [ChromaDB](https://www.trychroma.com)
directory, and the app makes no outbound calls. No API keys, no cloud costs, no telemetry.

**Who it's for:** anyone who can't send sensitive documents to third-party AI services, and
anyone evaluating what a fully private document Q&A setup looks like in practice.

> **Proof of concept.** This project demonstrates that private, no-per-query-fee document Q&A
> is feasible. Because the language model runs locally through Ollama, answers are **slow on
> modest hardware** (for example, a laptop with 8 GB of RAM), and speed depends heavily on your
> machine. It is not a production-speed system. See [Path to production](#path-to-production)
> for ways to make it faster.

### Project history

The RAG workflow (document ingestion, embeddings, retrieval and answer generation) was
originally designed and prototyped in [Langflow](https://www.langflow.org/). The app that ships
here was then built with [Claude Code](https://www.anthropic.com/claude-code), directed and
tested by the author. What ships is plain Python (Streamlit + ChromaDB + the Ollama client) —
you don't need Langflow to run it.

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

Install both before you start. Order doesn't matter, but both must be in place before step 3.

- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (or Docker Engine +
  Compose) — and actually running; look for the whale icon in your tray or menu bar
- **[Ollama](https://ollama.com/download)** — installs as a background service and starts on
  its own

## Setup

### 1. Get the code

```bash
git clone https://github.com/san-j-kv/local-pdf-rag-chatbot.git
cd local-pdf-rag-chatbot
```

### 2. Pull the models — optional

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Roughly 2.3 GB in total. **You can skip this**: if the models are missing, the app shows a
"Download required models" button on first run and pulls them for you with a progress bar.
Doing it here is just faster.

### 3. Start the app

```bash
docker compose up --build
```

The first build takes a few minutes — ChromaDB pulls in `onnxruntime` and friends. Later starts
are fast.

### 4. Open it

**http://localhost:8501**

Nothing answers that address until step 3 is running — installing Ollama alone doesn't start
anything on port 8501.

### What you'll see on first run

| Situation | What appears |
| --- | --- |
| Ollama isn't installed or isn't running | *"I can't reach Ollama"*, with a fix for your OS |
| Ollama is running, models not pulled | A warning, the `ollama pull` commands, and a download button |
| Ollama is running, models present | **The upload screen — ready to use** |

If port 8501 is already taken by something else, the container will start but the page won't
load. Change the host side of the mapping in `docker-compose.yml` (for example
`"127.0.0.1:8601:8501"`) and use that port instead.

### Try it with the sample document

No PDF handy? Upload [`samples/northwind-solar-handbook.pdf`](samples/northwind-solar-handbook.pdf)
— two pages of a fictional solar-panel handbook — and ask:

| Ask | What it shows |
| --- | --- |
| *What is the warranty period for the panels?* | A grounded answer, with the page cited: **(p. 2)** |
| *What does fault code E-14 mean?* | It answers, then names the part of your question the document doesn't cover |
| *Who won the 2022 football World Cup?* | *"I couldn't find that in the uploaded document."* |

That last one is the point of the whole app. The question never reaches the language model —
nothing in the document is close enough to the question, so it declines instead of guessing.

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

Docker isn't required. With Python 3.12+ installed, Ollama becomes the only prerequisite:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py --server.address=127.0.0.1
```

Then open **http://localhost:8501**.

> The `--server.address=127.0.0.1` matters. Streamlit's default is to listen on *all*
> interfaces, and this app has no login — without that flag, anyone on your network could open
> it and read the documents you upload. The Docker setup handles this for you.

**What you gain**

- **No Docker at all** — one less large install, and nothing to keep running in the background
- **Works the same on every OS.** The Linux bind step above becomes unnecessary, because the
  app talks to `http://localhost:11434` directly rather than across a container boundary
- **Faster to start**, and no multi-minute first build
- **Easier to hack on.** Edit `app.py` and Streamlit hot-reloads; no rebuild needed
- Embedded documents land in a plain `./chroma_data` folder you can inspect or delete directly

**What you give up**

- **Dependency isolation.** `chromadb`, `streamlit` and `onnxruntime` are pinned for a reason;
  installed into a shared environment they can collide with your other projects. Use a venv —
  the commands above do
- **A known-good Python.** The image pins 3.12; your system Python may be older or newer, and
  `chromadb` in particular is sensitive to this
- **Reproducibility.** "Works on my machine" problems become yours to debug rather than the
  image's to prevent
- **The safe default binding.** Compose binds the app to `127.0.0.1` for you; Streamlit's own
  default listens on every interface. Remembering the flag above is now on you
- Running as a non-root user with a read-only-ish filesystem, which the container gives you free

**Rule of thumb:** use Docker to *run* it, skip Docker to *modify* it.

## Path to production

This is a proof of concept. Below are two realistic routes to a faster, shared version. They
are recommendations only and are **not implemented in this repo**.

**Shared in-house machine — small teams with strict privacy needs.** Run the app and Ollama on
one machine with a capable GPU (or a Mac with plenty of unified memory) and let staff open it
from a browser over the office network. Documents stay in-house, and responses are usually much
faster than on a CPU-only laptop. Before doing this, **add user authentication**: as noted in
[Privacy](#privacy), the app has no login and is bound to `127.0.0.1` by default.

**Private model in your own cloud account — faster, with less maintenance.** Run the language
model inside the organization's own cloud account, under contractual data-protection terms (for
example, no training on your data). You pay per use, speed is high, and there is no hardware to
maintain. The trade-off is that documents leave the local machine, though they stay within the
organization's contractual and compliance boundary. This route needs a code change (the app
currently talks only to Ollama) as well as authentication.

Which route fits depends on how many people use it, how strict the privacy requirement is, and
how often it's used. At low or occasional usage, per-use pricing is usually cheaper than
running dedicated hardware.

## License

MIT — see [LICENSE](LICENSE).
