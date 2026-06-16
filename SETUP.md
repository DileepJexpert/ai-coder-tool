# watchdog setup — fully on your laptop

Everything in this guide runs locally. No accounts, no API keys, no cloud calls.
After the one-time installs + model downloads, you can use it offline forever.

There are three pieces:

| Piece           | What it does                       | Lives where     |
|-----------------|------------------------------------|------------------|
| **Ollama**      | runs the chat / coding LLM         | localhost:11434  |
| **Forge** (SD)  | generates images                   | localhost:7860   |
| **watchdog**    | the agent + browser UI you use     | localhost:8765   |

All three are local apps with weights you download once. None of them phone home.

---

## 1. Text agent — Ollama

1. Install Ollama from <https://ollama.com/download> (Win / Mac / Linux installers).
2. Pull the models watchdog uses by default:

   ```bash
   ollama pull qwen3:8b          # chat / tool-calling model (~5 GB)
   ollama pull nomic-embed-text  # embeddings for code search (~270 MB)
   ollama pull moondream         # tiny vision model, optional (~1.7 GB)
   ```

3. On Win/Mac the installer runs Ollama as a background service. On Linux:

   ```bash
   ollama serve &
   ```

Pick a smaller / larger model later by passing `-m <name>` to watchdog.

---

## 2. Image generator — Forge (Stable Diffusion)

Forge is a lighter, faster fork of AUTOMATIC1111's SD WebUI. AUTOMATIC1111 itself
works identically with watchdog — Forge is just easier on modest hardware.

1. Clone and install:

   ```bash
   git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git
   cd stable-diffusion-webui-forge
   ```

   - **Linux / macOS:** `./webui.sh --api`
   - **Windows:** `webui.bat --api`

   The first launch downloads Python deps (a few minutes). The `--api` flag is what
   lets watchdog talk to it.

2. Download a Stable Diffusion 1.5 checkpoint (`.safetensors` file, 2 – 4 GB).
   Any general-purpose one works; some popular ones:

   - **DreamShaper 8** — versatile realistic/illustration  
     <https://huggingface.co/Lykon/DreamShaper>
   - **Realistic Vision** — photorealism  
     <https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE>
   - **ToonYou** — cartoon / kid illustrations  
     <https://civitai.com/models/30240>

   Drop the file into:

   ```
   stable-diffusion-webui-forge/models/Stable-diffusion/
   ```

3. In the Forge web UI (opens automatically at <http://localhost:7860>), pick the
   model from the dropdown at the top-left. First load takes 30–60 s.

That's it — watchdog auto-detects Forge at `http://localhost:7860`. Set
`A1111_HOST=http://<host>:<port>` before launching watchdog if you moved it.

---

## 3. watchdog itself

1. Clone the repo and install Python deps:

   ```bash
   git clone https://github.com/DileepJexpert/ai-coder-tool.git
   cd ai-coder-tool
   pip install -r requirements.txt
   ```

2. (Optional but recommended) Build the semantic code-search index for the project
   you're working on. Run this inside that project's directory:

   ```bash
   python /path/to/ai-coder-tool/codebase_rag.py index .
   ```

3. Launch the web UI:

   ```bash
   python /path/to/ai-coder-tool/watchdog.py --web
   ```

   Open <http://127.0.0.1:8765>.

---

## Hardware notes

| Component           | RAM / VRAM        | Speed on a modest laptop          |
|---------------------|-------------------|------------------------------------|
| Ollama `qwen3:8b`   | ~6 GB             | 5 – 30 tok/s on CPU; faster on GPU |
| SD1.5 @ 512×512     | ~4 GB VRAM        | 1 – 5 s/image on RTX 3060+         |
|                     | (CPU fallback OK) | 30 s – 3 min on CPU / Apple Silicon |

If you have only 8 GB VRAM and want both running at once: stop Forge between image
batches, or run Ollama on CPU (`OLLAMA_NO_GPU=1`) and let SD have the GPU. For a
day-to-day workflow (occasional images), Forge can sit idle in the background — it
only uses GPU during the few seconds it's generating.

---

## Verifying everything stays local

Open a terminal while watchdog is running:

```bash
# Linux / macOS — lists every TCP connection your tools have open.
# You should see ONLY 127.0.0.1 / ::1 lines.
lsof -iTCP -sTCP:ESTABLISHED -P | grep -E 'python|ollama|webui|main'
```

No traffic leaves the machine.

---

## Troubleshooting

- **Tool result says "stub image saved to images/…svg"** — Forge isn't running, or it
  was started without `--api`. Re-launch it with `./webui.sh --api`.
- **`generate_image` errors with 500 from Forge** — usually means no checkpoint is
  loaded. Open <http://localhost:7860> in a browser and pick a model from the dropdown.
- **`search_code` errors with "no semantic index found"** — run
  `python codebase_rag.py index .` in the project you want to search.
- **Ollama errors with `model not found`** — `ollama pull <model-name>`.
- **`--web` errors with "fastapi required"** — `pip install -r requirements.txt`.
