# Synthetic Data Generator (Week 3 Day 5)

Generate synthetic tabular datasets with several LLMs taking turns in batches, then compare them.

- **Gemini** `gemini-2.5-flash` (free tier, via Google's OpenAI-compatible endpoint)
- **Ollama cloud** `gemma4:31b-cloud` (runs remotely, reached through your local Ollama)
- **Ollama local** `llama3.2` (3B) and `qwen2.5:3b` (3B)

Every row gets a `source_model` column, and the per-model stats table shows how many rows each model contributed and how often its batches failed.

## Setup

1. **Install Ollama** from https://ollama.com and make sure it is running (the desktop app starts it; or run `ollama serve`).

2. **Pull the models:**
   ```bash
   ollama pull llama3.2
   ollama pull qwen2.5:3b
   ```

3. **Sign in for the cloud model** (needs a free ollama.com account), then pull it:
   ```bash
   ollama signin
   ollama pull gemma4:31b-cloud
   ```
   Cloud models don't download weights; the pull just registers the model locally.

4. **Get a Gemini API key** from https://aistudio.google.com/apikey. Copy `.env.example` to `.env` and fill it in:
   ```
   GOOGLE_API_KEY=your_real_key
   ```

5. **Install Python packages** (a virtual environment is recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS / Linux
   pip install -r requirements.txt
   ```

## Run

```bash
python app.py
```

Open the URL it prints (usually http://127.0.0.1:7860). Pick a preset or write your own purpose and schema, then click **Generate**.
CSV files are saved in the `outputs/` folder and can also be downloaded from the UI.

If you change `.env`, restart the app. The key is read only once, at startup.

## Notes

- Missing Gemini key or Ollama not running? The app shows a warning and keeps going with the provider that works.
- A model that fails 3 batches in a row is removed from the rotation for that run.
- To change models, edit the `MODELS` dict at the top of `generator.py`.
