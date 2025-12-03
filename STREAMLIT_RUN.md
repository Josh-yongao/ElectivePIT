# Running the Streamlit App

1. (Optional) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. The app will open in your browser at `http://localhost:8501` by default.

Notes:
- The app loads models saved in `saved_models/` - ensure these files exist.
- Use the Demo mode to run a quick example using the included dataset.
- For production, containerize or host behind a web server as needed.
