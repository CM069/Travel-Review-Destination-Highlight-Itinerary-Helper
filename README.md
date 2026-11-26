# Ctrl+Trip — Travel Review Intelligence

Ctrl+Trip is a college NLP project that combines an existing Streamlit trip-planning MVP with an interpretable travel-review analysis workflow. It uses Python, pandas, scikit-learn, TF-IDF, Logistic Regression and optional SQLite storage.

## What is implemented

- Overview dashboard with the project palette, bundled destination cards, and clear synthetic-data notices; the sidebar uses a dark navigation surface against the light page.
- Destination profiles with real model predictions over bundled sample text, aspect evidence and source review snippets.
- Trip planner with destination, duration, interests, pace, travel mode and traveller inputs; draft itinerary, illustrative budget breakdown and CSV export.
- Destination Explorer cards with bundled, offline AI-generated illustrative images.
- Review workspace using the bundled labelled sample or a user CSV containing `review` and optionally `destination`.
- Text cleaning, regex tokenisation, explicit stopword filtering and a deterministic suffix/irregular-word lemmatiser that requires no external model download.
- TF-IDF unigrams/bigrams and class-weighted Logistic Regression.
- Stratified 75/25 train/test split (fixed seed 42), accuracy, macro precision, recall, F1, per-class report internally, and a confusion matrix.
- Aspect sentiment for food, accommodation, transport, safety, scenery and activities; each result shows matching source sentences.
- TF-IDF keyword extraction, catalogue-aware location/attraction recognition plus transparent proper-name rules, and extractive summaries that show their source sentences and predicted labels.
- Explainable destination ranking: 65% predicted positive share and 35% preference phrase matches over bundled sample reviews.
- Optional local SQLite persistence after an explicit checkbox and button.

## Dataset and provenance

`data/travel_reviews_sample.csv` is a **synthetic, reproducible demonstration dataset authored for this project**. It contains 75 short labelled examples across Manali, Jaipur and Goa. It is not scraped, licensed third-party content, representative customer feedback, or an empirical benchmark. No external dataset license is claimed. Use a properly licensed labelled dataset for research conclusions or deployment. The small sample exists to make the project runnable offline and to demonstrate a complete training/evaluation flow.

`data/destinations.csv` and `data/attractions.csv` are illustrative hand-authored catalogue records. Their descriptions, ratings and fees are demo values, not verified current information. Destination card images are AI-generated illustrations described in `assets/IMAGE_CREDITS.md`, not documentary travel photos.

## Windows PowerShell setup

Use Python 3.12 (Python 3.11 is also supported). From the extracted project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again. Open the local address printed by Streamlit (usually `http://localhost:8501`). First dependency installation requires internet access; runtime and sample data do not require an API key.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository. A public repository is the simplest option, but makes the source code visible to anyone. Use a private repository if needed and authorize Community Cloud to access it.
2. From this project folder in PowerShell, publish the project files (replace the URL with your repository URL):

   ```powershell
   git init
   git add .
   git status
   git commit -m "Prepare Ctrl+Trip for deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/ctrltrip.git
   git push -u origin main
   ```

   Check `git status` before committing: `.venv`, `.env`, secrets, and generated SQLite files should not be staged. The destination images, CSV files, `app.py`, `.streamlit/config.toml`, and `requirements.txt` should be included.
3. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub, choose **Create app**, then select your repository, the `main` branch, and `app.py` as the entrypoint.
4. In **Advanced settings**, select Python 3.12, then deploy. This project does not need API secrets.
5. Open the URL Community Cloud gives you. Future commits to the selected branch update the app.

Community Cloud does not guarantee that files written to its local filesystem will persist. The optional SQLite review-save feature is suitable for a local demo, not durable cloud storage; saved rows may disappear after a rebuild or restart. See the [Streamlit Community Cloud deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Review CSV format

Minimum:

```csv
review
The scenery was beautiful and the food was delicious.
```

Recommended columns: `review,destination,sentiment`. Sentiment labels in the sample are `negative`, `neutral`, `positive`; upload analysis does not require labels. The classifier evaluation and recommendation baseline are always measured using the bundled training sample, while predictions/aspects/summary/keywords/entities apply to the selected review text.

## Tests

Run `python -m pytest -q`. Tests cover preprocessing, class coverage, model evaluation output, aspect evidence, keyword and entity extraction, recommendation order, and local SQLite persistence using a temporary test database.

## Data and model limitations

The included review corpus is small synthetic demo data. Metrics can be unstable and must not be presented as real-world performance. Aspect detection is dictionary/phrase based; sentiment is sentence-level model prediction. Entity extraction uses the bundled attraction/destination catalogue plus simple capitalization rules, not a pretrained general-purpose NER model. Summaries quote source sentences and do not invent travel facts. Recommendations explain their formula and use only the small sample. The trip planner uses fixed illustrative cost assumptions and a short hand-authored catalogue. Ctrl+Trip does not implement live maps, live prices, booking, real-time availability, or route verification.

## Project structure

```text
app.py
src/            NLP, catalogue loading, itinerary and SQLite helpers
data/           synthetic review sample and illustrative catalogues
tests/          automated unit tests
docs/           feature notes and original architecture reference notes
requirements.txt
```
