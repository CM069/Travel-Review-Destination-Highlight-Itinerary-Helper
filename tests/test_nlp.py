import pandas as pd

from src.nlp import (build_model, evaluate_model, extract_aspects, extract_entities,
                     extract_keywords, lemmatize, load_reviews, preprocess,
                     recommend_destinations, summarize_reviews, tokenize)
from src.storage import review_count, save_reviews


def test_preprocessing_pipeline():
    assert tokenize("Cafés were lovely! https://example.com") == ["cafes", "were", "lovely"]
    assert "the" not in tokenize("the mountain views", remove_stopwords=True)
    assert lemmatize("waterfalls") == "waterfall"
    assert "mountain" in preprocess("The mountains were beautiful")


def test_sample_and_evaluation_have_all_labels():
    data = load_reviews()
    assert len(data) >= 60
    assert set(data.sentiment) == {"negative", "neutral", "positive"}
    result = evaluate_model(data)
    assert result["train_size"] + result["test_size"] == len(data)
    assert 0 <= result["accuracy"] <= 1
    assert result["confusion_matrix"].shape == (3, 3)
    assert set(result["labels"]) == {"negative", "neutral", "positive"}


def test_aspects_include_evidence_and_model_sentiment():
    model = build_model()
    aspects = extract_aspects("The food was delicious. The hotel room was dirty.", model)
    assert set(aspects) >= {"food", "accommodation"}
    assert all(item["evidence"] for item in aspects.values())
    assert all(item["sentiment"] in {"negative", "neutral", "positive"} for item in aspects.values())


def test_keywords_entities_summary_and_recommendations():
    data = load_reviews()
    model = build_model(data)
    assert extract_keywords(data.review.tolist())
    entities = extract_entities("We visited Amber Fort in Jaipur.", {"Jaipur", "Amber Fort"})
    assert any(e["label"] == "PLACE" for e in entities)
    summary = summarize_reviews(data.review.head(8).tolist(), model)
    assert summary["evidence"]
    recs = recommend_destinations(data, ["mountain"], model)
    assert len(recs) == 3
    assert recs.score.between(0, 100).all()


def test_sqlite_persistence_is_local_and_counted(tmp_path):
    path = tmp_path / "test.sqlite3"
    sample = pd.DataFrame([{"destination": "Test", "review": "A clean hotel."}])
    assert save_reviews(sample, path) == 1
    assert review_count(path) == 1
