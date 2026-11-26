"""Reproducible, interpretable NLP pipeline for Ctrl+Trip."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LABELS = ["negative", "neutral", "positive"]
ASPECTS = {
    "food": ["food", "meal", "restaurant", "cafe", "cuisine", "breakfast", "dinner", "dish"],
    "accommodation": ["hotel", "room", "hostel", "stay", "bed", "lodging", "resort", "accommodation"],
    "transport": ["transport", "bus", "taxi", "traffic", "train", "road", "driver", "metro", "commute"],
    "safety": ["safe", "safety", "unsafe", "security", "scam", "theft", "crowd", "crowded"],
    "scenery": ["scenery", "view", "mountain", "beach", "landscape", "sunset", "lake", "waterfall", "beautiful"],
    "activities": ["activity", "activities", "hike", "hiking", "museum", "fort", "tour", "sightseeing", "walk", "temple"],
}
POSITIVE = {"good", "great", "excellent", "clean", "friendly", "delicious", "beautiful", "safe", "peaceful", "helpful", "comfortable", "enjoyed", "amazing", "love", "lovely", "smooth", "recommend", "stunning", "fresh", "welcoming"}
NEGATIVE = {"bad", "awful", "dirty", "rude", "slow", "unsafe", "expensive", "crowded", "noisy", "poor", "terrible", "disappointing", "uncomfortable", "overpriced", "broken", "smelly", "unhelpful", "avoid", "dangerous", "worst"}
NEGATORS = {"not", "never", "no", "hardly", "isnt", "wasnt", "dont", "didnt"}
STOPWORDS = set("a an the and or but if while with to of in on at for from by as is am are was were be been being have has had do does did this that these those it its we they i you he she my our their your very really quite just also there here into about after before over under".split())
ROOT = Path(__file__).resolve().parents[1]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", clean_text(text))
    return [w for w in words if len(w) > 1 and (not remove_stopwords or w not in STOPWORDS)]


def lemmatize(token: str) -> str:
    """Small deterministic suffix lemmatizer; no downloaded language model required."""
    irregular = {"went": "go", "was": "be", "were": "be", "children": "child", "mice": "mouse", "better": "good", "best": "good"}
    if token in irregular:
        return irregular[token]
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        return stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def preprocess(text: str) -> str:
    return " ".join(lemmatize(t) for t in tokenize(text, remove_stopwords=True))


def sentiment_rule(text: str) -> str:
    tokens = tokenize(text)
    score = 0
    for i, token in enumerate(tokens):
        value = 1 if token in POSITIVE else -1 if token in NEGATIVE else 0
        if value and any(t in NEGATORS for t in tokens[max(0, i - 3):i]):
            value *= -1
        score += value
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def load_reviews() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "travel_reviews_sample.csv").fillna("")


def build_model(reviews: pd.DataFrame | None = None) -> Pipeline:
    frame = reviews if reviews is not None else load_reviews()
    return Pipeline([("tfidf", TfidfVectorizer(preprocessor=preprocess, tokenizer=str.split,
                                                 token_pattern=None, lowercase=False,
                                                 ngram_range=(1, 2), min_df=1,
                                                 sublinear_tf=True)),
                     ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                        random_state=42))]).fit(frame.review.astype(str), frame.sentiment.astype(str))


def evaluate_model(frame: pd.DataFrame | None = None) -> dict:
    data = (frame if frame is not None else load_reviews()).copy()
    train, test = train_test_split(data, test_size=0.25, random_state=42, stratify=data.sentiment)
    model = build_model(train)
    pred = model.predict(test.review.astype(str))
    precision, recall, f1, _ = precision_recall_fscore_support(test.sentiment, pred, labels=LABELS, zero_division=0, average="macro")
    report = classification_report(test.sentiment, pred, labels=LABELS, output_dict=True, zero_division=0)
    return {"model": model, "accuracy": accuracy_score(test.sentiment, pred), "precision": precision,
            "recall": recall, "f1": f1, "confusion_matrix": confusion_matrix(test.sentiment, pred, labels=LABELS),
            "labels": LABELS, "report": report, "train_size": len(train), "test_size": len(test)}


def extract_aspects(text: str, model: Pipeline) -> dict[str, dict]:
    sentences = re.split(r"(?<=[.!?])\s+", str(text))
    result = {}
    for aspect, terms in ASPECTS.items():
        matches = [s.strip() for s in sentences if any(re.search(r"\b" + re.escape(term) + r"\w*\b", s.lower()) for term in terms)]
        if matches:
            labels = list(model.predict(matches))
            counts = Counter(labels)
            result[aspect] = {"sentiment": counts.most_common(1)[0][0], "evidence": matches[:3], "mentions": len(matches)}
    return result


def extract_keywords(texts: list[str], limit: int = 12) -> list[tuple[str, float]]:
    vectorizer = TfidfVectorizer(preprocessor=preprocess, tokenizer=str.split, token_pattern=None,
                                 lowercase=False, ngram_range=(1, 2), min_df=1, max_features=3000)
    matrix = vectorizer.fit_transform([str(t) for t in texts if str(t).strip()])
    if matrix.shape[0] == 0:
        return []
    scores = matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    order = scores.argsort()[::-1]
    return [(str(terms[i]), float(scores[i])) for i in order[:limit]]


def extract_entities(text: str, known_places: set[str] | None = None) -> list[dict[str, str]]:
    known_places = known_places or set()
    found = {}
    # High precision for destinations and attractions supplied by the bundled catalogue.
    for place in sorted(known_places, key=len, reverse=True):
        if re.search(r"\b" + re.escape(place) + r"\b", text, re.I):
            found[place.lower()] = {"text": place, "label": "PLACE"}
    # A transparent rule-based fallback for proper-name entities, not a pretrained NER model.
    for match in re.finditer(r"\b(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+){0,2}\b", text):
        phrase = match.group(0)
        if phrase.lower() not in {"I".lower()}:
            found.setdefault(phrase.lower(), {"text": phrase, "label": "PROPER_NAME"})
    return list(found.values())


def summarize_reviews(reviews: list[str], model: Pipeline, max_sentences: int = 5) -> dict:
    candidates = []
    for review in reviews:
        for sentence in re.split(r"(?<=[.!?])\s+", str(review)):
            if len(tokenize(sentence)) >= 4:
                candidates.append(sentence.strip())
    if not candidates:
        return {"summary": "No review sentences are available.", "evidence": []}
    aspects = sorted({a for s in candidates for a, terms in ASPECTS.items()
                      if any(re.search(r"\b" + re.escape(t) + r"\w*\b", s.lower()) for t in terms)})
    # Rank concrete source sentences by aspect coverage and lexical specificity.
    ranked = sorted(enumerate(candidates), key=lambda item: (
        sum(any(re.search(r"\b" + re.escape(t) + r"\w*\b", item[1].lower()) for t in ASPECTS[a]) for a in aspects),
        len(set(tokenize(item[1], True))), -item[0]), reverse=True)
    chosen = [s for _, s in ranked[:max_sentences]]
    labels = model.predict(chosen)
    return {"summary": " ".join(chosen), "evidence": [{"sentence": s, "sentiment": str(label)} for s, label in zip(chosen, labels)]}


def recommend_destinations(frame: pd.DataFrame, preferences: list[str], model: Pipeline) -> pd.DataFrame:
    rows = []
    for destination, group in frame.groupby("destination"):
        hits = 0
        for review in group.review.astype(str):
            lower = review.lower()
            hits += sum(pref.lower() in lower for pref in preferences)
        sentiment = list(model.predict(group.review.astype(str)))
        positive_rate = sum(x == "positive" for x in sentiment) / max(1, len(sentiment))
        preference_rate = hits / max(1, len(group) * max(1, len(preferences)))
        score = 100 * (0.65 * positive_rate + 0.35 * preference_rate)
        rows.append({"destination": destination, "score": round(score, 1), "positive_review_share": round(positive_rate, 3),
                     "preference_signal": round(preference_rate, 3), "review_count": len(group),
                     "reason": f"{sum(x == 'positive' for x in sentiment)}/{len(group)} sample reviews predicted positive; preference terms found {hits} times."})
    return pd.DataFrame(rows).sort_values(["score", "destination"], ascending=[False, True]).reset_index(drop=True)
