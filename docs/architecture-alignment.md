# Architecture reference review

The project was upgraded from the user's Library file `Ctrl_Trip_Updated_Project.zip`, retaining its Streamlit interface, CSV destination/attraction catalogues and practical trip planner.

The supplied `Ctrl_Trip_Advanced_Architecture.html` describes a staged system: data ingestion, text preprocessing, NLP analysis, theme aggregation, evidence-backed summaries and transparent recommendations. The runnable MVP now implements the local/offline subset of that direction: CSV review input, preprocessing, supervised sentiment, phrase-based aspect signals, keywords, catalogue/name rules, extractive evidence and a review-signal ranking formula.

The architecture document also distinguishes future services from the MVP and cautions against pretending to know live conditions. For that reason, this build does not claim external maps, current prices, booking, media lookup, real-time availability, or generalized pretrained named-entity recognition. SQLite is used only for opt-in local review persistence. Source material itself remains in the user's Library; this note records how the product direction informed implementation.
