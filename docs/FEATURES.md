# Feature and scope notes

## Implemented

- Streamlit overview dashboard with sample destination imagery and project navigation.
- Destination profiles with sentiment distribution, aspect evidence and sample review snippets.
- Trip planner inputs: destination, solo/group, travellers, budget type, days, interests, pace and preferred transport.
- Day-wise attraction suggestions, fixed illustrative cost estimates, budget comparison and CSV export.
- Review CSV import or bundled synthetic labelled sample.
- Preprocessing, TF-IDF + Logistic Regression, held-out metrics, six phrase-based aspects, TF-IDF keyword extraction, catalogue-aware place extraction, evidence-backed extractive summary and review-signal recommendations.
- Optional SQLite storage for uploaded review records, invoked only after user selection.

## Data notes

Bundled review examples are synthetic, project-authored demonstration data; destination and attraction records are hand-authored illustrations. They are not current travel advice or third-party licensed review data. See the README for provenance, evaluation limits and run instructions.

## Deferred scope

No live maps, route service, external media, bookings, inventory, current prices, opening-hours feed, online database, API, or model-hosting service is connected. Add only after identifying a provider, valid licence, and integration requirements.
