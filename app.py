import streamlit as st
import pandas as pd
from pathlib import Path
from src.planner import build_itinerary, estimate_budget
from src.data import load_destinations, load_attractions

st.set_page_config(page_title="Ctrl+Trip | Travel Intelligence", page_icon="✈️", layout="wide")
st.markdown("""
<style>
.stApp{background:#f5f7fb}.block-container{padding-top:2rem;max-width:1250px}
.hero{padding:28px 32px;border-radius:22px;background:linear-gradient(120deg,#101c35,#294a78);color:white;margin-bottom:22px}
.hero h1{font-size:42px;margin:0}.hero p{color:#d4e2f4;margin:8px 0 0}
.metric{background:white;border:1px solid #e5eaf2;border-radius:15px;padding:15px}
.place{background:white;border:1px solid #e5eaf2;border-radius:14px;padding:15px;margin:8px 0}
.small{color:#6c7890;font-size:13px}
</style>
<div class="hero"><h1>⌘ Ctrl+Trip</h1><p>Press Ctrl. Plan your Trip. · Discover destinations, understand traveller reviews and build a practical itinerary.</p></div>
""", unsafe_allow_html=True)

destinations=load_destinations()
attractions=load_attractions()
pages=st.sidebar.radio("WORKSPACE",["Overview","Trip Planner","Destination Explorer","Review Intelligence","About & Data"])
st.sidebar.caption("Ctrl+Trip · NLP-powered travel intelligence")

if pages=="Overview":
    st.title("Plan a trip with better review insights")
    st.write("Explore sample destinations, draft an itinerary, and inspect what travel reviews say about six aspects of a trip.")
    try:
        from src.nlp import load_reviews
        review_total=len(load_reviews())
    except (FileNotFoundError, pd.errors.ParserError):
        review_total=0
    first,second,third=st.columns(3)
    first.metric("Destinations",len(destinations))
    second.metric("Bundled demo reviews",review_total)
    third.metric("Review aspects",6)
    st.subheader("Explore destinations")
    image_files={"Manali":"manali.png","Jaipur":"jaipur.png","Goa":"goa.png"}
    cards=st.columns(3)
    for index,(_,row) in enumerate(destinations.iterrows()):
        with cards[index%3]:
            with st.container(border=True):
                image_path=Path(__file__).resolve().parent/"assets"/"destinations"/image_files.get(row["destination"],"")
                if image_path.is_file():
                    st.image(str(image_path),width="stretch",alt=f"Illustrative destination image for {row['destination']}")
                st.subheader(row["destination"])
                st.caption(f"{row['region']} · {row['best_for']}")
                st.write(row["description"])
    left,right=st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Build an itinerary")
            st.write("Choose a destination, trip length, interests, pace and preferred transport. Get a day-by-day draft and clearly labelled cost estimate.")
    with right:
        with st.container(border=True):
            st.subheader("Understand review signals")
            st.write("Run TF-IDF + Logistic Regression sentiment analysis, inspect aspect evidence, extract keywords and places, and compare transparent sample-based recommendations.")
    st.info("The bundled reviews are synthetic demonstration data. Images are AI-generated illustrations; destination descriptions and cost estimates are samples, not live travel information.",title="About this demo")

elif pages=="Trip Planner":
    st.title("Build your trip")
    st.write("Tell us where you want to go, how long you have and what you can spend. Ctrl+Trip creates a day-wise draft.")
    c1,c2,c3=st.columns(3)
    with c1:
        destination=st.selectbox("Destination", sorted(destinations["destination"].unique()))
        trip_type=st.radio("Who is travelling?",["Solo","With friends"],horizontal=True)
    with c2:
        travellers=st.number_input("Number of travellers",1,15,1 if trip_type=="Solo" else 3,disabled=trip_type=="Solo")
        days=st.slider("Number of days",1,7,3)
    with c3:
        budget_type=st.radio("Budget applies to",["Total group budget","Per person"],horizontal=True)
        budget=st.number_input("Your budget (₹)",min_value=1000,value=15000,step=1000)
    interests=st.multiselect("What do you want to explore?",["Nature","Adventure","Culture","Food","History","Shopping","Relaxation"],default=["Nature","Culture"])
    pace=st.select_slider("Travel pace",options=["Relaxed","Balanced","Packed"],value="Balanced")
    transport=st.multiselect("Preferred ways to travel",["Public transport","Shared taxi","Rental car / bike","Walking"],default=["Public transport","Walking"])
    if st.button("Create my itinerary",type="primary",use_container_width=True):
        if not transport: st.error("Select at least one travel method.")
        else:
            total_budget=budget if budget_type=="Total group budget" else budget*travellers
            plan=build_itinerary(destination,days,interests,pace,transport,attractions)
            estimate=estimate_budget(destination,days,travellers,transport,attractions,plan)
            st.session_state["plan"]=plan
            st.session_state["budget_summary"]={"budget":budget,"budget_type":budget_type,"travellers":travellers,"total_budget":total_budget,"estimate":estimate,"destination":destination}
    if "plan" in st.session_state:
        plan=st.session_state["plan"]; s=st.session_state["budget_summary"]
        st.divider(); st.subheader(f"Your {s['destination']} plan")
        m1,m2,m3=st.columns(3)
        m1.metric("Trip duration",f"{len(plan)} days")
        m2.metric("Travellers",s["travellers"])
        m3.metric("Estimated group cost",f"₹{s['estimate']['total']:,}")
        if s["estimate"]["total"]>s["total_budget"]: st.warning("This draft is above your budget. Reduce paid activities, use walking/public transport, or increase the budget.")
        else: st.success(f"Estimated cost is within your budget by ₹{s['total_budget']-s['estimate']['total']:,}.")
        st.subheader("Day-wise itinerary")
        for day in plan:
            with st.expander(f"Day {day['day']} · {day['theme']}",expanded=True):
                for stop in day["stops"]:
                    st.markdown(f"**{stop['name']}** · {stop['category']}  \n{stop['description']}  \n<small>Suggested travel: {stop['travel']} · Approx. entry: ₹{stop['entry']:,}/person</small>",unsafe_allow_html=True)
        st.subheader("Estimated budget breakdown")
        df=pd.DataFrame([{"Category":k.title(),"Estimated group cost (₹)":v} for k,v in s["estimate"]["breakdown"].items()])
        st.bar_chart(df.set_index("Category"))
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.caption("Estimates are illustrative planning values, not live quotes. Verify opening hours, fares, weather and attraction access before travel.")
        st.download_button("Download itinerary (CSV)",pd.DataFrame([{"Day":d["day"],"Theme":d["theme"],"Place":p["name"],"Category":p["category"],"Travel":p["travel"]} for d in plan for p in d["stops"]]).to_csv(index=False),"ctrl_trip_itinerary.csv","text/csv")

elif pages=="Destination Explorer":
    st.title("Destination explorer")
    st.write("Browse the destinations currently included in the sample catalogue.")
    image_files={"Manali":"manali.png","Jaipur":"jaipur.png","Goa":"goa.png"}
    cards=st.columns(3)
    for index,(_,r) in enumerate(destinations.iterrows()):
        with cards[index%3]:
            with st.container(border=True):
                image_path=Path(__file__).resolve().parent/"assets"/"destinations"/image_files.get(r["destination"],"")
                if image_path.is_file():
                    st.image(str(image_path),width="stretch",alt=f"Illustrative destination image for {r['destination']}")
                st.subheader(r["destination"])
                st.caption(f"{r['region']} · {r['best_for']}")
                st.write(r["description"])
    st.divider()
    selected_destination=st.selectbox("Open a destination profile",sorted(destinations.destination.astype(str).unique()))
    from src.nlp import build_model, extract_aspects, load_reviews
    destination_reviews=load_reviews()
    destination_reviews=destination_reviews[destination_reviews.destination.eq(selected_destination)].copy()
    if not destination_reviews.empty:
        profile_image=Path(__file__).resolve().parent/"assets"/"destinations"/image_files.get(selected_destination,"")
        if profile_image.is_file():
            st.image(str(profile_image),width="stretch",alt=f"Illustrative destination image for {selected_destination}")
        st.subheader(f"{selected_destination} · sample review profile")
        model=build_model()
        destination_reviews["predicted_sentiment"]=model.predict(destination_reviews.review.astype(str))
        sentiment_counts=destination_reviews.predicted_sentiment.value_counts().reindex(["positive","neutral","negative"],fill_value=0)
        positive_share=sentiment_counts["positive"]/len(destination_reviews)
        m1,m2,m3=st.columns(3)
        m1.metric("Bundled sample reviews",len(destination_reviews))
        m2.metric("Predicted positive share",f"{positive_share:.0%}")
        m3.metric("Attractions in catalogue",int((attractions.destination==selected_destination).sum()))
        st.caption("Model predictions on synthetic sample text; this share is not a real-world destination rating.")
        st.subheader("Predicted sentiment distribution")
        st.bar_chart(sentiment_counts)
        aspect_rows=[]
        for _,review_row in destination_reviews.iterrows():
            for aspect,details in extract_aspects(review_row.review,model).items():
                aspect_rows.append({"Aspect":aspect.title(),"Sentiment":details["sentiment"],"Evidence":details["evidence"][0]})
        if aspect_rows:
            st.subheader("What the sample reviews mention")
            st.dataframe(pd.DataFrame(aspect_rows).drop_duplicates(subset=["Aspect","Evidence"]).head(12),width="stretch",hide_index=True)
        st.subheader("Example source reviews")
        for _,review_row in destination_reviews.head(3).iterrows():
            with st.container(border=True):
                st.write(f"“{review_row.review}”")
                st.caption(f"Sample record {review_row.review_id} · labelled {review_row.sentiment}")
    st.caption("Destination images are AI-generated illustrations. Catalogue details are sample data; verify travel information independently.")

elif pages=="Review Intelligence":
    st.title("Travel review intelligence")
    st.write("Explore a labelled demonstration dataset or upload your own review CSV. Predictions and recommendations are based only on the selected review text.")
    from src.nlp import (ASPECTS, build_model, evaluate_model, extract_aspects, extract_entities,
                         extract_keywords, load_reviews, recommend_destinations, summarize_reviews)
    mode=st.radio("Review source",["Bundled sample dataset","Upload CSV"],horizontal=True)
    upload=st.file_uploader("CSV with a `review` column; optional `destination` column",type=["csv"]) if mode=="Upload CSV" else None
    frame=load_reviews() if mode=="Bundled sample dataset" else (pd.read_csv(upload).fillna("") if upload else None)
    if frame is not None:
        if "review" not in frame.columns: st.error("CSV must contain a 'review' column.")
        elif len(frame.dropna(subset=["review"])) < 6: st.warning("Add at least six non-empty reviews for analysis.")
        else:
            frame=frame[frame.review.astype(str).str.strip().ne("")].copy()
            if "destination" not in frame: frame["destination"]="Uploaded reviews"
            eval_data=load_reviews()
            with st.spinner("Training the TF-IDF + Logistic Regression model…"):
                evaluation=evaluate_model(eval_data)
                model=build_model(eval_data)
            st.subheader("Held-out model evaluation")
            st.caption(f"Stratified 75/25 split · fixed random seed 42 · {evaluation['train_size']} train / {evaluation['test_size']} test reviews · macro-averaged precision, recall and F1. Scores describe this small labelled sample only.")
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Accuracy",f"{evaluation['accuracy']:.1%}");m2.metric("Precision",f"{evaluation['precision']:.1%}");m3.metric("Recall",f"{evaluation['recall']:.1%}");m4.metric("Macro F1",f"{evaluation['f1']:.1%}")
            st.write("Confusion matrix · rows are actual; columns are predicted · order: negative, neutral, positive")
            st.dataframe(pd.DataFrame(evaluation["confusion_matrix"],index=evaluation["labels"],columns=evaluation["labels"]),use_container_width=True)
            predicted=model.predict(frame.review.astype(str));frame["predicted_sentiment"]=predicted
            counts=pd.Series(predicted).value_counts()
            st.subheader("Review sentiment")
            a,b,c=st.columns(3);a.metric("Reviews",len(frame));b.metric("Predicted positive",int(counts.get("positive",0)));c.metric("Predicted negative",int(counts.get("negative",0)))
            st.dataframe(frame[[c for c in ["destination","review","sentiment","predicted_sentiment"] if c in frame.columns]],use_container_width=True,hide_index=True)
            st.subheader("Aspect sentiment with source evidence")
            all_aspects={}
            for _,row in frame.iterrows():
                for aspect,info in extract_aspects(row.review,model).items():
                    all_aspects.setdefault(aspect,[]).append((info["sentiment"],info["evidence"][0],row.get("destination","")))
            if all_aspects:
                st.dataframe(pd.DataFrame([{"Aspect":a.title(),"Mentions":len(v),"Most common sentiment":pd.Series([x[0] for x in v]).mode().iloc[0],"Evidence":v[0][1]} for a,v in all_aspects.items()]),use_container_width=True,hide_index=True)
            else: st.info("No aspect terms from food, accommodation, transport, safety, scenery or activities were found.")
            left,right=st.columns(2)
            with left:
                st.subheader("TF-IDF keywords");st.write(", ".join(k for k,_ in extract_keywords(frame.review.astype(str).tolist())) or "No keywords found.")
                st.subheader("Named entities");known=set(destinations.destination.astype(str))|set(attractions.name.astype(str))
                st.write(extract_entities(" ".join(frame.review.astype(str)),known) or "No place names detected.")
            with right:
                st.subheader("Evidence grounded summary")
                summary=summarize_reviews(frame.review.astype(str).tolist(),model)
                st.write(summary["summary"])
                for item in summary["evidence"]: st.caption(f"{item['sentiment'].title()} · “{item['sentence']}”")
            st.subheader("Transparent destination recommendations")
            preferences=st.multiselect("Prioritise terms mentioned in reviews",["food","accommodation","transport","safety","scenery","activities","beach","mountain","culture","history"])
            recs=recommend_destinations(eval_data,preferences,model)
            st.caption("Score = 65% predicted positive share + 35% preference-term match rate. The score is a demo ranking from bundled sample reviews, not a forecast.")
            st.dataframe(recs,use_container_width=True,hide_index=True)
            if mode=="Upload CSV" and upload and st.checkbox("Save these uploaded reviews locally in SQLite"):
                from src.storage import save_reviews
                if st.button("Save review rows"):
                    st.success(f"Saved {save_reviews(frame)} review rows to data/ctrl_trip.sqlite3 on this computer.")

else:
    st.title("About Ctrl+Trip")
    st.write("Ctrl+Trip is an NLP-powered travel review intelligence and itinerary helper.")
    st.markdown("""
    **Current modules**
    - Destination catalogue backed by CSV data.
    - Solo and group trip planning.
    - Budget input as total group budget or per-person budget.
    - 1–7 day itinerary drafts based on interests and travel pace.
    - Attraction suggestions and preferred transport methods.
    - Estimated accommodation, food, transport and activity costs.
    - CSV itinerary export.
    - TF-IDF + Logistic Regression classifier with a reproducible held-out evaluation and confusion matrix.
    - Cleaning, tokenisation, stopword filtering and deterministic rule-based lemmatisation.
    - Aspect sentiment for food, accommodation, transport, safety, scenery and activities with source sentences.
    - TF-IDF keywords, catalogue-aware place extraction and extractive review summaries with evidence.
    - Transparent sample-data destination ranking from review sentiment and user preference terms.
    - Optional local SQLite persistence for uploaded reviews.

    **Data and limits:** Included review text is synthetic and labelled sample data, not customer feedback. Model metrics are demonstration results on a small fixed dataset. Itinerary amounts are illustrative estimates. This app does not provide live maps, prices, bookings, availability, or route times.
    """)
