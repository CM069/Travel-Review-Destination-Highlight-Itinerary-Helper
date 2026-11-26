def build_itinerary(destination,days,interests,pace,transport,attractions):
    rows=attractions[attractions.destination.str.lower()==destination.lower()].copy()
    if rows.empty: rows=attractions.copy()
    if interests:
        rows["score"]=rows.apply(lambda r: sum(i.lower() in str(r.category).lower() or i.lower() in str(r.tags).lower() for i in interests),axis=1)
        rows=rows.sort_values(["score","rating"],ascending=False)
    else: rows=rows.sort_values("rating",ascending=False)
    per_day={"Relaxed":2,"Balanced":3,"Packed":4}[pace]
    chosen=rows.head(days*per_day).to_dict("records")
    plans=[]
    for d in range(days):
        stops=[]
        for r in chosen[d*per_day:(d+1)*per_day]:
            method=next((m for m in transport if m in ["Public transport","Shared taxi","Rental car / bike","Walking"]),"Public transport")
            stops.append({"name":r["name"],"category":r["category"],"description":r["description"],"entry":int(r["entry_fee"]), "travel":method})
        plans.append({"day":d+1,"theme":(stops[0]["category"]+" & local discovery") if stops else "Flexible exploration","stops":stops})
    return plans

def estimate_budget(destination,days,travellers,transport,attractions,plan):
    selected=[p for day in plan for p in day["stops"]]
    activity=sum(x["entry"] for x in selected)*travellers
    food=900*days*travellers
    stay=1800*max(days-1,0)*max(1,(travellers+1)//2)
    daily_transport=450 if "Rental car / bike" in transport else (250 if "Shared taxi" in transport else 120)
    transit=daily_transport*days*travellers
    breakdown={"accommodation":stay,"food":food,"transport":transit,"activities":activity}
    return {"total":sum(breakdown.values()),"breakdown":breakdown}
