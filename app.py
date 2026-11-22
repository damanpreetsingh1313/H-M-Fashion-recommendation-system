"""H&M Season-Aware Fashion Recommender - interactive Streamlit app.   Run:  streamlit run app.py"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from src import data as D
from src.evaluate import evaluate
from src.model import COMPONENTS, DEFAULT_WEIGHTS, HybridRecommender
from src.season import NORTH, PRESETS, SEASONS, SeasonCalendar
from src.simulate import (make_customers, make_demo_catalog, simulate_reviews, simulate_transactions)

st.set_page_config(page_title="H&M Season-Aware Recommender", page_icon="👗", layout="wide")
EMOJI = {"Winter": "❄️", "Spring": "🌸", "Summer": "☀️", "Monsoon": "🌧️", "Autumn": "🍂"}

# ------------------------------------------------------------------ sidebar: data + calendar
st.sidebar.title("⚙️ Settings")
source = st.sidebar.radio("Data source", ["Demo simulator", "Real H&M data (Kaggle)"])
data_dir = st.sidebar.text_input("H&M folder", "data/hm") if source.startswith("Real") else "data/hm"
n_cust = st.sidebar.number_input("Customers to sample", 2000, 200000, 20000, 2000) if source.startswith("Real") else 1500
seed_reviews = st.sidebar.checkbox("Seed synthetic reviews (H&M has none)", True)

st.sidebar.subheader("Season calendar")
preset = st.sidebar.selectbox("Preset", list(PRESETS) + ["Custom lengths"])
if preset == "Custom lengths":
    start = st.sidebar.selectbox("Winter starts in month", list(range(1, 13)), 11)
    defaults = {"Winter": 2, "Spring": 1, "Summer": 4, "Monsoon": 3, "Autumn": 2}
    lengths = {s: st.sidebar.number_input(f"{EMOJI[s]} {s} (months)", 0, 12, defaults[s]) for s in SEASONS}
    try:
        calendar = SeasonCalendar.from_lengths(start, lengths, "custom")
    except ValueError:
        st.sidebar.error(f"Months add up to {sum(lengths.values())}, must be 12")
        st.stop()
else:
    calendar = SeasonCalendar(PRESETS[preset], preset)
learn_active = st.sidebar.checkbox("Learn demand from this calendar (use for simulated/local data)", False)
learn_cal = calendar if learn_active else SeasonCalendar(NORTH, "north")
today = st.sidebar.date_input("Date to recommend for", dt.date.today())

st.sidebar.subheader("Blend weights")
weights = {c: st.sidebar.slider(c.capitalize(), 0.0, 1.0, DEFAULT_WEIGHTS[c], 0.05) for c in COMPONENTS}

if "version" not in st.session_state:
    st.session_state.version = 0


# ------------------------------------------------------------------ cached data + model
@st.cache_data(show_spinner="Loading data...")
def load_base(source, data_dir, n_cust, version):
    if source.startswith("Real"):
        if not D.hm_files_present(data_dir):
            return None
        return D.load_real_hm(data_dir, int(n_cust))
    art = make_demo_catalog()
    cust = make_customers(int(n_cust))
    tx = simulate_transactions(art, cust, SeasonCalendar(NORTH), "2018-10-01", "2020-09-22", 60000, seed=1)
    return art, tx


base = load_base(source, data_dir, n_cust, 0)
if base is None:
    st.error(f"Could not find articles.csv, customers.csv, transactions_train.csv in '{data_dir}'. "
             "Download them from the Kaggle 'H&M Personalized Fashion Recommendations' competition, "
             "or switch to the Demo simulator.")
    st.stop()
articles, base_tx = base


@st.cache_resource(show_spinner="Training models (TF-IDF + SVD + seasonal demand)...")
def build_model(source, data_dir, n_cust, learn_key, seed_rev, version):
    tx_all = pd.concat([base_tx, D.load_user_tx()], ignore_index=True)
    rev = [D.load_user_reviews()]
    if seed_rev:
        rev.insert(0, simulate_reviews(base_tx, articles, min(8000, len(base_tx) // 8)))
    rev = pd.concat(rev, ignore_index=True)
    return HybridRecommender(SeasonCalendar(NORTH), SeasonCalendar(dict(learn_key))).fit(articles, tx_all, rev)


model = build_model(source, data_dir, n_cust, learn_cal.key(), seed_reviews, st.session_state.version)
model.calendar = calendar                     # serving calendar can change instantly, no retraining
season = calendar.season_of(today)

st.title("👗 H&M Season-Aware Fashion Recommender")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current season", f"{EMOJI[season]} {season}")
c2.metric("Season lengths (months)", " / ".join(str(v) for v in calendar.lengths().values()), ", ".join(s[:3] for s in SEASONS))
c3.metric("Articles / customers", f"{len(model.articles):,} / {len(model.hist):,}")
c4.metric("Reviews", f"{len(model.reviews):,}")

# ------------------------------------------------------------------ who is the user?
st.subheader("1 · Who are you?")
mode = st.radio("Profile", ["Existing customer", "New customer (cold start)"], horizontal=True)
liked, user_id = [], None
if mode == "Existing customer":
    active = sorted(model.hist, key=lambda u: -len(model.hist[u][0]))[:300]
    user_id = st.selectbox("Customer", active, format_func=lambda u: f"{str(u)[:14]}…  ({len(model.hist[u][0])} items)")
else:
    user_id = st.text_input("Pick a name/ID (used for your reviews & purchases)", "guest")
    pool = model.recommend(None, [], today, k=150, weights={"season": 1, "popularity": 0.5, "collab": 0, "content": 0, "reviews": 0.2})[0]
    lab = dict(zip(pool.article_id, pool.prod_name + " · " + pool.product_type_name + " · " + pool.colour_group_name))
    liked = st.multiselect(f"Pick items you like (seasonal picks for {season})", list(lab), format_func=lab.get)
with st.expander("Filters"):
    f1, f2 = st.columns(2)
    filters = {"product_group_name": f1.multiselect("Product group", sorted(articles.product_group_name.dropna().unique())),
               "colour_group_name": f2.multiselect("Colour", sorted(articles.colour_group_name.dropna().unique()))}
k = st.slider("How many recommendations", 4, 24, 12, 4)

tab_rec, tab_rev, tab_buy, tab_sim, tab_an = st.tabs(
    ["🎯 Recommendations", "⭐ Rate & review", "🛍️ Log a purchase", "🌦️ Season studio", "📊 Analytics"])

# ------------------------------------------------------------------ recommendations
with tab_rec:
    recs, s = model.recommend(user_id, liked, today, k, weights, filters)
    st.caption(f"Recommendations for **{today}** → {EMOJI[s]} **{s}**. Change the date, calendar or weights in the sidebar and they update live.")
    cols = st.columns(4)
    for i, r in recs.iterrows():
        with cols[i % 4]:
            img = D.image_path(data_dir, int(r.article_id)) if source.startswith("Real") else None
            if img:
                st.image(img, use_container_width=True)
            st.markdown(f"**{r.prod_name}**  \n{r.product_type_name} · {r.colour_group_name}")
            st.caption(f"{r.why} · score {r.score:.2f}")
    if len(recs):
        st.markdown("**Why these items? (score contribution per component)**")
        chart = recs[[f"{c}_pts" for c in COMPONENTS]].rename(columns=lambda c: c.replace("_pts", ""))
        chart.index = [f"{i + 1}. {n}" for i, n in enumerate(recs.prod_name)]
        st.bar_chart(chart)
        st.dataframe(recs[["article_id", "prod_name", "product_type_name", "colour_group_name", "score", "why"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No items match your filters.")

# ------------------------------------------------------------------ reviews
with tab_rev:
    st.write("Reviews change item reputation (rating + text sentiment) and the recommendations immediately. "
             "4-5★ counts as a positive taste signal, 1-2★ hides the item for you.")
    choices = pd.concat([recs, articles.sample(min(200, len(articles)), random_state=1)]).drop_duplicates("article_id")
    lab2 = dict(zip(choices.article_id, choices.prod_name + " · " + choices.product_type_name + " · " + choices.colour_group_name))
    aid = st.selectbox("Item", list(lab2), format_func=lab2.get)
    rating = st.slider("Rating", 1, 5, 4)
    text = st.text_area("Your review", placeholder="Soft fabric, perfect for the summer…")
    if st.button("Submit review"):
        row = pd.DataFrame([dict(customer_id=user_id, article_id=aid, rating=rating, text=text, date=pd.Timestamp(today))])
        D.append_rows("reviews.csv", row)
        model.add_review(user_id, aid, rating, text, today)
        st.success("Saved! Open the Recommendations tab to see the updated list.")
    shown = model.reviews[model.reviews.article_id == aid].tail(5)
    st.markdown(f"**Latest reviews for this item** ({len(model.reviews[model.reviews.article_id == aid])} total)")
    st.dataframe(shown[["rating", "text", "date"]], hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ purchases
with tab_buy:
    aid_b = st.selectbox("Item bought", list(lab2), format_func=lab2.get, key="buy")
    d_b = st.date_input("Purchase date", today, key="bd")
    if st.button("Log purchase"):
        D.append_rows("new_tx.csv", pd.DataFrame([dict(t_dat=pd.Timestamp(d_b), customer_id=user_id, article_id=aid_b, price=0.03)]))
        model.add_purchase(user_id, aid_b, d_b)
        st.success("Purchase logged — your taste profile was updated. It is also included in the next full retrain.")

# ------------------------------------------------------------------ season studio / simulator
with tab_sim:
    st.write("Season months under the active calendar:")
    cal_df = pd.DataFrame({"month": [dt.date(2000, m, 1).strftime("%b") for m in range(1, 13)],
                           "season": [f"{EMOJI[calendar.month_map[m]]} {calendar.month_map[m]}" for m in range(1, 13)]}).set_index("month").T
    st.dataframe(cal_df, use_container_width=True)
    st.subheader("Generate a new dataset for this calendar")
    st.write("Creates season-aware purchases (items fit the season *as you defined it*), appends them to your data and retrains. "
             "Tick **'Learn demand from this calendar'** in the sidebar so the model learns the new season lengths from this data.")
    a, b, c = st.columns(3)
    start_d = a.date_input("From", dt.date(2020, 9, 23))
    end_d = b.date_input("To", dt.date(2021, 9, 22))
    n_new = c.number_input("Transactions", 1000, 200000, 20000, 1000)
    if st.button("Generate & retrain"):
        pool_c = list(model.hist)
        new = simulate_transactions(articles, pool_c, calendar, start_d, end_d, int(n_new), seed=int(dt.datetime.now().timestamp()) % 10000)
        D.append_rows("new_tx.csv", new)
        st.session_state.version += 1
        st.rerun()
    cA, cB = st.columns(2)
    if cA.button("Retrain on all current data"):
        st.session_state.version += 1
        st.rerun()
    if cB.button("🗑️ Reset my reviews & generated data"):
        D.reset_user_data()
        st.session_state.version += 1
        st.rerun()

# ------------------------------------------------------------------ analytics
with tab_an:
    st.subheader("Learned seasonal demand (lift > 1 = sells more than average in that season)")
    st.caption("Learned from transactions bucketed with the learning calendar. H&M data is Swedish, so the default learning calendar is Northern Hemisphere.")
    top = model.season_table.sort_values("n_sales", ascending=False).head(25)
    st.dataframe(top.style.background_gradient(subset=SEASONS, cmap="RdYlGn", vmin=0.3, vmax=2.0).format("{:.2f}", subset=SEASONS),
                 use_container_width=True)
    st.subheader("Review sentiment")
    if len(model.reviews):
        st.bar_chart(model.reviews.rating.value_counts().sort_index())
    st.subheader("Offline evaluation (temporal hold-out)")
    hd = st.slider("Hold-out window (days)", 7, 56, 28)
    if st.button("Run evaluation"):
        with st.spinner("Training on the past, predicting the held-out weeks…"):
            res = evaluate(articles, pd.concat([base_tx, D.load_user_tx()]), model.reviews, learn_cal, learn_cal, 400, hd)
        st.dataframe(res.style.format("{:.4f}"), use_container_width=True)
        st.bar_chart(res["MAP@12"])
        st.caption(f"{res.attrs['n_users']} users evaluated. Evaluation serves with the learning calendar so seasons match the data.")
