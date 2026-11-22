# H&M Season-Aware Fashion Recommender

Interactive hybrid recommender trained on the real **H&M Personalized Fashion Recommendations** data
(Kaggle). It takes user input (profile, purchases, star ratings + review text), and its recommendations
change with the **season calendar**: you can define how long winter, summer, monsoon etc. last.

## Setup
```bash
pip install -r requirements.txt
# Real data: download articles.csv, customers.csv, transactions_train.csv (and optionally images/)
# from https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data
# and put them in data/hm/
streamlit run app.py
```
No Kaggle files yet? Choose **Demo simulator** in the sidebar - the app works out of the box.

## Models (all library-based)
| Component | Library | What it does |
|---|---|---|
| Collaborative | `sklearn.TruncatedSVD` | latent factors from user x item matrix; users folded in live |
| Content | `sklearn.TfidfVectorizer` | text similarity to your taste profile (recency-weighted) |
| Seasonal | numpy/pandas | learned seasonal lift per product type + keyword warmth prior |
| Reviews | `vaderSentiment` | Bayesian-shrunk rating + sentiment per item |
| Popularity | pandas | last 8 weeks trending |

Final score = weighted blend (sliders in the sidebar); the app shows per-item score contributions.

## How the "dynamic" parts work
* **Season calendar**: presets (India, Northern/Southern Hemisphere) or custom month lengths
  (e.g. Winter 2, Spring 1, Summer 4, Monsoon 3, Autumn 2). Changing it re-ranks instantly.
* **Reviews / purchases**: saved to `user_data/*.csv`, applied instantly (taste profile + item reputation);
  a full retrain happens via the Season studio button.
* **Dataset generator**: Season studio creates new season-aware transactions for any date range under
  your calendar, appends them and retrains (tick "Learn demand from this calendar" for it to learn them).
* **Evaluation**: temporal hold-out, MAP@12 / Recall@12 vs popularity, content, collaborative, hybrid
  with and without season.

## Honest notes
* The H&M dataset has **no reviews**; reviews are collected in the app (optionally seeded with synthetic ones).
* H&M data is Swedish, so seasonal demand is learned with the Northern calendar by default and then
  *served* with whichever calendar you choose (Monsoon borrows Summer/Autumn patterns).
* Start with 20k sampled customers; raise it if you have RAM (~16 GB handles 100k+).
