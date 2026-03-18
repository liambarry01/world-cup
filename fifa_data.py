"""
Data pipeline for FIFA World Cup 2026 predictions.

Builds two datasets:
1. world_cup_training.csv — one row per team per tournament (1998–2022, ~206 rows).
   Each row is a team's profile *before* that World Cup started, plus whether they won.
2. world_cup_2026.csv — the same profile for each 2026 qualified team, ready for prediction.

Features per row:
- fifa_rank / fifa_points   — FIFA ranking at the time of the tournament
- confederation_code        — which continent (UEFA, CONMEBOL, etc.)
- wc_appearances            — how many World Cups they'd been to *before* this one
- prev_wc_wins              — how many times they'd won *before* this one
- is_host                   — whether they're hosting

Target (training only):
- won                       — 1 if the team won that World Cup, 0 otherwise

Everything is knowable before kickoff, so there's no data leakage from the future.
"""

import pandas as pd

# Six continental confederations in football for World Cup qualification
CONF_MAP = {"UEFA": 0, "CONMEBOL": 1, "CAF": 2, "AFC": 3, "CONCACAF": 4, "OFC": 5}

WC_START_DATES = {
    1998: "1998-06-10",
    2002: "2002-05-31",
    2006: "2006-06-09",
    2010: "2010-06-11",
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}

rankings_all = pd.read_csv("fifa_ranking-2024-06-20.csv", parse_dates=["rank_date"])

# FIFA rankings lookup
def get_rankings_at_date(target_date):
    target = pd.Timestamp(target_date)
    closest = rankings_all[rankings_all["rank_date"] <= target]["rank_date"].max()
    df = rankings_all[rankings_all["rank_date"] == closest].copy()
    df = df.rename(
        columns={
            "country_full": "team",
            "country_abrv": "country_code",
            "total_points": "fifa_points",
            "rank": "fifa_rank",
        }
    )
    return df[["team", "country_code", "fifa_rank", "fifa_points", "confederation"]]


# wc_results: one row per team per tournament — final stage reached and whether they won (target labels)
# wc_history: one row per team per tournament — appearances and wins *going into* that year ('pedigree' features, no leakage)
wc_results = pd.read_csv("wc_results.csv")
wc_history = pd.read_csv("wc_history.csv")
print(f"WC results: {len(wc_results)} rows | WC history: {len(wc_history)} rows")

# Build training data — join rankings + history + results for each tournament
rows = []
for year, date in WC_START_DATES.items():
    rankings = get_rankings_at_date(date)
    rank_lookup = {
        row['team']: (row["fifa_rank"], row["fifa_points"], row["confederation"])  # lookup: team → rank/points/confederation
        for idx, row in rankings.iterrows()
    }

    year_results = wc_results[wc_results["year"] == year]
    year_history = wc_history[wc_history["year"] == year].set_index("team")

    for idx, result in year_results.iterrows():
        team = result['team']
        if team not in rank_lookup:
            continue
        fifa_rank, fifa_points, confederation = rank_lookup[team]
        hist = year_history.loc[team] if team in year_history.index else None
        appearances = hist["wc_appearances"] if hist is not None else 0
        prev_wins = hist["prev_wc_wins"] if hist is not None else 0

        rows.append(
            {
                "team": team,
                "year": year,
                "fifa_rank": fifa_rank,
                "fifa_points": fifa_points,
                "confederation": confederation,
                "confederation_code": CONF_MAP.get(confederation, 4),
                "wc_appearances": appearances,
                "prev_wc_wins": prev_wins,
                "is_host": int(result['is_host']),
                "stage_reached": int(result['stage_reached']),
                "won": int(result['won']),
            }
        )

training_df = pd.DataFrame(rows)
training_df.to_csv("world_cup_training.csv", index=False)
print(f"\nTraining data: {len(training_df)} rows across {training_df['year'].nunique()} tournaments")
print(f"Winners: {training_df[training_df['won']==1]['team'].tolist()}")
print(training_df[["team", "year", "fifa_rank", "confederation", "stage_reached", "won"]].head(15).to_string(index=False))

# Build 2026 prediction data
rankings_2026 = pd.read_csv("fifa_rankings_2026.csv")  # January 2026 rankings

# Load 2026 history from a separate CSV (wc_history_2026.csv)
# This file has one row per team with their total appearances and wins going into 2026
try:
    history_2026 = pd.read_csv("wc_history_2026.csv").set_index("team")
except FileNotFoundError:
    # fallback: use 2022 history as proxy
    history_2026 = wc_history[wc_history["year"] == 2022].set_index("team")

HOSTS_2026 = {"United States", "Canada", "Mexico"}

rows_2026 = []
for idx, row in rankings_2026.iterrows():
    team = row['team']
    hist = history_2026.loc[team] if team in history_2026.index else None
    appearances = hist["wc_appearances"] if hist is not None else 0
    prev_wins = hist["prev_wc_wins"] if hist is not None else 0

    rows_2026.append(
        {
            "team": team,
            "country_code": row['country_code'],
            "fifa_rank": row['fifa_rank'],
            "fifa_points": row['fifa_points'],
            "confederation": row['confederation'],
            "confederation_code": CONF_MAP.get(row['confederation'], 4),
            "wc_appearances": appearances,
            "prev_wc_wins": prev_wins,
            "is_host": int(team in HOSTS_2026),
        }
    )

df_2026 = pd.DataFrame(rows_2026).sort_values("fifa_rank").reset_index(drop=True)
df_2026.to_csv("world_cup_2026.csv", index=False)
print(f"\n2026 teams: {len(df_2026)}")
print(df_2026[["team", "fifa_rank", "confederation", "wc_appearances", "is_host"]].head(15).to_string(index=False))