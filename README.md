# ⚽ FIFA World Cup 2026 Predictor

A machine learning model that predicts the winner of the 2026 FIFA World Cup using real historical FIFA ranking data and Random Forest classification.

**Model accuracy: AUC 0.869** (tested across 7 tournaments, 1998–2022)

---

## Results

| Team | FIFA Rank | Win Probability |
|------|-----------|----------------|
| 🇦🇷 Argentina | #2 | 12.0% |
| 🇫🇷 France | #3 | 11.4% |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | #4 | 10.5% |
| 🇪🇸 Spain | #1 | 9.5% |
| 🇮🇹 Italy | #13 | 9.0% |

---

## How It Works

The model is trained on data from all 7 World Cups (1998–2022), covering 224 teams. For each team it uses:

- **FIFA ranking** going into the tournament
- **FIFA points** (more granular than rank)
- **Confederation** (UEFA / CONMEBOL / CAF / AFC / CONCACAF)
- **World Cup appearances** (tournament experience)
- **Previous World Cup wins** (pedigree)
- **Host nation** flag

A Random Forest classifier with Leave-One-Tournament-Out cross-validation is used so the model is always tested on tournaments it has never seen.

---

## Data Pipeline

Each row in the training data is **one team entering one World Cup**. For the 7 tournaments from 1998–2022 that gives 206 rows. Every feature is something you'd know before the tournament starts:

| Feature | What it captures | Example (England 2022) |
|---|---|---|
| `fifa_rank` | How good FIFA says they are right now | 5 |
| `fifa_points` | More granular version of rank | 1728.47 |
| `confederation_code` | Which continent they qualify from | 0 (UEFA) |
| `wc_appearances` | How many World Cups they've been to before this one | 16 |
| `prev_wc_wins` | How many times they've won before this one | 1 |
| `is_host` | Whether they're hosting | 0 |
| **`won`** (target) | Did they win this tournament | 0 |

The history columns count *up to but not including* the current tournament, so there's no information leaking from the future. To predict 2026, the exact same row is built for each qualified team using the latest FIFA rankings and updated history, and the model returns a win probability for each one.

`fifa_data.py` joins these together and outputs `world_cup_training.csv` and `world_cup_2026.csv`.

---

## Data Sources

No single CSV has everything the model needs. Each one provides a different piece of a team's profile:

| File | Contains | Used for |
|------|----------|----------|
| `fifa_ranking-2024-06-20.csv` | Historical FIFA rankings from [Kaggle](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | Gives each team's `fifa_rank` and `fifa_points` at the time of each past World Cup |
| `wc_results.csv` | Who was in each tournament (1998–2022) and how far they got | Provides the team list per year, `is_host`, and the `won` target label |
| `wc_history.csv` | Cumulative World Cup appearances and wins per team per year | Gives the `wc_appearances` and `prev_wc_wins` features (counted *before* that year, so no leakage) |
| `fifa_rankings_2026.csv` | January 2026 [FIFA rankings](https://www.fifa.com/en/world-rankings) | Same as the Kaggle file but for the current year, since Kaggle only goes to 2024 |
| `wc_history_2026.csv` | Appearances and wins going into 2026 | Same structure as `wc_history.csv` but condensed to one row per team with cumulative counts through 2022, since no year column is needed for a single tournament |

`fifa_data.py` reads all five, joins rankings + results + history for each tournament year, and outputs one clean training file and one prediction file.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/liambarry01/world-cup.git
cd world-cup
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run data preparation

```bash
python fifa_data.py
```

This reads the Kaggle rankings and all CSV files to generate:
- `world_cup_training.csv` — 206 rows of historical training data
- `world_cup_2026.csv` — 2026 tournament teams with current rankings

### 4. Run the model

```bash
jupyter notebook fifa_model.ipynb
```

Then open `fifa_model.ipynb` in your browser and run all cells.

This trains the Random Forest model and prints win probabilities for all 2026 qualified teams.

---

## Project Structure

```
fifa-predictor/
├── fifa_data.py               # Data preparation
├── fifa_model.ipynb           # Model training + prediction
├── fifa_ranking-2024-06-20.csv  # Kaggle historical rankings
├── fifa_rankings_2026.csv     # January 2026 FIFA rankings
├── wc_results.csv             # Tournament results 1998–2022
├── wc_history.csv             # WC appearances/wins per year
├── wc_history_2026.csv        # History going into 2026
└── requirements.txt
```

---

## Requirements

```
pandas
numpy
scikit-learn
requests
optuna
shap
skimpy
matplotlib
```

---

## Potential Improvements

- **Simulate the tournament bracket** — rather than predicting outright winners, simulate match-by-match outcomes to produce path-dependent probabilities
- **Add recent form features** — incorporate win/draw/loss record from the 12 months prior to each tournament, not just FIFA rank/points
- **Head-to-head records** — include historical matchup data between teams as an additional feature
- **Expand training data** — include pre-1998 tournaments (1930–1994) to increase the training set, despite older ranking data being less reliable
- **Calibrate probabilities** — apply Platt scaling or isotonic regression so that predicted probabilities better reflect true likelihoods

---

## Stage Reference

Stage values used in `wc_results.csv`:

| Value | Round |
|-------|-------|
| 1 | Group stage exit |
| 2 | Round of 16 |
| 3 | Quarter-final |
| 4 | Semi-final |
| 5 | Third place |
| 6 | Runner-up (final) |
| 7 | Winner 🏆 |
