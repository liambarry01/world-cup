# ⚽ FIFA World Cup 2026 — AI Predictor

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

## Data Sources

| File | Source |
|------|--------|
| `fifa_ranking-2024-06-20.csv` | [Kaggle — cashncarry/fifaworldranking](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) |
| `fifa_rankings_2026.csv` | [FIFA official rankings — January 2026](https://www.fifa.com/en/world-rankings) |
| `wc_results.csv` | Tournament results 1998–2022 |
| `wc_history.csv` | WC appearances and wins per team per year |
| `wc_history_2026.csv` | Appearances and wins going into 2026 |

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
python fifa_model.py
```

This trains the Random Forest model and prints win probabilities for all 2026 qualified teams.

---

## Project Structure

```
fifa-predictor/
├── fifa_data.py               # Data preparation
├── fifa_model.ipynb              # Model training + prediction
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
