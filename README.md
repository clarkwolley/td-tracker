# 🏈 TD Tracker

Predict which NFL players will score a touchdown each week — then grade yourself automatically.

TD Tracker is a fully automated ML pipeline that pulls live NFL data, trains a calibrated model, publishes predictions to Telegram, and sends a report card after the games are played.

## How it works

```
Every Tuesday at 8 AM (cron):

  ┌──────────────────────────────────────────────────┐
  │  1. 📊  Pull fresh data (nflverse + ESPN)        │
  │  2. 📝  Grade last week's predictions            │
  │  3. 🏋️  Retrain model on all available data      │
  │  4. 🎯  Predict upcoming week's TD scorers       │
  │  5. 📱  Send results to Telegram                 │
  └──────────────────────────────────────────────────┘
```

## Quick start

```bash
# Clone & setup
git clone git@github.com:clarkwolley/td-tracker.git
cd td-tracker
python -m venv venv && source venv/bin/activate
pip install pandas scikit-learn python-dotenv requests pyarrow

# Configure Telegram (optional)
cp .env.example .env   # edit with your bot token

# Run the pipeline
python -m src.automation
```

## Usage

```bash
# Full weekly pipeline (grade + retrain + predict + notify)
python -m src.automation

# Backtest any historical week
python -m src.automation --backtest 2025 10

# Force model retrain
python -m src.automation --retrain

# Skip Telegram notifications
python -m src.automation --no-notify

# Custom number of predictions
python -m src.automation --top-n 25

# Install/check/remove the weekly cron job
./scripts/setup_cron.sh install
./scripts/setup_cron.sh status
./scripts/setup_cron.sh remove
```

## Sample output

**Predictions (Telegram):**
```
🏈 TD Predictions — 2025 Week 10

 1. 🔥 B.Nix (QB, DEN) — 91%
 2. 🔥 D.Jones (QB, IND) — 89%
 3. 🔥 L.Jackson (QB, BAL) — 88%
 ...
```

**Report card (Telegram):**
```
📊 Report Card — 2025 Week 10

Grade: A 🔥
Overall: 14/15 hit (93%)
Top 5:   5/5 (100%)
```

## Data sources

| Source | Seasons | What | Latency |
|---|---|---|---|
| [nflverse](https://github.com/nflverse/nflverse-data) | 2021–2024 | Player stats (parquet) | ~24hr post-game |
| ESPN API | 2025+ (auto-detected) | Game boxscores, odds | **Real-time** |
| nflverse players table | All-time | ESPN ↔ nflverse ID crosswalk | Always current |

When nflverse hasn't published stats for the current season, the pipeline **automatically falls back to ESPN** — fetching game-by-game boxscores, mapping player IDs across systems, and caching the results.

## Model

- **Algorithm**: HistGradientBoosting + isotonic calibration
- **Features**: 29 (rolling stats, streaks, Vegas lines, opponent defense)
- **Train/test**: Temporal split — always trains on past seasons, tests on the latest
- **Data leakage**: Zero. Every feature is shifted/lagged. 20 unit tests enforce it.

### Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.753 |
| Accuracy | 78.2% |
| Backtest (top 15) | 85–93% |
| Calibration | Predicted 84% → actual 85% |

### Top features (permutation importance)

```
 1. roll4_attempts          +0.044
 2. roll4_targets           +0.011
 3. roll4_carries           +0.009
 4. roll4_passing_yards     +0.008
 5. roll4_rushing_yards     +0.007
 6. td_rate                 +0.007
 7. team_spread             +0.004
 8. roll4_receptions        +0.003
 9. roll4_receiving_yards   +0.002
10. total_line              +0.002
```

## Project structure

```
td-tracker/
├── src/
│   ├── config.py                # Centralised settings & secrets
│   ├── data/
│   │   ├── nflverse.py          # Historical stats (parquet/CSV)
│   │   └── espn.py              # Real-time ESPN API + ID crosswalk
│   ├── features/
│   │   ├── builder.py           # Feature matrix orchestrator
│   │   ├── rolling.py           # Rolling averages, streaks, rates
│   │   └── context.py           # Game context (spread, defense, home/away)
│   ├── models/
│   │   └── td_model.py          # Train, evaluate, calibrate, save/load
│   ├── predictions/
│   │   └── engine.py            # Score players, rank, format output
│   ├── notifications/
│   │   └── telegram.py          # Telegram bot integration
│   └── automation/
│       ├── pipeline.py          # Weekly orchestration logic
│       ├── grading.py           # Grade predictions vs actuals
│       └── storage.py           # Save/load prediction CSVs
├── tests/
│   └── test_features.py         # 20 tests (leakage, correctness, edge cases)
├── scripts/
│   └── setup_cron.sh            # Install/remove/check cron job
└── data/
    ├── cache/                   # ESPN parquet cache (gitignored)
    └── predictions/             # Weekly prediction CSVs (gitignored)
```

## Configuration

Create a `.env` file at the project root:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

**Getting a Telegram bot token:**
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot`, pick a name
3. Copy the token into `.env`

**Finding your chat ID:**
The pipeline auto-discovers it — just send any message to your bot, then run:
```bash
python -m src.automation
```

## Tests

```bash
python -m pytest tests/ -v
```

All 20 tests validate feature correctness, data leakage prevention, and edge cases — no network calls required.

## License

MIT
