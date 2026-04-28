# Hogan Dependency Profiles

Hogan has three install profiles so the VPS runtime stays small while research
machines can opt into heavier tooling.

## Runtime

```bash
pip install -r requirements.txt
docker build -t hogan-bot:runtime .
```

Use for paper/live runtime, SQLite/Timescale candle access, standard sklearn
models, monitoring, and operational scripts.

## Modeling

```bash
pip install -r requirements.txt
pip install -r requirements-modeling.txt
docker build --build-arg INSTALL_MODELING=true -t hogan-bot:modeling .
```

Use for XGBoost, LightGBM, Optuna tuning, and MLflow governance.

## RL

```bash
pip install -r requirements.txt
pip install -r requirements-rl.txt
docker build --build-arg INSTALL_RL=true -t hogan-bot:rl .
```

Use for PPO training/tuning and RL inference images.

## Full Research

```bash
pip install -r requirements.txt
pip install -r requirements-modeling.txt
pip install -r requirements-rl.txt
docker build --build-arg INSTALL_MODELING=true --build-arg INSTALL_RL=true -t hogan-bot:research .
```

Use only on research hosts. This pulls the largest dependency set.

## CI

The default CI path validates the runtime profile. Manual GitHub Actions
`workflow_dispatch` runs validate the modeling, RL, and full research profiles.
