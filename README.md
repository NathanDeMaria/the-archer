# the-archer

Read basketball play-by-play data. First use case: shot probability model.

1. Load the latest plays with [endgame-aws](https://github.com/NathanDeMaria/EndGame/blob/fa71020e5f77362b9199de06d649cfe68f2369ae/py-endgame-aws/main.py#L135)
    - Use some flavor of [this py-launcher job](https://github.com/NathanDeMaria/EndGame/blob/9f6450d1a28e9cadeb3dabfc26f2f25f8cca208e/py-launcher/endgame_launcher/plays_backfill.py)
1. Pull+parse the play-by-play data saved by EndGame into `.parquet` files
    - `uv run python cli.py save-pbp-dfs`
1. Build the model for the league
    - `uv run python cli.py save-shot-model --league=mens`
1. Use the model to get this season's shooting ratings in [`team_shooting_ratings.ipynb`](./team_shooting_ratings.ipynb)
