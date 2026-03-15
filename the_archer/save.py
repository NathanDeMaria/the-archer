from collections.abc import AsyncIterator
from pathlib import Path
import warnings

import pandas as pd
from endgame_aws import get_pbp_store, read_seasons, Config
from endgame.types import Season
from endgame.ncaabb import NcaabbGender


_DATA_DIR = Path(__file__).parent / "data"


async def save_pbp_dfs(league: NcaabbGender) -> None:
    month_dfs = []
    async for df in _load_pbp_dfs(league):
        if len(df) == 0:
            continue
        this_month = _year_month(df["date"].iloc[0])
        if not month_dfs or _year_month(month_dfs[-1]["date"].iloc[0]) == this_month:
            month_dfs.append(df)
        else:
            _save_month_dfs(league, month_dfs)
            month_dfs = []
    _save_month_dfs(league, month_dfs)


def _year_month(date: pd.Timestamp) -> str:
    return f"{date.year}_{date.month:02d}"


def _save_month_dfs(league: NcaabbGender, month_dfs: list[pd.DataFrame]) -> None:
    if not month_dfs:
        return
    month = _year_month(month_dfs[0]["date"].iloc[0])
    print(f"Saving {month}...")
    month_df = pd.concat(month_dfs, ignore_index=True)
    if "favoredTeam.winProbability" in month_df.columns:
        month_df = month_df.assign(
            **{
                "favoredTeam.winProbability": pd.to_numeric(
                    month_df["favoredTeam.winProbability"], errors="coerce"
                )
            }
        )
    league_dir = _DATA_DIR / league.name
    league_dir.mkdir(exist_ok=True, parents=True)
    month_df.to_parquet(league_dir / f"{month}.parquet")


async def _load_pbp_dfs(league: NcaabbGender) -> AsyncIterator[pd.DataFrame]:
    seasons = [season async for season in _read_all_seasons(league)]
    games = {
        game.game_id: game
        for season in seasons
        for week in season.weeks
        for game in week.games
    }

    async with get_pbp_store() as store:
        async for pbps in store.load_all(league):
            for pbp in pbps:
                game = games.get(pbp["game_id"])
                if not game:
                    warnings.warn(
                        f"Game ID {pbp['game_id']} not found in seasons data; skipping"
                    )
                    continue
                yield pd.json_normalize(pbp["plays"]).assign(
                    game_id=pbp["game_id"],
                    home=game.home,
                    away=game.away,
                    date=game.date,
                )


async def _read_all_seasons(gender: NcaabbGender) -> AsyncIterator[Season]:
    # TODO: check S3 instead of this hard-coded start/end
    for year in range(2010, 2026):
        seasons = await read_seasons(
            Config.init_from_file().bucket, f"seasons/{year}/{gender.name}.pkl"
        )
        for season in seasons:
            yield season
