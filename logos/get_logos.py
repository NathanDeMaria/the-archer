"""Given all seasons, get the ESPN team ids to match team names, then fetch logos."""

import asyncio
import json
from pathlib import Path

import aiofiles
import fire
from bs4 import BeautifulSoup
from endgame.ncaabb import NcaabbGender

import aiohttp

from the_archer.save import read_all_seasons

_LOGO_DIR = Path(__file__).parent / "data"
_LOGO_URL = "https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/{id}.png&w=96&h=96&scale=crop&cquality=40&location=origin"


async def _get_ids(
    session: aiohttp.ClientSession, league: NcaabbGender, game_id: str
) -> tuple[str, str] | None:
    url = f"https://www.espn.com/{league.name}-college-basketball/playbyplay/_/gameId/{game_id}"

    async with session.get(url) as response:
        if response.status == 404:
            return None
        response.raise_for_status()
        raw = await response.text()
    soup = BeautifulSoup(raw, "html.parser")
    scripts = soup.select("script")
    fit_script = next(script for script in scripts if "espnfitt" in script.text)
    prefix = "window['__espnfitt__']="
    interesting_json = fit_script.text.split(prefix)[-1][:-1]
    data = json.loads(interesting_json)

    teams = data["page"]["content"]["gamepackage"]["prsdTms"]
    return teams["home"]["id"], teams["away"]["id"]


async def _fetch_logo(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    team: str,
    team_id: str,
    logo_root: Path,
) -> None:
    async with sem:
        url = _LOGO_URL.format(id=team_id)
        async with session.get(url) as r:
            if r.status == 404:
                print(f"No logo for {team} ({team_id})")
                return
            r.raise_for_status()
            content = await r.read()
    async with aiofiles.open(logo_root / f"{team}.png", "wb") as f:
        await f.write(content)


async def _main(league: str) -> None:
    league_enum = NcaabbGender[league]
    mapped_teams = dict()
    async with aiohttp.ClientSession() as session:
        async for season in read_all_seasons(league_enum):
            for week in season.weeks:
                for game in week.games:
                    if game.home in mapped_teams and game.away in mapped_teams:
                        continue
                    ids = await _get_ids(session, league_enum, game.game_id)
                    if ids is None:
                        continue
                    home_id, away_id = ids
                    mapped_teams[game.home] = home_id
                    mapped_teams[game.away] = away_id

    logo_root = _LOGO_DIR / league
    logo_root.mkdir(exist_ok=True, parents=True)
    sem = asyncio.Semaphore(10)
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            *[
                _fetch_logo(session, sem, team, team_id, logo_root)
                for team, team_id in mapped_teams.items()
            ]
        )

    (logo_root / "mapping.json").write_text(json.dumps(mapped_teams))


def main(league: str) -> None:
    asyncio.run(_main(league))


if __name__ == "__main__":
    fire.Fire(main)
