import asyncio

from endgame.ncaabb import NcaabbGender
from the_archer import save_pbp_dfs


if __name__ == "__main__":
    asyncio.run(save_pbp_dfs(NcaabbGender.mens))
