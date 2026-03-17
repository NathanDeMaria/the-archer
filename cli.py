from fire import Fire

from the_archer import save_pbp_dfs, save_shot_model


if __name__ == "__main__":
    Fire(
        {
            "save-pbp-dfs": save_pbp_dfs,
            "save-shot-model": save_shot_model,
        }
    )
