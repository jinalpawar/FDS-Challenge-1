import pandas as pd
from pathlib import Path

COMPETITION_NAME = "fds-pokemon-battles-prediction-2025"


def get_train_test(prefix):

    path = Path(prefix, "input") / COMPETITION_NAME

    train = pd.read_json(path / "train.jsonl", lines=True)
    test = pd.read_json(path / "test.jsonl", lines=True)

    return (train, test)


def effects(df):
    all_effects = set()

    for _, curr_battle in df.iterrows():
        all_effects.update(
            [
                turn["p1_pokemon_state"]["effects"][0]
                for turn in curr_battle.battle_timeline
            ]
        )
        all_effects.update(
            [
                turn["p2_pokemon_state"]["effects"][0]
                for turn in curr_battle.battle_timeline
            ]
        )
    return all_effects


def pokemon_types(df):
    all_pokemon_types = set()

    for _, curr_battle in df.iterrows():
        all_pokemon_types.update(
            [
                turn["p1_move_details"]["type"]
                for turn in curr_battle.battle_timeline
                if turn["p1_move_details"]
            ]
        )
        all_pokemon_types.update(
            [
                turn["p2_move_details"]["type"]
                for turn in curr_battle.battle_timeline
                if turn["p2_move_details"]
            ]
        )
    return all_pokemon_types


def categories(df):
    all_categories = set()

    for _, curr_battle in df.iterrows():
        all_categories.update(
            [
                turn["p1_move_details"]["category"]
                for turn in curr_battle.battle_timeline
                if turn["p1_move_details"]
            ]
        )
        all_categories.update(
            [
                turn["p2_move_details"]["category"]
                for turn in curr_battle.battle_timeline
                if turn["p2_move_details"]
            ]
        )
    return all_categories


def pokemon_names(df):
    all_pokemons = set()

    for _, curr_battle in df.iterrows():
        all_pokemons.update(
            [turn["p1_pokemon_state"]["name"] for turn in curr_battle.battle_timeline]
        )
        all_pokemons.update(
            [turn["p2_pokemon_state"]["name"] for turn in curr_battle.battle_timeline]
        )
    return all_pokemons


def statuses(df):
    all_statuses = set()

    for _, curr_battle in df.iterrows():
        all_statuses.update(
            [turn["p1_pokemon_state"]["status"] for turn in curr_battle.battle_timeline]
        )
        all_statuses.update(
            [turn["p2_pokemon_state"]["status"] for turn in curr_battle.battle_timeline]
        )
    return all_statuses


def submission(test, predictions):
    return pd.DataFrame({"battle_id": test["battle_id"], "player_won": predictions})


class DataGatherer:
    def __init__(self, df):
        self.effects = effects(df)
        self.pokemon_types = pokemon_types(df)
        self.categories = categories(df)
        self.pokemon_names = pokemon_names(df)
        self.statuses = statuses(df)
