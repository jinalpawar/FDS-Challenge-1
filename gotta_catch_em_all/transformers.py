import numpy as np
from sklearn.preprocessing import MinMaxScaler


def battle_id(curr_battle):

    return {"battle_id": curr_battle["battle_id"]}


def base_stats(curr_battle):
    """
    Aggregates p1 team's base stats into mean base stats. Collects p2 lead's base stats.
    """

    p1_team_stats = curr_battle["p1_team_details"]
    p2_lead_stats = curr_battle["p2_lead_details"]

    return {
        "p1_mean_hp": np.mean([pokemon.get("base_hp", 0) for pokemon in p1_team_stats]),
        "p1_mean_spe": np.mean(
            [pokemon.get("base_spe", 0) for pokemon in p1_team_stats]
        ),
        "p1_mean_atk": np.mean(
            [pokemon.get("base_atk", 0) for pokemon in p1_team_stats]
        ),
        "p1_mean_def": np.mean(
            [pokemon.get("base_def", 0) for pokemon in p1_team_stats]
        ),
        "p1_mean_spa": np.mean(
            [pokemon.get("base_spa", 0) for pokemon in p1_team_stats]
        ),
        "p1_mean_spd": np.mean(
            [pokemon.get("base_spd", 0) for pokemon in p1_team_stats]
        ),
        "p1_mean_level": np.mean(
            [pokemon.get("level", 0) for pokemon in p1_team_stats]
        ),
        "p2_lead_hp": p2_lead_stats["base_hp"],
        "p2_lead_spe": p2_lead_stats["base_spe"],
        "p2_lead_atk": p2_lead_stats["base_atk"],
        "p2_lead_def": p2_lead_stats["base_def"],
        "p2_lead_spa": p2_lead_stats["base_spa"],
        "p2_lead_spd": p2_lead_stats["base_spd"],
        "p2_lead_level": p2_lead_stats["level"],
    }


def mean_hp_pct(curr_battle):
    """
    Provides mean of minimum hp% per pokemon after 30 turns
    """

    battle_turn = curr_battle["battle_timeline"]

    p1_team = [pokemon["name"] for pokemon in curr_battle["p1_team_details"]]
    p2_team = set([pokemon["p2_pokemon_state"]["name"] for pokemon in battle_turn])

    p1_hp_pct = []
    p2_hp_pct = []

    for pokemon_name in p1_team:
        p1_hp_pct.append(
            (
                min(
                    [
                        turn["p1_pokemon_state"]["hp_pct"]
                        for turn in battle_turn
                        if turn["p1_pokemon_state"]["name"] == pokemon_name
                    ],
                    default=1.0,
                )
            )
        )

    for pokemon_name in p2_team:
        p2_hp_pct.append(
            (
                min(
                    [
                        turn["p2_pokemon_state"]["hp_pct"]
                        for turn in battle_turn
                        if turn["p2_pokemon_state"]["name"] == pokemon_name
                    ],
                    default=1.0,
                )
            )
        )

    return {"p1_mean_hp_pct": np.mean(p1_hp_pct), "p2_mean_hp_pct": np.mean(p2_hp_pct)}


def total_dmg_taken(curr_battle):
    """
    Provides total dmg taken per pokemon in p1
    """

    battle_turn = curr_battle["battle_timeline"]

    p1_team = [pokemon["name"] for pokemon in curr_battle["p1_team_details"]]
    dmg_taken = []

    for pokemon_name in p1_team:
        hp_pct_curr = min(
            [
                turn["p1_pokemon_state"]["hp_pct"]
                for turn in battle_turn
                if turn["p1_pokemon_state"]["name"] == pokemon_name
            ],
            default=1.0,
        )
        hp_init = [
            pokemon["base_hp"]
            for pokemon in curr_battle["p1_team_details"]
            if pokemon["name"] == pokemon_name
        ][0]
        dmg_taken.append(hp_init - hp_init * hp_pct_curr)

    return {"p1_total_dmg_taken": np.mean(dmg_taken)}


def pokemon_KOd(curr_battle):
    """
    Confirms if each member of P1 was KOd. Confirms if last member of P2 was KOd.
    """

    battle_turn = curr_battle["battle_timeline"]

    p1_team = [pokemon["name"] for pokemon in curr_battle["p1_team_details"]]
    p2_team = set([turn["p2_pokemon_state"]["name"] for turn in battle_turn])

    pokemon_KOd = {}

    for idx, pokemon_name in enumerate(p1_team):
        hp_pct_curr = min(
            [
                turn["p1_pokemon_state"]["hp_pct"]
                for turn in battle_turn
                if turn["p1_pokemon_state"]["name"] == pokemon_name
            ],
            default=1.0,
        )

        pokemon_KOd[f"p1_pokemon_{idx}_KOd"] = hp_pct_curr == 0.0

    for pokemon_name in p2_team:
        hp_pct_curr = min(
            [
                turn["p2_pokemon_state"]["hp_pct"]
                for turn in battle_turn
                if turn["p2_pokemon_state"]["name"] == pokemon_name
            ],
            default=1.0,
        )

        pokemon_KOd[f"p2_pokemon_last_KOd"] = hp_pct_curr == 0.0

    return pokemon_KOd


def pokemon_KOd_all(curr_battle):
    """
    Confirms if each member of P1 was KOd. Confirms if last member of P2 was KOd
    """

    battle_turn = curr_battle["battle_timeline"]

    p1_team = [pokemon["name"] for pokemon in curr_battle["p1_team_details"]]
    p2_team = set([turn["p2_pokemon_state"]["name"] for turn in battle_turn])

    pokemon_KOd = {}

    for idx, pokemon_name in enumerate(p1_team):
        hp_pct_curr = min(
            [
                turn["p1_pokemon_state"]["hp_pct"]
                for turn in battle_turn
                if turn["p1_pokemon_state"]["name"] == pokemon_name
            ],
            default=1.0,
        )

        pokemon_KOd[f"p1_pokemon_{idx}_KOd"] = hp_pct_curr == 0.0

    for idx, pokemon_name in enumerate(p2_team):
        hp_pct_curr = min(
            [
                turn["p2_pokemon_state"]["hp_pct"]
                for turn in battle_turn
                if turn["p2_pokemon_state"]["name"] == pokemon_name
            ],
            default=1.0,
        )

        pokemon_KOd[f"p2_pokemon_{idx}_KOd"] = hp_pct_curr == 0.0

    return pokemon_KOd


def mean_boosts(curr_battle):
    """
    Provides Mean of boosts per turn
    """

    battle_turn = curr_battle["battle_timeline"]

    return {
        "p1_mean_boosts": np.mean(
            [sum(turn["p1_pokemon_state"]["boosts"].values()) for turn in battle_turn]
        ),
        "p2_mean_boosts": np.mean(
            [sum(turn["p2_pokemon_state"]["boosts"].values()) for turn in battle_turn]
        ),
    }


def mean_boosts_OHE(curr_battle):
    """
    One Hot Encoding for Mean of Boosts
    """

    battle_turn = curr_battle["battle_timeline"]

    return {
        "p1_mean_boost_atk_OHE": np.mean(
            [turn["p1_pokemon_state"]["boosts"]["atk"] for turn in battle_turn]
        )
        > 0,
        "p1_mean_boost_def_OHE": np.mean(
            [turn["p1_pokemon_state"]["boosts"]["def"] for turn in battle_turn]
        )
        > 0,
        "p1_mean_boost_spa_OHE": np.mean(
            [turn["p1_pokemon_state"]["boosts"]["spa"] for turn in battle_turn]
        )
        > 0,
        "p1_mean_boost_spd_OHE": np.mean(
            [turn["p1_pokemon_state"]["boosts"]["spd"] for turn in battle_turn]
        )
        > 0,
        "p1_mean_boost_spe_OHE": np.mean(
            [turn["p1_pokemon_state"]["boosts"]["spe"] for turn in battle_turn]
        )
        > 0,
        "p2_mean_boost_atk_OHE": np.mean(
            [turn["p2_pokemon_state"]["boosts"]["atk"] for turn in battle_turn]
        )
        > 0,
        "p2_mean_boost_def_OHE": np.mean(
            [turn["p2_pokemon_state"]["boosts"]["def"] for turn in battle_turn]
        )
        > 0,
        "p2_mean_boost_spa_OHE": np.mean(
            [turn["p2_pokemon_state"]["boosts"]["spa"] for turn in battle_turn]
        )
        > 0,
        "p2_mean_boost_spd_OHE": np.mean(
            [turn["p2_pokemon_state"]["boosts"]["spd"] for turn in battle_turn]
        )
        > 0,
        "p2_mean_boost_spe_OHE": np.mean(
            [turn["p2_pokemon_state"]["boosts"]["spe"] for turn in battle_turn]
        )
        > 0,
    }


def mean_accuracy_OHE(curr_battle):
    """
    Provides Mean of boosts per turn
    """

    battle_turn = curr_battle["battle_timeline"]

    return {
        "p1_mean_accuracy_OHE": np.mean(
            [
                turn["p1_move_details"]["accuracy"]
                for turn in battle_turn
                if turn["p1_move_details"]
            ]
        )
        > 0.5,
        "p2_mean_accuracy_OHE": np.mean(
            [
                turn["p2_move_details"]["accuracy"]
                for turn in battle_turn
                if turn["p2_move_details"]
            ]
        )
        > 0.5,
    }


def effects_used(curr_battle, data):
    """
    One Hot Encoding for effects
    """

    battle_turn = curr_battle["battle_timeline"]

    effects_used_per_team = {}

    p1_effects = set([turn["p1_pokemon_state"]["effects"][0] for turn in battle_turn])
    p2_effects = set([turn["p2_pokemon_state"]["effects"][0] for turn in battle_turn])

    for effect in data.effects:
        effects_used_per_team[f"p1_{effect}"] = effect in p1_effects
        effects_used_per_team[f"p2_{effect}"] = effect in p2_effects

    return effects_used_per_team


def types_used(curr_battle, data):
    """
    One Hot Encoding for move types
    """

    battle_turn = curr_battle["battle_timeline"]

    types_used_per_team = {}

    p1_move_type = set(
        [
            turn["p1_move_details"]["type"]
            for turn in battle_turn
            if turn["p1_move_details"]
        ]
    )
    p2_move_type = set(
        [
            turn["p2_move_details"]["type"]
            for turn in battle_turn
            if turn["p2_move_details"]
        ]
    )

    for move in data.pokemon_types:
        types_used_per_team[f"p1_{move}"] = move in p1_move_type
        types_used_per_team[f"p2_{move}"] = move in p2_move_type

    return types_used_per_team


def pokemons_used(curr_battle, data):
    """
    One Hot Encoding for Pokemons
    """

    battle_turn = curr_battle["battle_timeline"]
    pokemons_used_per_team = {}

    p1_pokemon = set(
        [
            pokemon["p1_pokemon_state"]["name"]
            for pokemon in battle_turn
            if pokemon["p1_pokemon_state"]
        ]
    )
    p2_pokemon = set(
        [
            pokemon["p2_pokemon_state"]["name"]
            for pokemon in battle_turn
            if pokemon["p2_pokemon_state"]
        ]
    )

    for pokemon in data.pokemon_names:
        pokemons_used_per_team[f"p1_{pokemon}"] = pokemon in p1_pokemon
        pokemons_used_per_team[f"p2_{pokemon}"] = pokemon in p2_pokemon

    return pokemons_used_per_team


def categories_used(curr_battle, data):
    """
    One Hot Encoding for categories
    """

    battle_turn = curr_battle["battle_timeline"]
    categories_used_per_team = {}

    p1_categories = set(
        [
            turn["p1_move_details"]["category"]
            for turn in battle_turn
            if turn["p1_move_details"]
        ]
    )
    p2_categories = set(
        [
            turn["p2_move_details"]["category"]
            for turn in battle_turn
            if turn["p2_move_details"]
        ]
    )

    # One hot encoding for categories
    for category in data.categories:
        categories_used_per_team[f"p1_category_{category}"] = category in p1_categories
        categories_used_per_team[f"p2_category_{category}"] = category in p2_categories

    return categories_used_per_team


def statuses_used(curr_battle, data):
    """
    One Hot Encoding for Status
    """

    battle_turn = curr_battle["battle_timeline"]
    statuses_used_per_team = {}

    p1_status = set([turn["p1_pokemon_state"]["status"] for turn in battle_turn])
    p2_status = set([turn["p2_pokemon_state"]["status"] for turn in battle_turn])

    for status in data.statuses:
        statuses_used_per_team[f"p1_status_{status}"] = status in p1_status
        statuses_used_per_team[f"p2_status_{status}"] = status in p2_status

    return statuses_used_per_team


def mean_move_base_power(curr_battle):
    """
    Calculates mean of base power per move in turn
    """

    battle_turn = curr_battle["battle_timeline"]

    return {
        "p1_mean_move_base_power": np.mean(
            [
                turn["p1_move_details"]["base_power"]
                for turn in battle_turn
                if turn["p1_move_details"]
            ]
        ),
        "p2_mean_move_base_power": np.mean(
            [
                turn["p2_move_details"]["base_power"]
                for turn in battle_turn
                if turn["p2_move_details"]
            ]
        ),
    }


def no_of_moves(curr_battle):
    """
    Calculates number of moves carried out in 30 turns
    """

    battle_turn = curr_battle["battle_timeline"]

    return {
        "p1_no_of_moves": sum([1 for move in battle_turn if move["p1_move_details"]]),
        "p2_no_of_moves": sum([1 for move in battle_turn if move["p2_move_details"]]),
    }


def player_won(curr_battle):

    if "player_won" in curr_battle:
        return {"player_won": curr_battle["player_won"]}
    return {}


def scale(df):
    """
    Scale numerical columns
    """

    scaler = MinMaxScaler()

    columns_to_scale = [
        "p1_mean_hp",
        "p1_mean_spe",
        "p1_mean_atk",
        "p1_mean_def",
        "p1_mean_spa",
        "p1_mean_spd",
        "p1_mean_level",
        "p2_lead_hp",
        "p2_lead_spe",
        "p2_lead_atk",
        "p2_lead_def",
        "p2_lead_spa",
        "p2_lead_spd",
        "p2_lead_level",
        "p1_mean_hp_pct",
        "p2_mean_hp_pct",
        "p2_mean_move_base_power",
        "p1_mean_move_base_power",
        "p1_mean_boosts",
        "p2_mean_boosts",
        "p1_no_of_moves",
        "p2_no_of_moves",
        "p1_total_dmg_taken",
    ]

    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


def scale_wo_dmg(df):
    """
    Scale numerical columns
    """

    scaler = MinMaxScaler()
    columns_to_scale = [
        "p1_mean_hp",
        "p1_mean_spe",
        "p1_mean_atk",
        "p1_mean_def",
        "p1_mean_spa",
        "p1_mean_spd",
        "p1_mean_level",
        "p2_lead_hp",
        "p2_lead_spe",
        "p2_lead_atk",
        "p2_lead_def",
        "p2_lead_spa",
        "p2_lead_spd",
        "p2_lead_level",
        "p1_mean_hp_pct",
        "p2_mean_hp_pct",
        "p2_mean_move_base_power",
        "p1_mean_move_base_power",
        "p1_mean_boosts",
        "p2_mean_boosts",
        "p1_no_of_moves",
        "p2_no_of_moves",
    ]

    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
