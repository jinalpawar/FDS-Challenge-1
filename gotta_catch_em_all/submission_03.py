from sklearn.linear_model import LogisticRegression
from .helper import get_train_test, submission, DataGatherer
from . import transformers


def transform_row(row, data):
    return {
        **transformers.battle_id(row),
        **transformers.base_stats(row),
        **transformers.pokemon_KOd(row),
        **transformers.mean_hp_pct(row),
        **transformers.total_dmg_taken(row),
        **transformers.mean_boosts(row),
        **transformers.mean_boosts_OHE(row),
        **transformers.mean_accuracy_OHE(row),
        **transformers.types_used(row, data),
        **transformers.pokemons_used(row, data),
        **transformers.categories_used(row, data),
        **transformers.statuses_used(row, data),
        **transformers.mean_move_base_power(row),
        **transformers.no_of_moves(row),
        **transformers.player_won(row),
    }


def transform(df):
    data = DataGatherer(df)
    transformed_df = df.apply(
        lambda row: transform_row(row, data), axis=1, result_type="expand"
    )
    transformers.scale(transformed_df)

    return transformed_df.fillna(0)


def main(output_path, input_path="."):
    train, test = get_train_test(input_path)

    train_df = transform(train)
    test_df = transform(test)

    X_train = train_df.drop(columns=["battle_id", "player_won"])
    Y_train = train_df["player_won"]
    X_test = test_df[X_train.columns]

    model_lr = LogisticRegression(random_state=42, max_iter=10000)
    model_lr.fit(X_train, Y_train)
    predictions_lr = model_lr.predict(X_test)

    submission(test_df, predictions_lr).to_csv(output_path, index=False)


if __name__ == "__main__":
    main("submission_03.csv")
