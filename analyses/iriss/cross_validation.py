import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / 'data' / 'evaluated_compliant_ideas.csv'
OUTPUT_PATH = REPO_ROOT / 'data' / 'participant_level_cross_validation.csv'
MODEL_NAME = 'all-MiniLM-L6-v2'
PREDICTOR_COLUMNS = ['condition', 'object', 'submitter_id']  # Columns to use as predictors in the regression model


def participant_embeddings(data, embeddings):
    rows = []
    for (condition, obj, submitter_id), group in data.groupby(
        ['condition', 'object', 'submitter_id'], sort=False
    ):
        indices = group.index.to_numpy()
        rows.append({
            'condition': condition,
            'object': obj,
            'submitter_id': submitter_id,
            'embedding': np.mean(embeddings[indices], axis=0)
        })
    return pd.DataFrame(rows)


def similarity_outcomes(targets, references):
    outcomes = []
    for target in targets.itertuples(index=False):
        mask = (
            (references['condition'] == target.condition) &
            (references['object'] == target.object) &
            (references['submitter_id'] != target.submitter_id)
        )
        reference_embeddings = references.loc[mask, 'embedding'].tolist()

        if not reference_embeddings:
            outcomes.append(np.nan)
            continue

        centroid = np.mean(reference_embeddings, axis=0)
        outcomes.append(1 - cosine(target.embedding, centroid))

    return np.asarray(outcomes)


def build_model():
    preprocessing = ColumnTransformer([
        (
            'categorical',
            OneHotEncoder(handle_unknown='ignore'),
            PREDICTOR_COLUMNS
        )
    ])
    return Pipeline([
        ('preprocessing', preprocessing),
        ('model', Ridge(alpha=1.0))
    ])


def main():
    data = pd.read_csv(DATA_PATH)
    data['use'] = data['use'].fillna('').astype(str).str.strip()
    data = data[data['use'].ne('')].copy()
    data = data.reset_index(drop=True)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(data['use'].tolist(), show_progress_bar=True)
    participants = participant_embeddings(data, embeddings)

    n_splits = min(5, participants['submitter_id'].nunique())
    cross_validator = GroupKFold(n_splits=n_splits)
    predictions = []

    for fold, (train_indices, test_indices) in enumerate(
        cross_validator.split(
            participants,
            groups=participants['submitter_id']
        ),
        start=1
    ):
        train = participants.iloc[train_indices].reset_index(drop=True)
        test = participants.iloc[test_indices].reset_index(drop=True)

        train_outcomes = similarity_outcomes(train, train)
        test_outcomes = similarity_outcomes(test, train)

        valid_train = np.isfinite(train_outcomes)
        valid_test = np.isfinite(test_outcomes)
        estimator = build_model()
        estimator.fit(train.loc[valid_train, PREDICTOR_COLUMNS], train_outcomes[valid_train])
        test_predictions = estimator.predict(test.loc[valid_test, PREDICTOR_COLUMNS])

        predictions.append(pd.DataFrame({
            'fold': fold,
            'condition': test.loc[valid_test, 'condition'].to_numpy(),
            'object': test.loc[valid_test, 'object'].to_numpy(),
            'submitter_id': test.loc[valid_test, 'submitter_id'].to_numpy(),
            'observed_similarity': test_outcomes[valid_test],
            'predicted_similarity': test_predictions
        }))

    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(OUTPUT_PATH, index=False)

    metrics = {
        'rmse': mean_squared_error(
            predictions['observed_similarity'],
            predictions['predicted_similarity']
        ) ** 0.5,
        'mae': mean_absolute_error(
            predictions['observed_similarity'],
            predictions['predicted_similarity']
        ),
        'r2': r2_score(
            predictions['observed_similarity'],
            predictions['predicted_similarity']
        ),
    }
    print(pd.Series(metrics))
    print(predictions.groupby('condition').apply(
        lambda group: pd.Series({
            'rmse': mean_squared_error(
                group['observed_similarity'],
                group['predicted_similarity']
            ) ** 0.5,
            'mae': mean_absolute_error(
                group['observed_similarity'],
                group['predicted_similarity']
            )
        }),
        include_groups=False
    ))


if __name__ == '__main__':
    main()