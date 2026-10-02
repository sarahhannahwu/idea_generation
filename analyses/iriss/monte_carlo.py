from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / 'data' / 'evaluated_compliant_ideas.csv'
SUMMARY_PATH = REPO_ROOT / 'data' / 'monte_carlo_condition_results.csv'
DIFFERENCE_PATH = REPO_ROOT / 'data' / 'monte_carlo_condition_differences.csv'
MODEL_NAME = 'all-MiniLM-L6-v2'
N_REPETITIONS = 5000
RANDOM_SEED = 42
GROUP_COLUMNS = ['condition', 'object', 'submitter_id']


def sample_one_idea_per_participant(data, rng):
    sampled_indices = (
        data.groupby(GROUP_COLUMNS, sort=False, group_keys=False)
        .apply(lambda group: group.sample(n=1, random_state=rng), include_groups=False)
        .index
    )
    return data.loc[sampled_indices].reset_index(drop=True)


def sampled_similarity(data, embeddings, rng):
    sampled = sample_one_idea_per_participant(data, rng)
    sampled_embeddings = embeddings[sampled['row_index'].to_numpy()]
    similarities = np.full(len(sampled), np.nan)

    for (condition, obj), group_indices in sampled.groupby(
        ['condition', 'object'], sort=False
    ).groups.items():
        indices = np.asarray(list(group_indices))
        if len(indices) < 2:
            continue

        group_embeddings = sampled_embeddings[indices]
        group_sum = group_embeddings.sum(axis=0)
        for position in indices:
            other_centroid = (group_sum - sampled_embeddings[position]) / (len(indices) - 1)
            similarities[position] = 1 - cosine(
                sampled_embeddings[position], other_centroid
            )

    sampled['similarity'] = similarities
    return sampled


def summarize_repetitions(repetition_results):
    condition_summary = (
        repetition_results
        .groupby(['repetition', 'condition'], as_index=False)['similarity']
        .mean()
    )
    condition_means = (
        condition_summary
        .groupby('condition')['similarity']
        .agg(
            mean='mean',
            sd='std',
            ci_lower=lambda values: values.quantile(0.025),
            ci_upper=lambda values: values.quantile(0.975),
            n_repetitions='count'
        )
        .reset_index()
    )

    wide = condition_summary.pivot(
        index='repetition', columns='condition', values='similarity'
    )
    reference = 'Low-Agency'
    differences = []
    for condition in sorted(set(wide.columns) - {reference}):
        difference = wide[condition] - wide[reference]
        differences.append(pd.DataFrame({
            'condition': condition,
            'reference': reference,
            'difference': difference.to_numpy(),
            'repetition': difference.index.to_numpy()
        }))

    difference_results = pd.concat(differences, ignore_index=True)
    difference_summary = (
        difference_results
        .groupby(['condition', 'reference'])['difference']
        .agg(
            mean='mean',
            sd='std',
            ci_lower=lambda values: values.quantile(0.025),
            ci_upper=lambda values: values.quantile(0.975),
            proportion_positive=lambda values: np.mean(values > 0),
            n_repetitions='count'
        )
        .reset_index()
    )
    return condition_summary, condition_means, difference_results, difference_summary


def main():
    data = pd.read_csv(DATA_PATH)
    data['use'] = data['use'].fillna('').astype(str).str.strip()
    data = data[data['use'].ne('')].copy().reset_index(drop=True)
    data['row_index'] = np.arange(len(data))

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(data['use'].tolist(), show_progress_bar=True)
    rng = np.random.default_rng(RANDOM_SEED)

    repetition_results = []
    for repetition in range(N_REPETITIONS):
        sampled = sampled_similarity(data, embeddings, rng)
        sampled['repetition'] = repetition
        repetition_results.append(
            sampled[['repetition', 'condition', 'object', 'submitter_id', 'similarity']]
        )

    repetition_results = pd.concat(repetition_results, ignore_index=True)
    (
        condition_summary,
        condition_means,
        difference_results,
        difference_summary
    ) = summarize_repetitions(repetition_results)

    condition_summary.to_csv(SUMMARY_PATH, index=False)
    difference_results.to_csv(DIFFERENCE_PATH, index=False)

    print('Condition means across random one-idea samples:')
    print(condition_means.to_string(index=False))
    print('\nDifferences from Low-Agency:')
    print(difference_summary.to_string(index=False))


if __name__ == '__main__':
    main()
