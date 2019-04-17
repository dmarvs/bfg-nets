from typing import List

import matplotlib.pyplot as plt

from rsCNN.evaluation import samples, shared


plt.switch_backend('Agg')  # Needed for remote server plotting


def plot_raw_and_transformed_input_samples(
        sampled: samples.Samples,
        max_pages: int = 8,
        max_samples_per_page: int = 10,
        max_features_per_page: int = 5,
        max_responses_per_page: int = 5
) -> List[plt.Figure]:
    figures = []
    idx_current_sample = 0
    idx_current_feature = 0
    idx_current_response = 0
    while idx_current_sample < sampled.num_samples and len(figures) < max_pages:
        idx_last_sample = min(sampled.num_samples, idx_current_sample + max_samples_per_page)
        range_samples = range(idx_current_sample, idx_last_sample)
        while idx_current_feature < sampled.num_features:
            idx_last_feature = min(sampled.num_features, idx_current_feature + max_features_per_page)
            range_features = range(idx_current_feature, idx_last_feature)
            idx_last_response = min(sampled.num_responses, idx_current_response + max_responses_per_page)
            range_responses = range(idx_current_response, idx_last_response)
            fig = _plot_input_page(sampled, range_samples, range_features, range_responses)
            fig.suptitle('Input Example Plots (page {})'.format(len(figures)))
            figures.append(fig)
            idx_current_feature = min(sampled.num_features, idx_current_feature + max_features_per_page)
            idx_current_response = min(sampled.num_responses, idx_current_response + max_responses_per_page)
        idx_current_sample = min(sampled.num_samples, idx_current_sample + max_samples_per_page)
    return figures


def _plot_input_page(sampled: samples.Samples, range_samples: range, range_features: range, range_responses: range)\
        -> plt.Figure:
    nrows = 1 + len(range_samples)
    ncols = 1 + 2 * (len(range_features) + len(range_responses))
    fig, grid = shared.get_figure_and_grid(nrows, ncols)
    idx_col = 0
    for idx_sample in range_samples:
        for idx_feature in range_features:
            ax = plt.subplot(grid[idx_sample, idx_col])
            shared.plot_raw_features(sampled, idx_sample, idx_feature, ax, idx_sample == 0, idx_col == 0)
            idx_col += 1
        for idx_feature in range_features:
            ax = plt.subplot(grid[idx_sample, idx_col])
            shared.plot_transformed_features(sampled, idx_sample, idx_feature, ax, idx_sample == 0, idx_col == 0)
            idx_col += 1
        for idx_response in range_responses:
            ax = plt.subplot(grid[idx_sample, idx_col])
            shared.plot_raw_responses(sampled, idx_sample, idx_response, ax, idx_sample == 0, idx_col == 0)
            idx_col += 1
        for idx_response in range_responses:
            ax = plt.subplot(grid[idx_sample, idx_col])
            shared.plot_transformed_responses(sampled, idx_sample, idx_response, ax, idx_sample == 0, idx_col == 0)
            idx_col += 1
        ax = plt.subplot(grid[idx_sample, idx_col])
        shared.plot_weights(sampled, ax, idx_sample == 0)
    return fig