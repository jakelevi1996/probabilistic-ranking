import numpy as np
from scipy.linalg import solve
from data.fileio import load

def gibbs_rank(games, num_players, num_steps=1100):
    assert games.max() < num_players
    num_games = games.shape[0]
    # Initialise skills to prior mean (IE zero, by symmetry)
    skills = np.zeros([num_players, 1])
    # Initialise prior precision of skills
    prior_skills_precision = 2 * np.identity(num_players)
    for _ in range(num_steps):
        # Sample per-game performance differences, given skills and outcomes:
        d_skills = skills[games[:, 0]] - skills[games[:, 1]]
        assert d_skills.size == num_games
        d_perf = d_skills + np.random.normal(size=[num_games, 1])
        reject_inds = d_perf < 0
        # Reject any performance differences which disagree with game outcomes:
        while reject_inds.any():
            d_perf[reject_inds] = d_skills[reject_inds] + np.random.normal(
                size=reject_inds.sum()
            )
            reject_inds = d_perf < 0
        # Initialise posterior precision of skills to prior
        post_skills_precision = 1.0 * prior_skills_precision
        # Update diagonal terms of posterior
        players, matches_per_player = np.unique(games, return_counts=True)
        post_skills_precision[players, players] += matches_per_player
        # Update off-diagonal terms of posterior
        pairs, matches_per_pair = np.unique(games, return_counts=True, axis=0)
        post_skills_precision[pairs[:, 0], pairs[:, 1]] -= matches_per_pair
        post_skills_precision[pairs[:, 1], pairs[:, 0]] -= matches_per_pair
        # Calculate the product of the posterior skills precision and mean
        post_skills_precision_mean = (np.where(
            games[:, 0].reshape(-1, 1) == np.arange(num_players), d_perf, 0
        ) - np.where(
            games[:, 1].reshape(-1, 1) == np.arange(num_players), d_perf, 0
        ).sum(axis=0)).reshape(-1, 1)
        # d_perf[games[:, 0]] - d_perf[games[:, 1]]
    # return n

if __name__ == "__main__":
    np.random.seed(27)
    games, names = load()
    num_players = names.size
    print(games.shape)
    print(num_players)
    print(gibbs_rank(games, num_players, 1))