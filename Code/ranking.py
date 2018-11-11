import numpy as np
from scipy.linalg import cholesky, solve_triangular as solve
from time import time

def gibbs_rank(games, num_players, num_steps=1100, print_every=None):
    if print_every is None: print_every = int(num_steps / 5)
    assert games.max() < num_players
    num_games = games.shape[0]
    # Initialise skills to prior mean (IE zero, by symmetry)
    skills = np.zeros([num_players, 1])
    skills_history = np.zeros([num_players, num_steps])
    # Initialise prior precision of skills
    prior_skills_precision = 2 * np.identity(num_players)
    for step in range(num_steps):
        if step % print_every == 0: print("Step = {}".format(step))
        # Sample performance differences per game, given skills and outcomes:
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
        # Update diagonal terms of posterior precision of skills
        players, matches_per_player = np.unique(games, return_counts=True)
        post_skills_precision[players, players] += matches_per_player
        # Update off-diagonal terms of posterior precision of skills
        pairs, matches_per_pair = np.unique(games, return_counts=True, axis=0)
        post_skills_precision[pairs[:, 0], pairs[:, 1]] -= matches_per_pair
        post_skills_precision[pairs[:, 1], pairs[:, 0]] -= matches_per_pair
        # Calculate the product of the posterior skills precision and mean
        post_skills_precision_mean = (np.where(
            games[:, 0].reshape(-1, 1) == np.arange(num_players), d_perf, 0
        ) - np.where(
            games[:, 1].reshape(-1, 1) == np.arange(num_players), d_perf, 0
        )).sum(axis=0).reshape(-1, 1)
        # Perform cholesky decomposition on the posterior precision of skills
        L = cholesky(post_skills_precision, lower=False, check_finite=False)
        # Sample skills per player, given performance differences:
        skills = solve(
            L, solve(
                L.T, post_skills_precision_mean,
                lower=True, check_finite=False
            ) + np.random.normal(size=[num_players, 1]),
            lower=False, check_finite=False
        )
        # Store the results:
        skills_history[:, step] = skills.reshape(-1)

    mean_skills = skills_history.mean(axis=1)
    std_skills = skills_history.std(axis=1)
    return skills_history, mean_skills, std_skills

if __name__ == "__main__":
    # np.random.seed(27)
    
    # games = np.array([
    #     [0, 1],
    #     [0, 2],
    #     [2, 0],
    #     [1, 2],
    #     [1, 2],
    #     [4, 3],
    #     [4, 3],
    #     [1, 4],
    #     [1, 3]
    # ])
    
    games = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [2, 3],
        [2, 3],
        [2, 3],
        [2, 3],
        [2, 3],
        [3, 4],
        [4, 5],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 4],
    ])
    num_players = games.max() +1

    num_steps = 2000
    print("Ranking {} games between {} players".format(
        games.shape[0], num_players
    ))
    start_time = time()
    skills_history, mean_skills, std_skills = gibbs_rank(
        games, num_players, num_steps
    )
    print("Time taken for {} steps = {:.3f} s".format(
        num_steps, time() - start_time
    ))
    print(np.concatenate(
        [mean_skills.reshape(-1, 1), std_skills.reshape(-1, 1)], axis=1)
    )
