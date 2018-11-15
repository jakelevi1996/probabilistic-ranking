import numpy as np
from scipy.linalg import cholesky, solve_triangular as solve
from scipy.stats import norm
from time import time

def gibbs_rank(games, num_players, num_steps=1100, print_every=None):
    if print_every is None: print_every = min(100, int(num_steps / 5))
    assert games.max() < num_players
    num_games = games.shape[0]
    # Initialise skills to prior mean (IE zero, by symmetry)
    skills = np.zeros([num_players, 1])
    skills_history = np.zeros([num_players, num_steps])
    # Initialise prior variance of skills
    prior_skills_var = 0.5
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
        post_skills_precision = np.identity(num_players) / prior_skills_var
        # Update diagonal terms of posterior precision of skills
        players, matches_per_player = np.unique(games, return_counts=True)
        post_skills_precision[players, players] += matches_per_player
        # Update off-diagonal terms of posterior precision of skills
        pairs, matches_per_pair = np.unique(games, return_counts=True, axis=0)
        post_skills_precision[pairs[:, 0], pairs[:, 1]] -= matches_per_pair
        post_skills_precision[pairs[:, 1], pairs[:, 0]] -= matches_per_pair
        # Calculate the posterior skills 'natural mean' (precision * mean)
        post_skills_natural_mean = (np.where(
            games[:, 0].reshape(-1, 1) == np.arange(num_players), d_perf, 0
        ) - np.where(
            games[:, 1].reshape(-1, 1) == np.arange(num_players), d_perf, 0
        )).sum(axis=0).reshape(-1, 1)
        # Perform cholesky decomposition on the posterior precision of skills
        L = cholesky(post_skills_precision, lower=False, check_finite=False)
        # Sample skills per player, given performance differences:
        skills = solve(
            L, solve(
                L.T, post_skills_natural_mean,
                lower=True, check_finite=False
            ) + np.random.normal(size=[num_players, 1]),
            lower=False, check_finite=False
        )
        # Store the results:
        skills_history[:, step] = skills.reshape(-1)

    mean_skills = skills_history.mean(axis=1)
    std_skills = skills_history.std(axis=1)
    return skills_history, mean_skills, std_skills

def compute_marginal_skills(
    prior_skills_variance, games, num_players,
    game_to_skill_means, game_to_skill_precs
):
    """This code function gets called in two different places, so best to put
    it in a function
    """
    skill_precs = 1.0 / prior_skills_variance + (np.where(
        games.reshape(-1, 1) == np.arange(num_players),
        game_to_skill_precs.reshape(-1, 1), 0
    )).sum(axis=0).reshape(num_players, 1)
    skill_means = 1.0 / skill_precs * (np.where(
        games.reshape(-1, 1) == np.arange(num_players),
        (game_to_skill_precs * game_to_skill_means).reshape(-1, 1), 0
    )).sum(axis=0).reshape(num_players, 1)
    return skill_means, skill_precs

def psi_func(x):
    return norm.pdf(x) / norm.cdf(x)

def lambda_func(x):
    return psi_func(x) * (psi_func(x) + x)

def ep_rank(games, num_players, num_steps=20, print_every=10):
    assert games.max() < num_players
    num_games = games.shape[0]
    # Initialise skills to prior mean (IE zero, by symmetry)
    skill_means_history = np.zeros([num_players, num_steps])
    skill_stds_history = np.zeros([num_players, num_steps])
    # Initialise prior variance of skills
    prior_skills_variance = 0.5
    # Step 0: nitialise game to skill messages to uninformative prior:
    game_to_skill_means = np.zeros(games.shape)
    game_to_skill_precs = np.zeros(games.shape)
    # Step 1: compute initial marginal skills per player:
    skill_means, skill_precs = compute_marginal_skills(
        prior_skills_variance, games, num_players,
        game_to_skill_means, game_to_skill_precs
    )
    for step in range(num_steps):
        if step % print_every == 0: print("Step = {}".format(step))
        # Step 2: compute skill to game messages per game:
        skill_to_game_precs = (
            skill_precs[games].reshape(games.shape)
        ) - game_to_skill_precs
        skill_to_game_means = ((
            (skill_precs * skill_means)[games].reshape(games.shape)
        ) - (game_to_skill_precs * game_to_skill_means)) / skill_to_game_precs
        # Step 3: compute game to performance messages per game:
        game_to_perf_vars = (
            1.0 + (1.0 / skill_to_game_precs).sum(axis=1)
        ).reshape(num_games, 1)
        game_to_perf_stds = np.sqrt(game_to_perf_vars)
        game_to_perf_means = (
            skill_to_game_means[:, 0] - skill_to_game_means[:, 1]
        ).reshape(num_games, 1)
        # Step 4: approximate the marginal performance differences per game:
        perf_means = game_to_perf_means + game_to_perf_stds * psi_func(
            game_to_perf_means / game_to_perf_stds
        )
        perf_precs = 1.0 / (game_to_perf_vars * (1.0 - lambda_func(
            game_to_perf_means / game_to_perf_stds
        )))
        # Step 5: compute performance to game messages per game:
        perf_to_game_precs = perf_precs - 1.0 / game_to_perf_vars
        perf_to_game_means = (
            perf_means * perf_precs - game_to_perf_means / game_to_perf_vars
        ) / perf_to_game_precs
        # Step 6: compute game to skill messages per game
        game_to_skill_precs = 1.0 / (
            1.0 + np.tile(
                1.0 / perf_to_game_precs, 2
            ) + 1.0 / skill_to_game_precs[:, [1, 0]]
        )
        game_to_skill_means = np.append(
            perf_to_game_means, -perf_to_game_means, axis=1
        ) + skill_to_game_means[:, [1, 0]]
        # Step 1: compute marginal skills per player:
        skill_means, skill_precs = compute_marginal_skills(
            prior_skills_variance, games, num_players,
            game_to_skill_means, game_to_skill_precs
        )
        # Store the results:
        skill_means_history[:, step] = skill_means.reshape(-1)
        skill_stds_history[:, step] = 1.0 / np.sqrt(skill_precs).reshape(-1)
    
    return skill_means_history, skill_stds_history

def relative_gaussian_probability(mu_1, var_1, mu_2, var_2, noise_var=0):
    return norm.cdf((mu_1 - mu_2) / (var_1 + var_2 + noise_var))

def relative_empirical_probability():
    pass

def ep_skill_table(final_mean_skills, final_std_skills, players=range(4)):
    """Return table of probabilities of having a higher skill, according to EP
    predictions.

    Rows of the table refer to Player 1
    Column refer to Player 2
    Value in the table refers to probability that Player 1 has a higher skill
    than Player 2 (returned as an upper-triangular matrix with no diagonal)
    """
    table = np.zeros([len(players) - 1, len(players)])
    for i in range(table.shape[0]):
        for j in range(i + 1, table.shape[1]):
            table[i, j] = relative_gaussian_probability(
                final_mean_skills[players[i]], final_std_skills[players[i]],
                final_mean_skills[players[j]], final_std_skills[players[j]],
            )
    return table

def ep_performance_table(
    final_mean_skills, final_std_skills, players=range(4)
):
    """Return table of probabilities of having a higher skill, according to EP
    predictions.

    Rows of the table refer to Player 1
    Column refer to Player 2
    Value in the table refers to probability that Player 1 has a higher skill
    than Player 2 (returned as an upper-triangular matrix with no diagonal)

    TODO: This code is really similar to the `ep_skill_table` function; should
    probably have some better form of code reuse
    """
    table = np.zeros([len(players) - 1, len(players)])
    for i in range(table.shape[0]):
        for j in range(i + 1, table.shape[1]):
            table[i, j] = relative_gaussian_probability(
                final_mean_skills[players[i]], final_std_skills[players[i]],
                final_mean_skills[players[j]], final_std_skills[players[j]],
                noise_var=1.0
            )
    return table

def gibbs_marginal_skill_table():
    pass

def gibbs_joint_skill_table():
    pass

if __name__ == "__main__":
    # np.random.seed(27)
    
    games = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 4],
        # [2, 3],
        # [2, 3],
        # [2, 3],
        # [2, 3],
        # [2, 3],
        # [0, 17]
    ])
    num_players = games.max() +1

    num_steps = 200
    print("Ranking {} games between {} players".format(
        games.shape[0], num_players
    ))
    start_time = time()
    # Do Gibbs sampling
    skills_history, mean_skills, std_skills = gibbs_rank(
        games, num_players, num_steps
    )
    print("Time taken for {} steps = {:.3f} s".format(
        num_steps, time() - start_time
    ))
    print("Gibbs results:\n", np.concatenate(
        [mean_skills.reshape(-1, 1), std_skills.reshape(-1, 1)], axis=1
    ))
    # Do EP inference
    ep_skill_means, ep_skill_stds = ep_rank(
        games, num_players, num_steps=6
    )
    print("EP results:\n", np.concatenate([
        ep_skill_means[:, -1].reshape(-1, 1),
        ep_skill_stds[:, -1].reshape(-1, 1)
    ], axis=1))
    # print("EP mean history:\n", skill_means_history)
    # print(skill_means_history[:, 1:] - skill_means_history[:, :-1])
    print("EP skills table:\n", ep_skill_table(
        ep_skill_means[:, -1], ep_skill_stds[:, -1]
    ))
    print("EP performance table:\n", ep_performance_table(
        ep_skill_means[:, -1], ep_skill_stds[:, -1]
    ))
    


    
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
