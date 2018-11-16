import numpy  as np
from time import time

import plotting, ranking
from data.fileio import load

def question_a(games, names, num_players, players, num_steps=2000):
    skill_history, skill_means, skill_stds = ranking.gibbs_rank(
        games, num_players, num_steps
    )
    plotting.plot_gibbs_skills(skill_history, skill_means, names, players)
    plotting.plot_gibbs_correlations(
        skill_history, skill_means, names, players, maxlags=100
    )
    return skill_history, skill_means, skill_stds

def question_b(games, names, num_players, players, num_steps=81):
    ep_skill_means, ep_skill_stds = ranking.ep_rank(
        games, num_players, num_steps
    )
    plotting.plot_ep_skills(
        ep_skill_means, ep_skill_stds, names, players, plot_std=True
    )
    plotting.plot_ep_skills_cyclic_colours(
        ep_skill_means, ep_skill_stds, names, range(num_players),
    )
    return ep_skill_means, ep_skill_stds

def question_c(ep_skill_means, ep_skill_stds, names, players):
    print("EP skills table:")
    print(names[players])
    print(ranking.marginal_skill_table(
        ep_skill_means, ep_skill_stds, players
    ))
    print("EP performance table:")
    print(names[players])
    print(ranking.marginal_performance_table(
        ep_skill_means, ep_skill_stds, players
    ))

def question_d(
    gibbs_skill_samples, gibbs_skill_means, gibbs_skill_stds, names, players
):
    print("Gibbs marginal skills table:")
    print(names[players])
    print(ranking.marginal_skill_table(
        gibbs_skill_means, gibbs_skill_stds, players
    ))
    print("Gibbs joint skills table:")
    print(names[players])
    print(ranking.joint_skill_table(gibbs_skill_samples, players))

def question_e(
    games, names, num_players, gibbs_skill_means, ep_skill_means
):
    empirical_skills, _ = ranking.empirical_rank(
        games, num_players
    )
    gibbs_sort = ranking.skill_rank(gibbs_skill_means)
    # plotting.plot_rank_bars(
    #     empirical_skills, gibbs_skill_means, ep_skill_means,
    #     gibbs_sort, names, num_players
    # )
    plotting.plot_rank_lines(
        empirical_skills, gibbs_skill_means, ep_skill_means,
        gibbs_sort, names, num_players
    )

def save_example_skills_history(
    num_steps=2000, random_seed=0,
    filename="Code/results/example_skills_history"
):
    np.random.seed(random_seed)
    games, names = load()
    num_players = names.size
    num_steps = 2000
    print_every = 100
    skills_history, _, _ = ranking.gibbs_rank(
        games, num_players, num_steps, print_every
    )
    np.save(filename, skills_history)

if __name__ == "__main__":
    np.random.seed(0)
    top_4_players = [15, 0, 4, 10]
    games, names = load()
    num_players = names.size
    # save_example_skills_history()
    # gibbs_skill_samples, gibbs_skill_means, gibbs_skill_stds = question_a(
    #     games, names, num_players, top_4_players, num_steps=2000
    # )
    ep_skill_means_history, ep_skill_stds_history = question_b(
        games, names, num_players, top_4_players, num_steps=100
    )
    ep_skill_means = ep_skill_means_history[:, -1]
    ep_skill_stds = ep_skill_stds_history[:, -1]
    question_c(
        ep_skill_means, ep_skill_stds,
        names, top_4_players
    )
    # question_d(
    #     gibbs_skill_samples, gibbs_skill_means, gibbs_skill_stds,
    #     names, [0, 15]
    # )
    # question_d(
    #     gibbs_skill_samples, gibbs_skill_means, gibbs_skill_stds,
    #     names, top_4_players
    # )
    # question_e(games, names, num_players, gibbs_skill_means, ep_skill_means)
