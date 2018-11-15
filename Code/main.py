import numpy  as np
from time import time

import plotting, ranking
from data.fileio import load

def question_a(games, names, num_players, num_steps=2000):
    skills_history, mean_skills, _ = ranking.gibbs_rank(
        games, num_players, num_steps
    )
    plotting.plot_gibbs_skills(skills_history, mean_skills, names)
    plotting.plot_gibbs_correlations(
        skills_history, mean_skills, names, maxlags=15
    )

def question_b(games, names, num_players, num_steps=81):
    ep_skill_means, ep_skill_stds = ranking.ep_rank(
        games, num_players, num_steps
    )
    plotting.plot_ep_skills(
        ep_skill_means, ep_skill_stds, names, 6, plot_std=True
    )
    return ep_skill_means, ep_skill_stds

def question_c(ep_skill_means, ep_skill_stds, names, players=range(4)):
    print("EP skills table:")
    print(names[players])
    print(ranking.ep_skill_table(
        ep_skill_means[:, -1], ep_skill_stds[:, -1], players
    ))
    print("EP performance table:")
    print(names[players])
    print(ranking.ep_performance_table(
        ep_skill_means[:, -1], ep_skill_stds[:, -1], players
    ))

def question_d():
    pass

def question_e():
    pass

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
    games, names = load()
    num_players = names.size
    # save_example_skills_history()
    # question_a(games, names, num_players, num_steps=2000)
    ep_skill_means, ep_skill_stds = question_b(
        games, names, num_players, num_steps=81
    )
    question_c(ep_skill_means, ep_skill_stds, names)
