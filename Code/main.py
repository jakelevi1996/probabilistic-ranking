import numpy  as np
from time import time

import plotting, ranking
from data.fileio import load

def question_a(games, names, num_players, num_steps):
    skills_history, mean_skills, _ = ranking.gibbs_rank(
        games, num_players, num_steps
    )
    plotting.plot_gibbs_skills(skills_history, mean_skills, names)
    plotting.plot_gibbs_correlations(
        skills_history, mean_skills, names, maxlags=15
    )

def question_b(games, names, num_players, num_steps):
    skill_means_history, skill_stds_history = ranking.ep_rank(
        games, num_players, num_steps
    )
    plotting.plot_ep_skills(
        skill_means_history, skill_stds_history, names, 6, plot_std=True
    )

def question_c():
    pass

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
    question_a(games, names, num_players, num_steps=2000)
    # question_b(games, names, num_players, num_steps=81)
