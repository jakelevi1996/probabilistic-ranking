import matplotlib.pyplot as plt
import numpy as np
from data.fileio import load

DEFAULT_FOLDER = "Code/results/"
DEFAULT_FILENAME = DEFAULT_FOLDER + "img"

def plot_gibbs_skills(
    skills_history, mean_skills, names, num_players=5,
    filename=DEFAULT_FILENAME, figsize=[16, 9], xlim=None
):
    plt.figure(figsize=figsize)
    for player in range(num_players):
        plt.plot(skills_history[player], alpha=0.7)
    plt.gca().set_prop_cycle(None)
    for player in range(num_players):
        plt.plot(
            [0, skills_history.shape[1]],
            [mean_skills[player], mean_skills[player]]
        )
    plt.grid(True)
    plt.legend(["{} ({})".format(names[i], i+1) for i in range(num_players)])
    plt.xlabel("Gibbs iteration")
    plt.ylabel("Skill")
    plt.title("Tennis player skills found using Gibbs ranking")
    if xlim is not None: plt.xlim([0, xlim])
    plt.savefig(filename)

def plot_gibbs_correlations(
    skills_history, mean_skills, names, num_players=5, maxlags=100,
    filename=DEFAULT_FILENAME, figsize=[16, 9]
):
    plt.figure(figsize=figsize)
    
    for player in range(num_players):
        trace = skills_history[player] - mean_skills[player]
        plt.xcorr(
            trace, trace, maxlags=maxlags, usevlines=False, linestyle="-",
            marker=None, alpha=0.7
        )
    plt.grid(True)
    plt.legend(["{} ({})".format(names[i], i+1) for i in range(num_players)])
    plt.xlabel("Lag")
    plt.ylabel("Auto-correlation")
    plt.title("Auto-correlations for skill traces found using Gibbs ranking")
    plt.savefig(filename)

if __name__ == "__main__":
    games, names = load()
    skills_history = np.load("Code/results/example_skills_history.npy")
    mean_skills = skills_history.mean(axis=1)
    plot_gibbs_correlations(skills_history, mean_skills, names, maxlags=15)
