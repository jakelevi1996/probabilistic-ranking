import matplotlib.pyplot as plt
import numpy as np
from data.fileio import load

DEFAULT_FILENAME = "Code/results/img"

def plot_gibbs_skills(
    skills_history, mean_skills, names, players,
    filename=DEFAULT_FILENAME, figsize=[16, 9], xlim=None, fmt=""
):
    plt.figure(figsize=figsize)
    # Plot traces:
    for p in players:
        plt.plot(skills_history[p], fmt, alpha=0.7)
    # Reset colour cycle and plot means:
    plt.gca().set_prop_cycle(None)
    for p in players:
        plt.plot(
            [0, skills_history.shape[1]],
            [mean_skills[p], mean_skills[p]]
        )
    # Format and save:
    plt.grid(True)
    plt.legend(["{} ({})".format(names[p], p+1) for p in players])
    plt.xlabel("Gibbs iteration")
    plt.ylabel("Skill")
    plt.title("Tennis player skills found using Gibbs ranking")
    if xlim is not None: plt.xlim([0, xlim])
    plt.savefig(filename)
    plt.close()

def plot_gibbs_correlations(
    skills_history, mean_skills, names, players, maxlags=100,
    filename=DEFAULT_FILENAME, figsize=None
):
    if figsize is not None: plt.figure(figsize=figsize)
    else: plt.figure()
    
    for p in players:
        trace = skills_history[p] - mean_skills[p]
        plt.xcorr(
            trace, trace, maxlags=maxlags, usevlines=False, linestyle="-",
            marker=None, alpha=0.7
        )
    plt.grid(True)
    plt.legend(["{} ({})".format(names[p], p+1) for p in players])
    plt.xlabel("Lag")
    plt.ylabel("Auto-correlation")
    plt.title("Auto-correlations for skill traces found using Gibbs ranking")
    plt.savefig(filename)
    plt.close()

def plot_ep_skills(
    skill_means_history, skill_stds_history, names, players,
    filename=DEFAULT_FILENAME, figsize=[16, 9], plot_std=True
):
    plt.figure(figsize=figsize)
    for p in players:
        plt.plot(skill_means_history[p])
    plt.gca().set_prop_cycle(None)
    if plot_std:
        for p in players:
            plt.fill_between(
                range(skill_means_history.shape[1]),
                (skill_means_history[p] + 2.0*skill_stds_history[p]),
                (skill_means_history[p] - 2.0*skill_stds_history[p]),
                alpha=0.2
            )
    plt.grid(True)
    plt.legend(["{} ({})".format(names[p], p+1) for p in players])
    plt.xlabel("EP iteration")
    plt.ylabel("Skill")
    plt.title("Tennis player skills found using EP ranking")
    plt.savefig(filename)
    plt.close()

def plot_ep_skills_cyclic_colours(
    skill_means_history, skill_stds_history, names, players,
    filename=DEFAULT_FILENAME, figsize=[16, 9], alpha=0.3
):
    plt.figure(figsize=figsize)
    # colours = iter(plt.cm.get_cmap("viridis"))
    # colours = iter(plt.cm.bwr(np.linspace(0, 1, len(players))))
    for p in players:
        # plt.plot(skill_means_history[p], c=next(colours), alpha=alpha)
        plt.plot(skill_means_history[p], 'b', alpha=alpha)
    plt.grid(True)
    # plt.legend(["{} ({})".format(names[p], p+1) for p in players])
    plt.xlabel("EP iteration")
    plt.ylabel("Skill")
    plt.title("Tennis player skills found using EP ranking")
    plt.savefig(filename)
    plt.close()

def scale_array(array, upper=1.0, lower=0.0):
    return lower + (upper - lower) * (
        (array - min(array)) / (max(array) - min(array))
    )

def plot_rank_bars(
    empirical_skills, gibbs_skill_means, ep_skill_means, gibbs_sort,
    names, num_players, filename=DEFAULT_FILENAME, figsize=[16, 9], alpha=0.5
):
    plt.figure(figsize=figsize)
    for skill in [empirical_skills, gibbs_skill_means, ep_skill_means]:
        plt.barh(
            range(num_players), scale_array(skill[gibbs_sort]),
            height=1, tick_label=names[gibbs_sort], alpha=alpha
        )
    plt.gca().xaxis.grid(True)
    plt.xlabel("Normalised skill")
    plt.ylabel("Player name")
    plt.title("Normalised skills for each player, ordered by Gibbs rank")
    plt.legend(["Empirical skills", "Gibbs skills", "EP skills"])
    plt.yticks(fontsize=5)
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_rank_lines(
    empirical_skills, gibbs_skill_means, ep_skill_means, gibbs_sort,
    names, num_players, filename=DEFAULT_FILENAME, figsize=[16, 9], alpha=0.8
):
    plt.figure(figsize=figsize)
    for skill in [empirical_skills, gibbs_skill_means, ep_skill_means]:
        plt.plot(
            range(num_players, 0, -1), scale_array(skill[gibbs_sort]),
            alpha=alpha
        )
    plt.grid(True)
    plt.xlabel("Player rank (Gibbs)")
    plt.ylabel("Normalised skill")
    plt.title("Normalised skills for each player as a function of Gibbs rank")
    plt.legend(["Empirical skills", "Gibbs skills", "EP skills"])
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    games, names = load()
    skills_history = np.load("Code/results/example_skills_history.npy")
    mean_skills = skills_history.mean(axis=1)
    fmt = ""
    xlim = None
    # fmt = "o-"
    # xlim = 200
    plot_gibbs_skills(
        skills_history, mean_skills, names, range(5), xlim=xlim, fmt=fmt
    )
    # plot_gibbs_correlations(skills_history, mean_skills, names, maxlags=15)
    print(scale_array(np.array([3, 6, 7])))
