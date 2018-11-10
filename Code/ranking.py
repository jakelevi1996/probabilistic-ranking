import numpy as np
from scipy.linalg import solve
from data.fileio import load

def gibbs_rank(games, num_players, num_steps=1100):
    num_games = games.shape[0]
    # Initialise skills to prior mean (IE zero, by symmetry)
    skills = np.zeros([num_players, 1])
    # Initialise prior variance over skills
    prior_skills_variance = 0.5 * np.identity(num_players)
    for _ in range(num_steps):
        # Sample performance differences, given skills and outcomes:
        d_skills = skills[games[:, 0]] - skills[games[:, 1]]
        d_perf = d_skills + np.random.normal(size=[num_games, 1])
        reject_inds = d_perf < 0
        # Reject any performance differences which disagree with game outcomes:
        while reject_inds.any():
            d_perf[reject_inds] = d_skills[reject_inds] + np.random.normal(
                size=reject_inds.sum()
            )
            reject_inds = d_perf < 0
        
        # Initialise posterior precision of skills to prior
        post_skills_precision = 1 / prior_skills_variance
        # Update diagonal terms of posterior
        uniques, counts = np.unique(games, return_counts=True)
        post_skills_precision[uniques, uniques] += counts
        # Update off-diagonal terms of posterior
        uniques, counts = np.unique(games, return_counts=True, axis=0)
        post_skills_precision[uniques[:, 0], uniques[:, 1]] -= counts
        post_skills_precision[uniques[:, 1], uniques[:, 0]] -= counts

    
    # return n

if __name__ == "__main__":
    np.random.seed(27)
    games, names = load()
    print(games.shape)
    num_players = names.size
    # print(gibbs_rank(games, num_players, 1))
    print(num_players)