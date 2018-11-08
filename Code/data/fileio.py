import numpy as np

DEFAULT_FILENAME = "Code/data/tennis_data.npz"

def csvs_to_npz(
    names_filename="Code/data/names.csv",
    games_filename="Code/data/games.csv",
    output_filename=DEFAULT_FILENAME
):
    games = np.loadtxt(games_filename, delimiter=',', dtype=np.int16)
    names = np.loadtxt(names_filename, dtype=np.str)
    np.savez(DEFAULT_FILENAME, games=games, names=names)

def load(filename=DEFAULT_FILENAME):
    with np.load(filename) as data:
        games = data['games']
        names = data['names']
    return games, names

if __name__ == "__main__":
    csvs_to_npz()
    games, names = load()
    print(names)
    print(games)
