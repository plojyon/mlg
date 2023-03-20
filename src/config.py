"""Project configuration."""

base = "spotify_million_playlist_dataset"
pickles = base + "/pickles"

# full dataset
class big:
  dataset_path = base + "/data"
  pickled_graph = pickles + "/G.pkl"
  pickled_datasets = pickles + "/datasets.pkl"
  pickled_ghetero = pickles + "/ghetero.pkl"
  pickled_top_G = lambda i: pickles + f"/top-G-{i}.pkl"

# example dataset (override above)
dataset_path = base + "/example"
pickled_graph = pickles + "/G_example.pkl"
pickled_datasets = pickles + "/datasets_example.pkl"
pickled_ghetero = pickles + "/ghetero_example.pkl"
pickled_top_G = lambda i: pickles + f"/top-G-{i}_example.pkl"