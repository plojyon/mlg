import json
import os
import pickle

import networkx as nx
import torch
import torch_geometric
import torch_geometric.transforms as T
from tqdm import tqdm

import config


def load_graph(config=config):
    """Load a nx.Graph from disk."""
    dataset_path = config.dataset_path
    filenames = os.listdir(dataset_path)
    G = nx.DiGraph()
    for i in tqdm(range(len(filenames)), unit="files"):
        with open(os.path.join(dataset_path, filenames[i])) as json_file:
            playlists = json.load(json_file)["playlists"]
            for playlist in playlists:
                playlist_name = f"spotify:playlist:{playlist['pid']}"
                G.add_node(playlist_name, node_type="playlist", num_followers=playlist["num_followers"], num_tracks=playlist["num_tracks"], num_artists=playlist["num_artists"], num_albums=playlist["num_albums"], duration_ms=playlist["duration_ms"], collaborative=playlist["collaborative"], num_edits=playlist["num_edits"])
                for track in playlist["tracks"]:
                    G.add_node(track["track_uri"], node_type="track", duration=track["duration_ms"])
                    G.add_node(track["album_uri"], node_type="album")
                    G.add_node(track["artist_uri"], node_type="artist")

                    G.add_edge(track["track_uri"], playlist_name, edge_type="track-playlist")
                    G.add_edge(track["track_uri"], track["album_uri"], edge_type="track-album")
                    G.add_edge(track["track_uri"], track["artist_uri"], edge_type="track-artist")
    return G

def nx2hetero(G, pickle_node_index=None):
    """Convert a nx.Graph into a torch_geometric.data.HeteroData object."""
    ids_by_type = {
        "playlist": {},
        "track": {},
        "artist": {},
        "album": {}
    }

    def node_id(node_type, id):
        d = ids_by_type[node_type]
        if id not in d:
            d[id] = len(d)
        return d[id]

    node_features_by_type = {
        "playlist": [],
        "track": [],
        "artist": [],
        "album": []
    }

    # {
    #     "name": "musical",
    #     "collaborative": "false",
    #     "pid": 5,
    #     "modified_at": 1493424000,
    #     "num_albums": 7,
    #     "num_tracks": 12,
    #     "num_followers": 1,
    #     "num_edits": 2,
    #     "duration_ms": 2657366,
    #     "num_artists": 6,
    #     "tracks": [
    #         {
    #             "pos": 0,
    #             "artist_name": "Degiheugi",
    #             "track_uri": "spotify:track:7vqa3sDmtEaVJ2gcvxtRID",
    #             "artist_uri": "spotify:artist:3V2paBXEoZIAhfZRJmo2jL",
    #             "track_name": "Finalement",
    #             "album_uri": "spotify:album:2KrRMJ9z7Xjoz1Az4O6UML",
    #             "duration_ms": 166264,
    #             "album_name": "Dancing Chords and Fireflies"
    #         },
    #     ],

    # })

    for node in G.nodes(data=True):
        t = node[1]["node_type"]
        node_id(t, node[0])
        if t == "playlist":
            if node[1]["collaborative"] not in ("true", "false"):
                raise ValueError(f"collaborative is not a boolean: {node[1]['collaborative']}")
            node_features_by_type["playlist"] += [[node[1]["num_followers"], node[1]["collaborative"] == 'true', node[1]["num_albums"], node[1]["num_tracks"], node[1]["num_edits"], node[1]["duration_ms"], node[1]["num_artists"]]]
        elif t == "track":
            distances = nx.single_source_shortest_path_length(G, node[0], cutoff=2)
            node_features_by_type["track"] += [[node[1]["duration"], len(distances)]]
        elif t == "artist":
            distances = nx.single_source_shortest_path_length(G, node[0], cutoff=2)
            node_features_by_type["artist"] += [[len(distances)]]
        elif t == "album":
            distances = nx.single_source_shortest_path_length(G, node[0], cutoff=2)
            node_features_by_type["album"] += [[len(distances)]]

    edge_index_by_type = {
        ("track", "contains", "playlist"): [],
        ("track", "includes", "album"): [],
        ("track", "authors", "artist"): []
    }
    existing_edges = set()
    for edge in G.edges(data=True):
        track_node = edge[0]
        other_node = edge[1]
        if "track" not in track_node:
            track_node, other_node = other_node, track_node

        if (track_node, other_node) in existing_edges:
            continue

        if G[edge[0]][edge[1]]["edge_type"] == "track-playlist":
            s_id = node_id("track", track_node)
            d_id = node_id("playlist", other_node)

            edge_index_by_type[("track", "contains", "playlist")] += [(s_id, d_id)]
            
        elif G[edge[0]][edge[1]]["edge_type"] == "track-album":
            s_id = node_id("track", track_node)
            d_id = node_id("album", other_node)

            edge_index_by_type[("track", "includes", "album")] += [(s_id, d_id)]

        elif G[edge[0]][edge[1]]["edge_type"] == "track-artist":
            s_id = node_id("track", track_node)
            d_id = node_id("artist", other_node)

            edge_index_by_type[("track", "authors", "artist")] += [(s_id, d_id)]

        existing_edges.add((track_node, other_node))

    # construct HeteroData
    hetero = torch_geometric.data.HeteroData()

    # add initial node features
    hetero["playlist"].x = torch.FloatTensor(node_features_by_type["playlist"]).reshape(-1,len(node_features_by_type["playlist"][0]))
    hetero["track"].x = torch.FloatTensor(node_features_by_type["track"]).reshape(-1,len(node_features_by_type["track"][0]))
    hetero["artist"].x = torch.FloatTensor(node_features_by_type["artist"]).reshape(-1,len(node_features_by_type["artist"][0]))
    hetero["album"].x = torch.FloatTensor(node_features_by_type["album"]).reshape(-1,len(node_features_by_type["album"][0]))

    # add edge indices
    hetero["track", "contains", "playlist"].edge_index = torch.tensor(edge_index_by_type[("track", "contains", "playlist")]).t()
    hetero["track", "includes", "album"].edge_index = torch.tensor(edge_index_by_type[("track", "includes", "album")]).t()
    hetero["track", "authors", "artist"].edge_index = torch.tensor(edge_index_by_type[("track", "authors", "artist")]).t()

    # post-processing
    hetero = torch_geometric.transforms.ToUndirected()(hetero)
    # hetero = torch_geometric.transforms.RemoveIsolatedNodes()(hetero)
    hetero = torch_geometric.transforms.NormalizeFeatures()(hetero)

    # save node index
    if pickle_node_index is not None:
        pickle.dump(ids_by_type, open(pickle_node_index, "wb"))
        print("Saved node index to", pickle_node_index)
    return hetero

def ghetero2datasets(ghetero):
    """Split the dataset into train, validation and test sets."""
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3, # TODO: move settings to config.py
            neg_sampling_ratio=2.0,
            add_negative_train_samples=False,
            edge_types=("track", "contains", "playlist"),
            rev_edge_types=("playlist", "rev_contains", "track"),
        )
    ])

    return transform(ghetero)  # 3-tuple: data_train, data_val, data_test

def get_cached(var, pickled_filename, fallback, ignore_cache=False):
    """Get a variable from cache.
    
    First, check global memory (variable `var`).
    If not found, check pickle (file `pickled_filename`).
    If not found, generate anew (use `fallback` function).
    """
    if not ignore_cache and var in globals():
        print(f"Using {var} from global memory ...")
        return globals()[var]
    elif not ignore_cache and os.path.exists(pickled_filename):
        print(f"Loading {var} from pickle ...")
        return pickle.load(open(pickled_filename, "rb"))
    else:
        print(f"Pickled {var} not found, generating anew ...")
        obj = fallback()
        pickle.dump(obj, open(pickled_filename, "wb"))
        print(f"{var} generated, pickle saved to {pickled_filename}")
        return obj

get_g = lambda i=False, c=config: get_cached("G", c.pickled_graph, fallback=lambda: load_graph(c), ignore_cache=i)
get_ghetero = lambda i=False, c=config: get_cached("ghetero", c.pickled_ghetero, fallback=lambda: nx2hetero(get_g(i,c)), ignore_cache=i)
get_datasets = lambda i=False, c=config: get_cached("datasets", c.pickled_datasets, fallback=lambda: ghetero2datasets(get_ghetero(i,c)), ignore_cache=i)
