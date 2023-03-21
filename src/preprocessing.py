import itertools
import time
import pickle
import config
import networkx as nx
from tqdm import tqdm

def get_neigh_of_edge_type(G, edge_type, node):
    undirected_neigh = itertools.chain(G.neighbors(node), G.predecessors(node))
    return [n for n in undirected_neigh if G.succ[node].get(n, dict()).get('edge_type', None) == edge_type or G.pred[node].get(n, dict()).get('edge_type', None) == edge_type]

def top_n_by_followers(G, n, node_type):
  """Get all nodes of type `node_type`."""
  playlists = [node for node in G.nodes(data=True) if node[1]["node_type"] == node_type]
  return sorted(playlists, key=lambda x:"num_followers" in x[1] and x[1]["num_followers"], reverse=True)[:n]

def get_smart_playlist_subset(G, playlists_to_keep):
    keep_nodes = set()
    for node in playlists_to_keep:
        keep_nodes.add(node[0])
        tracks = get_neigh_of_edge_type(G, "track-playlist", node[0])
        artists_and_albums = []

        for track in tracks:
            artists_and_albums += get_neigh_of_edge_type(G, "track-artist", track)
            artists_and_albums += get_neigh_of_edge_type(G, "track-album", track)
        
        keep_nodes = keep_nodes.union(set(tracks))
        keep_nodes = keep_nodes.union(set(artists_and_albums))
    return keep_nodes

def smart_split(G, splits=[100,500,1000,5000,10000], pickle_location=config.pickled_top_G):
    for i in tqdm(splits):
        print(f"[{i}] started")
        start = time.time()
        playlists_to_keep = top_n_by_followers(G, i, "playlist")
        print(f"[{i}] got top n playlists in {time.time() - start} seconds")
        start = time.time()
        keep_nodes = get_smart_playlist_subset(G, playlists_to_keep)
        print(f"[{i}] finshed getting neighbors in {time.time() - start} seconds")
        print(f"\t({len(keep_nodes)} nodes = {len(keep_nodes)/len(G.nodes)} % of graph)")
        start = time.time()
        G_sub = nx.Graph(G.subgraph(keep_nodes))
        print(f"[{i}] finished subgraphing in {time.time() - start} seconds")
        start = time.time()
        pickle.dump(G_sub, open(pickle_location(i), "wb"))
        print(f"[{i}] finished pickling in {time.time() - start} seconds")