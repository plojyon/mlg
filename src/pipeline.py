import sys
sys.path.insert(0, '/home/yon/jupyter-server/mlg/src/')
import pickle
import torch
import requests
from model import *

graph_location = "../spotify_million_playlist_dataset/pickles/top-ghetero-5000-fixed-maybe.pkl"
model_location = "../spotify_million_playlist_dataset/pickles/carloss72.pkl"
name2id_location = "../spotify_million_playlist_dataset/pickles/top-idx-5000.pkl"
index_location = "../spotify_million_playlist_dataset/pickles/index.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Unpickling 1/5: graph")
graph = pickle.load(open(graph_location, "rb"))
print("Unpickling 2/5: model")
model = pickle.load(open(model_location, "rb"))
print("Unpickling 3/5: name2id")
name_to_id = pickle.load(open(name2id_location, "rb"))
id_to_name = {T: {v : k for k, v in dictionary.items()} for T, dictionary in name_to_id.items()}
print("Unpickling 4/5: index")
index = pickle.load(open(index_location, "rb"))
print("Unpicking 5/5: your're mom")

def add_tracks(token, playlist, tracks):
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}',
    }

    params = {
        'uris': ",".join(tracks),
    }

    response = requests.post(f'https://api.spotify.com/v1/playlists/{playlist}/tracks', params=params, headers=headers)
    if response.status_code // 100 != 2:
        print(response.content.decode("utf-8"))
        # retry one by one
        if len(tracks) > 1:
            for track in tracks:
                add_tracks(token, playlist, [track])
    else:
        print("Added tracks to playlist:", tracks)


def get_tracks(token, playlist):
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}',
    }

    limit = 50
    offset = 0
    response = requests.get(
        f'https://api.spotify.com/v1/playlists/{playlist}/tracks?fields=items(track(uri))&limit={limit}&offset={offset}',
        headers=headers,
    )
    if response.status_code // 100 != 2:
        print(response.content.decode("utf-8"))

    tracks = response.json()["items"]
    return [track["track"]["uri"].split(":")[-1] for track in tracks]


def count_unique_connections(graph, edge_type, position=1):
    return len(torch.unique(graph[edge_type].edge_index[position, :], dim=0))

def add_playlist(graph, playlist):
    # Set new playlist id
    playlist_id = len(graph["playlist"].x)

    # Expected playlist parameters
    key_defaults = {
        "num_followers": 0,
        "collaborative": False,
        "num_edits": 0,
        "tracks": []
    }

    # Set default values for missing parameters
    playlist = {**key_defaults, **playlist}

    # Count unique elements
    playlist["num_tracks"] = len(playlist["tracks"])
    playlist["num_artists"] = count_unique_connections(graph, ("track", "authors", "artist"))
    playlist["num_albums"] = count_unique_connections(graph, ("track", "includes", "album"))

    # Sum durations
    playlist["duration_ms"] = sum([graph["track"].x[track_id][0] for track_id in playlist["tracks"]])

    # Extract playlist embedding
    playlist_embedding_array = [playlist["num_followers"], playlist["collaborative"], playlist["num_albums"], playlist["num_tracks"], playlist["num_edits"], playlist["duration_ms"], playlist["num_artists"]]
    playlist_embedding = torch.FloatTensor(playlist_embedding_array).reshape(1, -1)

    # Add playlist to graph
    graph["playlist"].x = torch.cat((graph["playlist"].x, playlist_embedding), dim=0)

    # Add track to playlist connections
    for track_id in playlist["tracks"]:
        graph["track", "contains", "playlist"].edge_index = torch.cat((graph["track", "contains", "playlist"].edge_index, torch.LongTensor([[track_id], [playlist_id]])), dim=1)

        track_artist_index = graph["track", "authors", "artist"].edge_index[0, :].eq(track_id).nonzero()
        if track_artist_index.size(0) == 0:
            raise Exception("Track {} has no artist".format(track_id))
        artist_id = graph["track", "authors", "artist"].edge_index[1, track_artist_index]
        artist_id = artist_id if artist_id.dim() == 0 else artist_id[0]
        

        track_album_index = graph["track", "includes", "album"].edge_index[0, :].eq(track_id).nonzero()
        if track_album_index.size(0) == 0:
            raise Exception("Track {} has no album".format(track_id))
        album_id = graph["track", "includes", "album"].edge_index[1, track_album_index]
        album_id = album_id if album_id.dim() == 0 else album_id[0]

        graph["track", "authors", "artist"].edge_index = torch.cat((graph["track", "authors", "artist"].edge_index, torch.LongTensor([[track_id], [artist_id]])), dim=1)
        graph["track", "includes", "album"].edge_index = torch.cat((graph["track", "includes", "album"].edge_index, torch.LongTensor([[track_id], [album_id]])), dim=1)

    # Construct index of all track connections to new playlist for prediction
    track_ids = torch.LongTensor([i for i in range(len(graph["track"].x))])
    playlist_ids = torch.LongTensor([playlist_id] * len(graph["track"].x))
    new_playlist_tracks_edge_index = torch.cat((track_ids.reshape(1, -1), playlist_ids.reshape(1, -1)), dim=0)
    graph["track", "contains", "playlist"].edge_label_index = new_playlist_tracks_edge_index



def pipeline(token, playlist_id):
    global model
    global graph
    track_ids = get_tracks(token, playlist_id)
    tracks_indices = [name_to_id["track"][track_id] for track_id in track_ids if track_id in name_to_id["track"]]
    track_names = [index["track"][idx] for idx in track_ids if idx in index["track"]]
    unknown_tracks = [idx for idx in track_ids if idx not in index["track"]]

    print("Tracks in playlist:")
    for track in track_names:
        print("\t", track)

    print("Unknown tracks:")
    for track in unknown_tracks:
        print("\t", track)

    graph = graph.to("cpu")
    add_playlist(graph, {
        "collaborative": False,
        "num_edits": 1,
        "num_followers": 1000,  # playlist will be predicted better if it's popular
        "tracks": tracks_indices
    })
    model = model.to(device)
    graph = graph.to(device)

    with torch.no_grad():
        pred = model(graph)
        most_likely = torch.topk(pred, 10, dim=0)
    
    new_track_ids = [id_to_name["track"][i.item()] for i in most_likely.indices]
    add_tracks(token, playlist_id, new_track_ids)
    return [index["track"][id.split(":")[2]] for id in new_track_ids]

import sys
if __name__ == "__main__":
    token = sys.argv[1]  # "BQCBFR_URJjTO5tTxTf1-2vwHtxsvQhnETfBLCFh6m82FIFHlA16npksr7iYVuZAHfY1ayNtN0AYsmCEEFLGQo504oAuDQAj0BIZ50ihCQcWIn2HMvxLTHccuiYb7VpeHJ67XU1SYantb9SQllVRPJmc1chW157cjQceRT_NWBNX0o4WQDOGvNfHeXm3s-NbdsoHCl5jHqaqxSOUx8KtXBWd01xMDmI6LenykFj_lAUDzJ_lLGWi7_Hw2VT8Js1nRg"
    playlist_id = sys.argv[2]  # "37i9dQZF1DXcBWIGoYBM5M"
    print("Tracks added: {}".format(pipeline(token, playlist_id)))
