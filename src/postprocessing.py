import os
from tqdm import tqdm
import json

def make_index(dataset_path):
    index = {
        "playlist": {},
        "track": {},
        "artist": {},
        "album": {},
    }

    filenames = os.listdir(dataset_path)
    for i in tqdm(range(len(filenames)), unit="files"):
        with open(os.path.join(dataset_path, filenames[i])) as json_file:
            playlists = json.load(json_file)["playlists"]
            for playlist in playlists:
                index["playlist"][playlist["pid"]] = playlist["name"]
                for track in playlist["tracks"]:
                    index["track"][track["track_uri"].split(":")[2]] = track["track_name"]
                    index["artist"][track["artist_uri"].split(":")[2]] = track["artist_name"]
                    index["album"][track["album_uri"].split(":")[2]] = track["album_name"]
    return index


def make_song2artist(dataset_path):
    song2artist = {}

    filenames = os.listdir(dataset_path)
    for i in tqdm(range(len(filenames)), unit="files"):
        with open(os.path.join(dataset_path, filenames[i])) as json_file:
            playlists = json.load(json_file)["playlists"]
            for playlist in playlists:
                for track in playlist["tracks"]:
                    song2artist[track["track_uri"]] = track["artist_uri"]
