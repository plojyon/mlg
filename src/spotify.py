import requests
import json

token = "BQDa6MSn76FGEYYHkPhx_W-7LrbS0qAdaTXPsvCmhs3p-1b8gxi5AhXYb_Aa05TqQnbqyzBAPhFqeEmyLnP-mh85HlWRY_IzsxV2qxvKfoDC6l8y2FkGSa_VKzxcZ-wmDEAjQN-1bCv7WRxup14VbHQPB42RIOOu5kK1QJBo23EWggSixyumeMVoAS4gkHHBHBy-j8mkkJ41crrVBrSprTe-qiRcOLA1oea3cuxZn3aHtMMypFr1CEEqTfSjx_Ax6X"

def make_playlist(token, name, desc, user, public=False):
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}',
    }

    json_data = {
        'name': name,
        'description': desc,
        'public': public,
    }

    resp = requests.post(
        f'https://api.spotify.com/v1/users/{user}/playlists',
        headers=headers,
        json=json_data,
    )
    if resp.status_code // 100 != 2:
        raise RuntimeError(resp.content.decode("utf-8"))

    return json.loads(resp.content.decode("utf-8"))["uri"]

"""
{
   "collaborative":false,
   "description":"My new playlist",
   "external_urls":{
      "spotify":"https://open.spotify.com/playlist/28mrnxlCxntIsg0PzcBjEW"
   },
   "followers":{
      "href":"None",
      "total":0
   },
   "href":"https://api.spotify.com/v1/playlists/28mrnxlCxntIsg0PzcBjEW",
   "id":"28mrnxlCxntIsg0PzcBjEW",
   "images":[
      
   ],
   "name":"New Playlist 2",
   "owner":{
      "display_name":"plojyon",
      "external_urls":{
         "spotify":"https://open.spotify.com/user/316yrpt7nwy7lnxo2ihtwdwez65u"
      },
      "href":"https://api.spotify.com/v1/users/316yrpt7nwy7lnxo2ihtwdwez65u",
      "id":"316yrpt7nwy7lnxo2ihtwdwez65u",
      "type":"user",
      "uri":"spotify:user:316yrpt7nwy7lnxo2ihtwdwez65u"
   },
   "primary_color":"None",
   "public":false,
   "snapshot_id":"MSwwYmVmMTJhMWQxNjczYjJkMjQwNGI0YjgzNDE4ZTdlYTlmOTRlNDFj",
   "tracks":{
      "href":"https://api.spotify.com/v1/playlists/28mrnxlCxntIsg0PzcBjEW/tracks",
      "items":[
         
      ],
      "limit":100,
      "next":"None",
      "offset":0,
      "previous":"None",
      "total":0
   },
   "type":"playlist",
   "uri":"spotify:playlist:28mrnxlCxntIsg0PzcBjEW"
}
"""

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
        raise RuntimeError(response.content.decode("utf-8"))


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
        raise RuntimeError(response.content.decode("utf-8"))

    tracks = response.json()["items"]
    return [track["track"]["uri"].split(":")[-1] for track in tracks]

playlist_id = make_playlist(token, "New Playlist 2", "My new playlist", "316yrpt7nwy7lnxo2ihtwdwez65u").split(":")[-1]
tracks = [
    'spotify:track:5UTZUFeHTZSBzd80m5qLnv',
    'spotify:track:2rfRICVLPoWlFVGx7f48Cw',
    'spotify:track:2HKm3A8EfzKA9rjo3b5ztj',
    'spotify:track:3cjO7F0gJka4Yf1RoFXwqp',
    'spotify:track:4dh34dyav8M5jCN3yidn8Z',
    'spotify:track:4W7bw1eTUKS6dYtDNzoIJI',
    'spotify:track:56dVmVX5YYdQ019Yq3j6wL',
    'spotify:track:1xJ4bCEkzAyf1WUZ99yoKU',
    'spotify:track:3yu6Kx2hHNcRiarcKkJHJp',
    'spotify:track:3qaPwSJau5W7KiObZmeJCb',
    'spotify:track:7lXZ4Vm3j2fXuNzWI0ixvG'
]
print("Playlist ID:", playlist_id)
add_tracks(token, playlist_id, tracks)

tracks = get_tracks(token, playlist_id)
print(tracks)
