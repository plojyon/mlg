import subprocess
import sys

from flask import Flask, request
from flask_cors import CORS

#import sys
#sys.path.insert(0, '/home/yon/jupyter-server/mlg/src/')
import pickle
import torch
import requests
import torch_geometric
from torcheval.metrics import BinaryAccuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = torch_geometric.nn.SAGEConv((-1, -1), hidden_channels, normalize=True, dropout=True, bias=True, dropout_prob=0.1)
        self.conv2 = torch_geometric.nn.SAGEConv((-1, -1), hidden_channels, normalize=True, dropout=True, bias=True, dropout_prob=0.1)
        self.conv3 = torch_geometric.nn.SAGEConv((-1, -1), hidden_channels)
        #self.conv4 = torch_geometric.nn.SAGEConv((-1, -1), hidden_channels, normalize=True, dropout=True)

        self.reset_parameters()


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.conv3(x, edge_index)
        #x = torch.nn.functional.leaky_relu(x, negative_slope=0.5)
        #x = self.conv4(x, edge_index)
        #x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        #self.conv4.reset_parameters()

class LinkPredictor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        #self.linear1 = torch.nn.LazyLinear(32)
        #self.linear2 = torch.nn.LazyLinear(1)

    def forward(self, x_track, x_playlist, track_playlist_edge):
        track_embedding = x_track[track_playlist_edge[0]]
        playlist_embedding = x_playlist[track_playlist_edge[1]]

        #print(track_embedding.shape)
        #print(playlist_embedding.shape)


        #print(playlist_embedding)

        # Apply dot-product to get a prediction per supervision edge:

        #linear_in = torch.cat([track_embedding, playlist_embedding], dim=-1)
        #linear_out = self.linear1(linear_in)
        #linear_out = torch.nn.functional.leaky_relu(linear_out, negative_slope=0.2)
        #linear_out = self.linear2(linear_out)

        return (track_embedding * playlist_embedding).sum(dim=-1)

class HeteroModel(torch.nn.Module):
    def __init__(self, hidden_channels, node_features, metadata):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:

        self.node_lin = {
            k: torch.nn.Linear(v.shape[1], hidden_channels).to(device) for k, v in node_features.items()
        }
        
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels).to(device)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = torch_geometric.nn.to_hetero(self.gnn, metadata=metadata).to(device)

        self.classifier = LinkPredictor().to(device)
    
    def embed(self, data):
        x_dict = {
            k: self.node_lin[k](v) for k, v in data.x_dict.items()
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        return x_dict

    def forward(self, data):
        x_dict = self.embed(data)
        pred = self.classifier(
            x_dict["track"],
            x_dict["playlist"],
            data["track", "contains", "playlist"].edge_label_index,
        )
        return pred

    def reset_parameters(self):
        for _, v in self.node_lin.items():
            torch.nn.init.xavier_uniform_(v.weight)
        self.gnn.reset_parameters()

def dummy_generator(source):
    for e in source:
        yield e

outs = []

def test(model, data_test):
    with torch.no_grad():
        test_out = model(data_test.to(device)).to('cpu')
        truth = data_test["track", "contains", "playlist"].edge_label.to('cpu')

    test_loss = torch.nn.functional.mse_loss(
        test_out,
        truth
    )
    metric = BinaryAccuracy()
    metric.update(test_out, truth)
    return float(test_loss), metric.compute(), test_out, truth

def train(model, train_loader, optimizer, batch_wrapper=dummy_generator):
    model.train()

    accuracy = 0

    total_examples = total_loss = 0
    for i, batch in enumerate(batch_wrapper(train_loader)):
        optimizer.zero_grad()
        
        out = model(batch)
        truth = batch["track", "contains", "playlist"].edge_label


        ind = torch.randint(len(out),(5,))

        if(i % 10 == 0):
            #print(out[:10])
            #print(batch["track", "contains", "playlist"].edge_label[:10])
            pass
        loss = torch.nn.functional.mse_loss(
            out, truth
        )
        loss.backward()
        optimizer.step()

        aute_gledam = out.to('cpu')

        outs.append(aute_gledam)

        metric = BinaryAccuracy()
        metric.update(aute_gledam, truth.to('cpu'))
        accuracy += metric.compute() * len(out)

        total_examples += len(out)
        total_loss += float(loss) * len(out)

    return total_loss / total_examples, accuracy / total_examples

graph_location = "./pickles/top-ghetero-5000-fixed-maybe_cpu.pkl"
model_location = "./pickles/carloss72_cpu.pkl"
name2id_location = "./pickles/top-idx-5000_cpu.pkl"

print("Unpickling 1/3: graph")
graph = pickle.load(open(graph_location, "rb"))
print("Unpickling 2/3: model")
model = pickle.load(open(model_location, "rb"))
print("Unpickling 3/3: name2id")
name_to_id = pickle.load(open(name2id_location, "rb"))
id_to_name = {T: {v : k for k, v in dictionary.items()} for T, dictionary in name_to_id.items()}

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
        raise RuntimeError(response.content.decode("utf-8"))

    tracks = response.json()["items"]
    return [track["track"]["uri"] for track in tracks]


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
    playlist_ids = torch.LongTensor([playlist_id] * len(track_ids))
    new_playlist_tracks_edge_index = torch.cat((track_ids.reshape(1, -1), playlist_ids.reshape(1, -1)), dim=0)
    graph["track", "contains", "playlist"].edge_label_index = new_playlist_tracks_edge_index



def pipeline(token, playlist_id):
    global model
    global graph
    track_ids = get_tracks(token, playlist_id)
    tracks_indices = [name_to_id["track"][track_id] for track_id in track_ids if track_id in name_to_id["track"]]

    track_names = [name_to_id["track"][idx] for idx in track_ids if idx in name_to_id["track"]]
    unknown_tracks = [idx for idx in track_ids if idx not in name_to_id["track"]]

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
        pred[tracks_indices] = -1  # do not predict tracks already on the playlist
        most_likely = torch.topk(pred, 10, dim=0)
    
    new_track_ids = [id_to_name["track"][i.item()] for i in most_likely.indices]
    add_tracks(token, playlist_id, new_track_ids)
    return [id.split(":")[2] for id in new_track_ids]

import sys
if __name__ == "__main__":
    token = sys.argv[1]
    playlist_id = sys.argv[2]

app = Flask(__name__)
CORS(app)

@app.route("/extend", methods = ["POST"])
def extend():
    # get token and playlist id from request
    token = request.json["token"]
    playlist_id = request.json["playlist_id"]
    return "Tracks added: {}".format(pipeline(token, playlist_id))