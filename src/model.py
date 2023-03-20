import torch
import torch_geometric


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, dropout):
        super().__init__()
        self.conv1 = torch_geometric.nn.SAGEConv((-1, -1), hidden_channels, normalize=True, dropout=dropout)
        self.conv2 = torch_geometric.nn.SAGEConv((-1, -1), hidden_channels, normalize=True, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(torch.nn.Module):
    def forward(self, x_track, x_playlist, track_playlist_edge):
        track_embedding = x_track[track_playlist_edge[0]]
        playlist_embedding = x_playlist[track_playlist_edge[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (playlist_embedding * track_embedding).sum(dim=-1)

class HeteroModel(torch.nn.Module):
    def __init__(self, hidden_channels, node_features, metadata, dropout=0):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:

        self.node_lin = {
            k: torch_geometric.nn.Linear(v.shape[1], hidden_channels, weight_initializer="glorot") for k, v in node_features.items()
        }

        
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels, dropout=dropout)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = torch_geometric.nn.to_hetero(self.gnn, metadata=metadata)

        self.classifier = LinkPredictor()

    def forward(self, data):
        x_dict = {
            k: self.node_lin[k](v) for k, v in data.x_dict.items()
        }
        
        x_dict = self.gnn(x_dict, data.edge_index_dict)
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

def train(model, train_loader, optimizer, batch_wrapper=dummy_generator):
    model.train()

    total_examples = total_loss = 0
    for batch in batch_wrapper(train_loader):
        optimizer.zero_grad()
        
        out = model(batch)
        loss = torch.nn.functional.cross_entropy(
            out, batch["track", "contains", "playlist"].edge_label
        )
        loss.backward()
        optimizer.step()

        total_examples += len(out)
        total_loss += float(loss) * len(out)

    return total_loss / total_examples