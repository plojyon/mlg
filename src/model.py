import torch
import torch_geometric
from torcheval.metrics import BinaryAccuracy


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, args):
        super().__init__()
        self.conv = []
        for i in range(args["layers"]):
            self.conv.append(torch_geometric.nn.GraphConv((-1, -1), hidden_channels, normalize=args["normalize"][i], dropout=args["dropout"][i], project=args["project"][i]))
        

    def forward(self, x, edge_index):
        for conv in self.conv:
            x = conv(x, edge_index)
        return x

class LinkPredictor(torch.nn.Module):
    def forward(self, x_track, x_playlist, track_playlist_edge):
        track_embedding = x_track[track_playlist_edge[0]]
        playlist_embedding = x_playlist[track_playlist_edge[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (playlist_embedding * track_embedding).sum(dim=-1)

class HeteroModel(torch.nn.Module):
    def __init__(self, hidden_channels, node_features, metadata, graph_sage_arg, weight_initializer="glorot"):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:

        self.node_lin = {
            k: torch_geometric.nn.Linear(v.shape[1], hidden_channels, weight_initializer=weight_initializer) for k, v in node_features.items()
        }

        
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels, args=graph_sage_arg)
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

    def to_device(self, *args, **kwargs):
        self.to(*args, **kwargs)
        self.gnn.to(*args, **kwargs)
        for _, v in self.node_lin.items():
            v.to(*args, **kwargs)

def dummy_generator(source):
    for e in source:
        yield e

def train(model, train_loader, optimizer, batch_wrapper=dummy_generator):
    model.train()

    accuracy = total_examples = total_loss = 0
    for batch in batch_wrapper(train_loader):
        optimizer.zero_grad()
        
        # forward pass
        out = model(batch)
        true_labels = batch["track", "contains", "playlist"].edge_label
        loss = torch.nn.functional.mse_loss(out, true_labels)
        loss.backward()
        optimizer.step()

        # calculate loss
        total_examples += len(out)
        total_loss += float(loss) * len(out)

        # calculate binary accuracy
        metric = BinaryAccuracy()
        metric.update(out.to('cpu'), true_labels.to('cpu'))
        accuracy += metric.compute() * len(out)

    return total_loss / total_examples, accuracy / total_examples