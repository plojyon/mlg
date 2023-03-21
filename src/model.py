from torcheval.metrics import BinaryAccuracy
import torch
import torch_geometric
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = torch_geometric.nn.SAGEConv((-1, -1), hidden_channels, normalize=True, dropout=True)
        self.conv2 = torch_geometric.nn.SAGEConv((-1, -1), hidden_channels, normalize=True, dropout=True)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

class LinkPredictor(torch.nn.Module):
    def forward(self, x_track, x_playlist, track_playlist_edge):
        track_embedding = x_track[track_playlist_edge[0]]
        playlist_embedding = x_playlist[track_playlist_edge[1]]

        #print(playlist_embedding)

        # Apply dot-product to get a prediction per supervision edge:
        return (playlist_embedding * track_embedding).sum(dim=-1)

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

outs = []

def test(model, data_test):
    test_out = model(data_test.to(device)).to('cpu')
    truth = data_test["track", "contains", "playlist"].edge_label.to('cpu')

    test_loss = torch.nn.functional.mse_loss(
        test_out,
        truth
    )
    metric = BinaryAccuracy()
    metric.update(test_out, truth)
    return float(test_loss), metric.compute()

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