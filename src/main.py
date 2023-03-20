import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
import tqdm
from matplotlib import pyplot as plt

import config
import loader
import model as M

args = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hidden_size": 150,
    "lr": 0.1,
    "weight_decay": 0.0,
    "epochs": 100,
    "loader_num_neighbors": [10, 10],
    "loader_neg_sampling_ratio": 1.0,
    "loader_batch_size": 150,
    "loader_shuffle": True,
    "sage_args": {
        "layers": 2,
        "normalize": [True, True],
        "dropout": [0.1, 0.1],
        "project": [True, True],
    },
    "linear_layer_weights": "glorot",
}

default_config = config

class Main():
    def __init__(self, args=args, config=default_config, use_cache=True):
        # Store variables
        self.args = args
        self.config = config
        self.use_cache = use_cache
        self.device = torch.device(args["device"])

        # Load data
        ghetero = loader.get_ghetero(use_cache, config)
        data_train, data_val, data_test = loader.get_datasets(use_cache, config)

        # Create model
        self.model = M.HeteroModel(args["hidden_size"], ghetero.x_dict, ghetero.metadata(), args["sage_args"], args["linear_layer_weights"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

        # Move model to device
        self.model.to_device(self.device)

        # Create train loader
        edge_label_index = data_train["track", "contains", "playlist"].edge_label_index
        edge_label = data_train["track", "contains", "playlist"].edge_label
        self.train_loader = torch_geometric.loader.LinkNeighborLoader(
            data=data_train,
            num_neighbors=args["loader_num_neighbors"],
            neg_sampling_ratio=args["loader_neg_sampling_ratio"],
            edge_label_index=(("track", "contains", "playlist"), edge_label_index),
            edge_label=edge_label,
            batch_size=args["loader_batch_size"],
            shuffle=args["loader_shuffle"],
            transform=T.ToDevice(self.device),
        )

    def to_device(self, device):
        self.model.to_device(device)
        self.device = device

    def get_args(self):
        return self.args

    def run(self, graph_render_freq=-1, batch_wrapper=loader.dummy_generator):
        losses = []
        accuracies = []

        for i in range(self.args["epochs"]):
            loss, accuracy = M.train(self.model, self.train_loader, self.optimizer, batch_wrapper=batch_wrapper)
            losses.append(loss)
            accuracies.append(accuracy)
            
            print(f"Epoch {i}: loss={loss}, accuracy={accuracy}")

            if i % graph_render_freq == 0:
                plt.plot(losses, label="loss")
                plt.plot(accuracies, label="accuracy")
                plt.legend()
                plt.yticks(np.arange(0, 1.1, 0.1))
                plt.show()

