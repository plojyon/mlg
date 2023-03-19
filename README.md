# Graph Neural Network for Spotify Playlist Prediction
This project is a Graph Neural Network (GNN) model trained to predict songs that would fit into a Spotify playlist. The model is built using the PyTorch Geometric library and trained on a dataset of user-generated playlists from Spotify.

# Overview
The goal of this project is to provide a tool for Spotify users to easily generate playlists that fit their desired mood or genre. The GNN takes as input a playlist graph, where each node represents a song and each edge represents a similarity between songs. The model then predicts the likelihood of each song in the dataset to fit into the given playlist.

# Dataset
The dataset used to train the model is a collection of user-generated playlists from Spotify. Each playlist is represented as a graph where each node represents a song and each edge represents a similarity between songs. The similarity between songs is calculated based on Spotify's audio features and metadata. The dataset is preprocessed and split into training, validation, and test sets.

Model Architecture
The GNN model used in this project is a variation of the Graph Convolutional Network (GCN) architecture. The model takes as input a graph with song nodes and calculates the node embeddings using graph convolutions. The embeddings are then fed into a multi-layer perceptron (MLP) to predict the likelihood of each song to fit into the playlist. The model is trained using a binary cross-entropy loss function.
