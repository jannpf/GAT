import argparse
import time

import optuna
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from . import GAT, DGIModel, clustering


def tune(data, accumulation_steps=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.to(device)
    G_nx = to_networkx(data, to_undirected=True)
    n_features = data.num_features
    data_loader = DataLoader([data], batch_size=1, shuffle=True)

    def objective(trial):
        start_time = time.time()

        # Hyperparameter ranges
        hidden_channels = trial.suggest_int('hidden_channels', 8, 64)
        out_channels = trial.suggest_int('out_channels', 4, 32)
        num_layers = 2
        heads = trial.suggest_int('heads', 2, 16)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.8, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        # # deezer
        # hidden_channels = trial.suggest_int('hidden_channels', 16, 128)
        # out_channels = trial.suggest_int('out_channels', 16, 64)
        # num_layers = trial.suggest_int('num_layers', 2, 4)
        # heads = trial.suggest_int('heads', 2, 32)
        # learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        # dropout = trial.suggest_float('dropout', 0.1, 0.8, log=True)
        # weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        # Initialize the model with the sampled hyperparameters
        model = DGIModel.DGIModel(
            in_channels=n_features,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout)
        model = model.to(device)

        # Define the optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Training loop
        num_epochs = 100
        for _ in range(num_epochs):
            for i, batch in enumerate(data_loader):
                model.train()
                optimizer.zero_grad()

                pos_z, neg_z, summary = model(batch)
                loss = model.dgi.loss(pos_z, neg_z, summary)
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()


        # get embeddings
        model.eval()
        with torch.no_grad():
            embeddings = model.encoder(data.x, data.edge_index).cpu().numpy()

        # apply kmeans and optics, determine modularity
        metrics, best_n, communities = clustering.kmeans(G_nx, embeddings)
        best_m = metrics['Modularity']
        best_method = "kmeans"

        metrics_optics, optics_n, c_opt = clustering.optics(G_nx, embeddings)
        optics_m = metrics_optics['Modularity']
        if optics_m > best_m:
            metrics = metrics_optics
            best_m = optics_m
            best_n = optics_n
            best_method = "optics"
            communities = c_opt

        end_time = time.time()
        elapsed_time = end_time - start_time

        trial.set_user_attr('elapsed_time', elapsed_time)
        trial.set_user_attr('metrics', metrics)
        trial.set_user_attr('final_loss', loss.item())
        trial.set_user_attr('best_method', best_method)
        trial.set_user_attr('n_clusters', best_n)
        trial.set_user_attr('communities', communities)

        # return the final modularity as the objective to maximize
        return best_m

    study = optuna.create_study(direction='maximize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    return study

    # return {
    #     'best_params': study.best_params,
    #     'best_value': study.best_value,
    #     'n_clusters': study.best_trial.user_attrs['n_clusters'],
    #     'final_loss': study.best_trial.user_attrs['final_loss'],
    #     'best_method': study.best_trial.user_attrs['best_method']
    # }


def tune_variance_contrastive(data, G, loss_function = "variance"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.to(device)
    n_features = data.num_features

    def objective(trial):
        start_time = time.time()

        # Hyperparameter ranges
        hidden_channels = trial.suggest_int('hidden_channels', 8, 64)
        out_channels = trial.suggest_int('out_channels', 4, 32)
        heads = trial.suggest_int('heads', 2, 16)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.8, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        # Initialize the model with the sampled hyperparameters
        model = GAT.GAT(
            num_features=n_features,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_heads=heads,
            dropout=dropout)
        model = model.to(device)

        # Define the optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Define the training loop
        def train():
            model.train()
            optimizer.zero_grad()
            out = model(data)
            if loss_function == "variance":
                loss = -torch.var(out)
            else:
                loss = GAT.contrastive_loss(out, G)
            loss.backward()
            optimizer.step()
            return loss.item()

        # Training process
        NUM_EPOCHS = 100
        for epoch in range(NUM_EPOCHS):
            loss = train()
            # if epoch % 10 == 0:
            #     print(f"Epoch {epoch}, Loss: {loss:.4f}")

        # get embeddings
        model.eval()
        with torch.no_grad():
            embeddings = model(data).cpu().numpy()

        # apply kmeans and optics, determine modularity
        metrics, best_n, communities = clustering.kmeans(G, embeddings)
        best_m = metrics['Modularity']
        best_method = "kmeans"

        metrics_optics, optics_n, c_opt = clustering.optics(G, embeddings)
        optics_m = metrics_optics['Modularity']
        if optics_m > best_m:
            metrics = metrics_optics
            best_m = optics_m
            best_n = optics_n
            best_method = "optics"
            communities = c_opt

        end_time = time.time()
        elapsed_time = end_time - start_time

        trial.set_user_attr('elapsed_time', elapsed_time)
        trial.set_user_attr('metrics', metrics)
        trial.set_user_attr('final_loss', loss)
        trial.set_user_attr('n_clusters', best_n)
        trial.set_user_attr('best_method', best_method)
        trial.set_user_attr('communities', communities)

        # return the final modularity as the objective to maximize
        return best_m

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    return study

    # return {
    #     'best_params': study.best_params,
    #     'best_value': study.best_value,
    #     'n_clusters': study.best_trial.user_attrs['n_clusters'],
    #     'final_loss': study.best_trial.user_attrs['final_loss'],
    #     'best_method': study.best_trial.user_attrs['best_method']
    # }
