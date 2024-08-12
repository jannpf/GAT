import optuna
import torch

from sklearn.cluster import KMeans, OPTICS

from src.GAT import GAT
from . import community_results


def tune(data, G):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.to(device)
    n_features = data.num_features

    def objective(trial):
        # Hyperparameter ranges
        hidden_channels = trial.suggest_int('hidden_channels', 8, 64)
        out_channels = trial.suggest_int('out_channels', 4, 32)
        heads = trial.suggest_int('heads', 2, 16)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.8, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        # Initialize the model with the sampled hyperparameters
        model = GAT(
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
            from src.GAT_baseline import contrastive_loss
            model.train()
            optimizer.zero_grad()
            out = model(data)
            # loss = -torch.var(out)
            loss = contrastive_loss(out, G)
            loss.backward()
            optimizer.step()
            return loss.item()

        # Training process
        NUM_EPOCHS = 100
        for epoch in range(NUM_EPOCHS):
            loss = train()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        # get embeddings
        model.eval()
        with torch.no_grad():
            embeddings = model(data).detach().numpy()

        # apply kmeans and optics, determine modularity
        best_m = -1
        best_n = 0

        # kmeans
        for n in range(2, 12):
            kmeans = KMeans(n_clusters=n)
            clusters = kmeans.fit_predict(embeddings)
            modularity = community_results.community_metrics(G, dict(zip(range(G.number_of_nodes()), clusters)))['Modularity']
            if modularity > best_m:
                best_n, best_m = n, modularity

        # optics
        optics = OPTICS(min_samples=5)
        clusters = optics.fit_predict(embeddings)
        modularity = community_results.community_metrics(G, dict(zip(range(G.number_of_nodes()), clusters)))['Modularity']
        if modularity > best_m:
            best_n, best_m = 'OPTICS', modularity


        trial.set_user_attr('final_loss', loss)
        trial.set_user_attr('n_clusters', best_n)

        # return the final modularity as the objective to maximize
        return best_m

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # return study

    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_clusters': study.best_trial.user_attrs['n_clusters'],
        'final_loss': study.best_trial.user_attrs['final_loss']
    }


if __name__ == "__main__":
    import argparse
    from src.GAT_baseline import load_data
    from torch_geometric.data import Data
    parser = argparse.ArgumentParser(description="Select a dataset")
    parser.add_argument(
        "-d",
        type=str,
        choices=["karate", "univ", "enron", "deezer"],
        required=True,
        help="Specify the dataset: karate, univ, enron, or deezer",
    )
    # load data
    args = parser.parse_args()
    DATASET = args.d
    G, edge_index, adj_matrix = load_data(DATASET)
    data = Data(edge_index=edge_index)
    data.num_nodes = G.number_of_nodes()
    num_features = data.num_nodes  # We'll use one-hot encodings of nodes as features
    data.x = torch.eye(data.num_nodes)
    best_p = tune(data, G)
    print(best_p)
