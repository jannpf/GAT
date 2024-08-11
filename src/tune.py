import optuna
import torch
from torch_geometric.loader import DataLoader

from . import DGIModel


def tune(data, accumulation_steps=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.to(device)
    n_features = data.num_features
    data_loader = DataLoader([data], batch_size=16, shuffle=True)

    def objective(trial):
        # Hyperparameter ranges
        hidden_channels = trial.suggest_int('hidden_channels', 8, 128)
        out_channels = trial.suggest_int('out_channels', 8, 64)
        num_layers = trial.suggest_int('num_layers', 2, 5)
        heads = trial.suggest_int('heads', 1, 16)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.8, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        # Initialize the model with the sampled hyperparameters
        model = DGIModel.DGIModel(in_channels=n_features, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, heads=heads, dropout=dropout)
        model = model.to(device)
        
        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Training loop
        num_epochs = 100
        for _ in range(num_epochs):
            for i, data in enumerate(data_loader):
                model.train()
                optimizer.zero_grad()
                
                pos_z, neg_z, summary = model(data)
                loss = model.dgi.loss(pos_z, neg_z, summary)
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        # For Optuna, we return the final loss as the objective to minimize
        return loss.item()
    

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    return study.best_params
