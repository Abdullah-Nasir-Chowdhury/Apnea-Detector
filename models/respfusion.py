import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np

class RespFusion(nn.Module):
    def __init__(self, tft_model, tcn_model, ets_model, bilstm_model, meta_learner_path=None, weights=None, strategy='stacking',):
        super(RespFusion, self).__init__()
        self.tft = tft_model
        self.tcn = tcn_model
        self.ets = ets_model
        self.bilstm = bilstm_model
        self.strategy = strategy  # 'stacking' or other strategies

        # Initialize XGBoost meta-learner
        self.meta_learner = xgb.XGBRegressor()  # Or XGBClassifier for classification

        # Load the meta-learner if a path is provided
        if meta_learner_path is not None:
            self.meta_learner.load_model(meta_learner_path)
            print(f"Meta-learner loaded from {meta_learner_path}")

        # Storage for stacking training data
        self.stacking_features = []
        self.stacking_targets = []
        
        # Set model weights for ensembling, default to equal weights for weighted_average strategy
        if strategy == 'weighted_average':
            if weights is None:
                self.weights = [1.0, 1.0, 1.0, 1.0]
            else:
                assert len(weights) == 4, "Weights must match the number of models."
                self.weights = weights
                
        
    def forward(self, x):
        # Get predictions from each base model
        tft_output = self.tft(x).detach().cpu().numpy()
        tcn_output = self.tcn(x).detach().cpu().numpy()
        ets_output = self.ets(x).detach().cpu().numpy()
        bilstm_output = self.bilstm(x).detach().cpu().numpy()

        if self.strategy == 'stacking':
            # Combine outputs into features for the meta-learner
            features = np.column_stack((tft_output, tcn_output, ets_output, bilstm_output))
            # During inference, use the meta-learner to make predictions
            ensemble_output = self.meta_learner.predict(features)
            return torch.tensor(ensemble_output).to(x.device).float()
        
        elif self.strategy == 'voting':
            # For soft voting, calculate the average
            ensemble_output = torch.mean(torch.stack([tft_output, tcn_output, ets_output, bilstm_output], dim=0), dim=0)
            return ensemble_output

        elif self.strategy == 'weighted_average':
            # Weighted average of outputs
            ensemble_output = (
                self.weights[0] * tft_output +
                self.weights[1] * tcn_output +
                self.weights[2] * ets_output +
                self.weights[3] * bilstm_output
            ) / sum(self.weights)
            return ensemble_output
        
        elif self.strategy == 'simple_average':
            # Simple average of outputs
            ensemble_output = (tft_output + tcn_output + ets_output + bilstm_output) / 4
            return ensemble_output
        
        
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}. Currently supports only 'stacking', 'voting', 'weighted_average', and 'simple_average'.")

    def collect_stacking_data(self, x, y):
        """Collect base model outputs and corresponding targets for meta-learner training."""
        tft_output = self.tft(x).detach().cpu().numpy()
        tcn_output = self.tcn(x).detach().cpu().numpy()
        ets_output = self.ets(x).detach().cpu().numpy()
        bilstm_output = self.bilstm(x).detach().cpu().numpy()

        # Stack features and store
        features = np.column_stack((tft_output, tcn_output, ets_output, bilstm_output))
        self.stacking_features.append(features)
        self.stacking_targets.append(y.detach().cpu().numpy())

    def train_meta_learner(self, save_path=None):
        """Train the XGBoost meta-learner on collected data and save the model."""
        # Concatenate all collected features and targets
        X = np.vstack(self.stacking_features)
        y = np.concatenate(self.stacking_targets)

        # Train the XGBoost model
        self.meta_learner.fit(X, y)
        print("Meta-learner trained successfully!")

        # Save the trained meta-learner
        if save_path:
            self.meta_learner.save_model(save_path)
            print(f"Meta-learner saved to {save_path}")

