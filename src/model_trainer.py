import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
from data_processor import CryptoDataset
import pandas as pd
from sklearn.metrics import r2_score
import itertools
import pickle

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.4
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        
        # L1 regularization
        self.l1_lambda = 0.01
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        normalized = self.batch_norm(last_out)
        out = self.fc1(normalized)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Add L1 regularization
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        
        return out, self.l1_lambda * l1_reg

class ModelTrainer:
    def __init__(self, config, data_processor):
        self.config = config
        self.data_processor = data_processor
        self.metrics = {}
        self.writer = SummaryWriter(log_dir=str(config.model_dir / 'runs'))
        
    def optimize_hyperparameters(self, X, y):
        """Optimize RF and SVR hyperparameters"""
        logger.info("Optimizing model hyperparameters...")
        
        # Random Forest hyperparameter grid
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        # SVR hyperparameter grid
        svr_param_grid = {
            'kernel': ['rbf', 'linear'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
        
        # Optimize Random Forest
        rf_grid = GridSearchCV(
            RandomForestRegressor(), 
            rf_param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        rf_grid.fit(X.reshape(X.shape[0], -1), y)
        
        # Optimize SVR
        svr_grid = GridSearchCV(
            SVR(), 
            svr_param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        svr_grid.fit(X.reshape(X.shape[0], -1), y)
        
        return rf_grid.best_estimator_, svr_grid.best_estimator_
    
    def train_ensemble(self, X_train, X_test, y_train, y_test, epochs):
        logger.info("Training ensemble models...")
        
        # Get feature names from data processor
        feature_names = self.data_processor.feature_names
        
        # Train individual models
        lstm_model = self._train_lstm(X_train, X_test, y_train, y_test, epochs)
        rf_model = self._train_rf(X_train, y_train)
        svr_model = self._train_svr(X_train, y_train)
        
        # Get feature importances using random forest
        X_reshaped = X_train.reshape(-1, X_train.shape[2])
        y_reshaped = np.repeat(y_train, X_train.shape[1])
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_reshaped, y_reshaped)
        
        importances = rf.feature_importances_
        important_features = np.argsort(importances)[-5:]  # Select top 5 features
        
        return EnsembleModel(
            lstm_model, 
            rf_model, 
            svr_model, 
            self.config.ensemble_weights,
            important_features
        )
    
    def _train_lstm(self, X_train, X_test, y_train, y_test, epochs):
        input_dim = X_train.shape[2]
        model = LSTMModel(input_dim, self.config.lstm_units)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            threshold=self.config.min_delta
        )
        criterion = nn.MSELoss()
        
        train_dataset = CryptoDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        val_dataset = CryptoDataset(X_test, y_test)
        val_loader = DataLoader(val_dataset, batch_size=len(X_test))
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                y_pred, l1_loss = model(batch_X)
                mse_loss = criterion(y_pred, batch_y.unsqueeze(1))
                loss = mse_loss + l1_loss  # Combined loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                optimizer.step()
                total_loss += loss.item()
            
            # Validation with both MSE and L1
            model.eval()
            with torch.no_grad():
                val_X, val_y = next(iter(val_loader))
                val_pred, val_l1_loss = model(val_X)
                val_mse = criterion(val_pred, val_y.unsqueeze(1))
                val_loss = val_mse + val_l1_loss
            
            scheduler.step(val_loss)
            
            if val_loss < best_loss - self.config.min_delta:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), str(self.config.model_dir / 'best_model.pth'))
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss = {total_loss/len(train_loader):.4f}, "
                           f"val_loss = {val_loss:.4f}, lr = {optimizer.param_groups[0]['lr']:.6f}")
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        model.load_state_dict(torch.load(str(self.config.model_dir / 'best_model.pth')))
        return model
    
    def _predict_lstm(self, model, X):
        model.eval()
        with torch.no_grad():
            return model(torch.FloatTensor(X))[0].numpy().flatten()
    
    def _log_metrics(self, y_true, y_pred, model_name):
        # Existing metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # New trading-specific metrics
        returns_true = np.diff(y_true) / y_true[:-1]
        returns_pred = np.diff(y_pred) / y_pred[:-1]
        
        # Sharpe Ratio (annualized)
        sharpe = np.sqrt(252) * np.mean(returns_pred) / np.std(returns_pred)
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns_pred)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Profit Factor
        winning_trades = returns_pred[returns_pred > 0]
        losing_trades = returns_pred[returns_pred < 0]
        profit_factor = abs(np.sum(winning_trades) / np.sum(losing_trades))
        
        # Win Rate
        win_rate = len(winning_trades) / len(returns_pred)
        
        # Risk-adjusted returns
        volatility = np.std(returns_pred) * np.sqrt(252)
        sortino_ratio = np.sqrt(252) * np.mean(returns_pred) / np.std(returns_pred[returns_pred < 0])
        
        # Maximum consecutive losses
        losses = returns_pred < 0
        max_consecutive_losses = max(sum(1 for _ in group) for value, group in itertools.groupby(losses) if value)
        
        self.metrics[model_name].update({
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'volatility': volatility,
            'sortino_ratio': sortino_ratio,
            'max_consecutive_losses': max_consecutive_losses
        })
        
        logger.info(f"\nModel: {model_name}")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R²: {r2:.4f}")
        logger.info(f"Directional Accuracy: {directional_accuracy:.2%}")
        logger.info(f"\nAdvanced Trading Metrics for {model_name}:")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Volatility: {volatility:.2f}")
        logger.info(f"Sortino Ratio: {sortino_ratio:.2f}")
        logger.info(f"Max Consecutive Losses: {max_consecutive_losses}")
        
    def get_metrics(self):
        return self.metrics

    def analyze_model_contributions(self, X_test, y_test):
        lstm_pred = self._predict_lstm(self.lstm, X_test)
        rf_pred = self.rf.predict(X_test.reshape(X_test.shape[0], -1))
        svr_pred = self.svr.predict(X_test.reshape(X_test.shape[0], -1))
        
        # Log individual model metrics
        self._log_metrics(y_test, lstm_pred, 'LSTM')
        self._log_metrics(y_test, rf_pred, 'Random Forest')
        self._log_metrics(y_test, svr_pred, 'SVR')
        
        # Calculate correlation between predictions
        pred_df = pd.DataFrame({
            'LSTM': lstm_pred,
            'RF': rf_pred,
            'SVR': svr_pred,
            'True': y_test
        })
        
        logger.info("\nPrediction Correlations:")
        logger.info(pred_df.corr())

    def _select_features(self, X, y, threshold=0.05):
        """Select most important features using Random Forest"""
        logger.info("Selecting important features...")
        rf = RandomForestRegressor(n_estimators=100)
        
        # Reshape X for feature selection
        X_reshaped = X.reshape(-1, X.shape[2])
        y_reshaped = np.repeat(y, X.shape[1])
        
        rf.fit(X_reshaped, y_reshaped)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Select features that contribute at least threshold% to prediction
        important_features = np.where(importances > threshold)[0]
        
        # Ensure we keep at least 5 features
        if len(important_features) < 5:
            important_features = np.argsort(importances)[-5:]
        
        logger.info(f"Selected {len(important_features)} important features")
        logger.info(f"Feature importances: {importances}")
        
        return important_features

    def _validate_model(self, model, X, y, n_splits=5):
        """Time series cross validation"""
        scores = []
        for i in range(n_splits):
            split_idx = len(X) - (i + 1) * len(X) // n_splits
            X_val = X[split_idx:]
            y_val = y[split_idx:]
            
            pred = model.predict(X_val)
            score = r2_score(y_val, pred)
            scores.append(score)
        
        return np.mean(scores)

    def _train_rf(self, X_train, y_train):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        # Calculate expected features
        n_samples, n_timesteps, n_features = X_train.shape
        expected_features = n_timesteps * n_features
        
        # Reshape data for RF ensuring correct dimensions
        X_reshaped = X_train.reshape(n_samples, expected_features)
        
        # Initialize and train RF model
        rf_model = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_reshaped, y_train)
        return rf_model

    def _train_svr(self, X_train, y_train):
        """Train SVR model"""
        logger.info("Training SVR model...")
        
        # Reshape data for SVR
        X_reshaped = X_train.reshape(X_train.shape[0], -1)
        
        # Initialize and train SVR model
        svr_model = SVR(
            kernel=self.config.svr_kernel,
            C=1.0,
            gamma='scale'
        )
        
        svr_model.fit(X_reshaped, y_train)
        return svr_model

    def _predict_rf(self, model, X):
        """Make predictions with Random Forest model"""
        X_reshaped = X.reshape(X.shape[0], -1)
        return model.predict(X_reshaped)

    def _predict_svr(self, model, X):
        """Make predictions with SVR model"""
        X_reshaped = X.reshape(X.shape[0], -1)
        return model.predict(X_reshaped)

    def save_ensemble(self, model, symbol):
        """Save the trained ensemble model"""
        model_path = self.config.model_dir / f"ensemble_{symbol.replace('/', '_')}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model to {model_path}")

    def load_ensemble(self, symbol):
        """Load a trained ensemble model"""
        model_path = self.config.model_dir / f"ensemble_{symbol.replace('/', '_')}.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"No saved model found at {model_path}")

class EnsembleModel:
    def __init__(self, lstm_model, rf_model, svr_model, weights, feature_indices):
        self.lstm = lstm_model
        self.rf = rf_model
        self.svr = svr_model
        self.weights = weights
        self.feature_indices = feature_indices
        self.input_shape = None  # Store input shape during training
        
    def predict(self, X):
        """Make ensemble predictions"""
        # Store input shape if not already stored
        if self.input_shape is None:
            self.input_shape = X.shape
            
        # LSTM prediction (using all features)
        lstm_pred = self._predict_lstm(X)
        
        # RF and SVR predictions (using reshaped data)
        X_reshaped = X.reshape(X.shape[0], -1)  # Reshape to (n_samples, n_features*timesteps)
        
        rf_pred = self.rf.predict(X_reshaped)
        svr_pred = self.svr.predict(X_reshaped)
        
        # Combine predictions using weights
        ensemble_pred = (
            self.weights['lstm'] * lstm_pred +
            self.weights['rf'] * rf_pred +
            self.weights['svr'] * svr_pred
        )
        
        return ensemble_pred
        
    def _predict_lstm(self, X):
        """Make predictions with LSTM model"""
        self.lstm.eval()
        with torch.no_grad():
            predictions, _ = self.lstm(torch.FloatTensor(X))
            return predictions.numpy().flatten()