import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.fig_dir = Path('figures')
        self.fig_dir.mkdir(exist_ok=True)
        
    def plot_predictions(self, y_true, predictions):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(self.config.fig_width//100, self.config.fig_height//100))
        plt.plot(y_true, label='Actual Price', color='blue', alpha=0.6)
        plt.plot(predictions, label='Predicted Price', color='red', alpha=0.6)
        plt.title('Price Predictions')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_model_performance(self, metrics, save=True):
        """Plot model performance metrics"""
        try:
            # Create subplots for MSE and MAE
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Mean Squared Error', 'Mean Absolute Error')
            )
            
            models = list(metrics.keys())
            mse_values = [m['mse'] for m in metrics.values()]
            mae_values = [m['mae'] for m in metrics.values()]
            
            # Add MSE bar chart
            fig.add_trace(
                go.Bar(x=models, y=mse_values, name='MSE'),
                row=1, col=1
            )
            
            # Add MAE bar chart
            fig.add_trace(
                go.Bar(x=models, y=mae_values, name='MAE'),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Model Performance Metrics",
                showlegend=False,
                height=500
            )
            
            if save:
                fig.write_html(self.fig_dir / 'performance.html')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting model performance: {str(e)}")
            raise
    
    def plot_feature_importance(self, feature_names, importance_scores, save=True):
        """Plot feature importance scores"""
        try:
            # Sort features by importance
            sorted_idx = np.argsort(importance_scores)
            pos = np.arange(sorted_idx.shape[0]) + .5
            
            fig = go.Figure(go.Bar(
                x=importance_scores[sorted_idx],
                y=[feature_names[i] for i in sorted_idx],
                orientation='h'
            ))
            
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=max(400, len(feature_names) * 20)
            )
            
            if save:
                fig.write_html(self.fig_dir / 'feature_importance.html')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise

    def plot_trades(self, y_true, predictions, trades):
        """Plot price with trade entry/exit points"""
        plt.figure(figsize=(self.config.fig_width//100, self.config.fig_height//100))
        
        # Plot actual prices
        plt.plot(y_true, label='Actual Price', color='blue', alpha=0.6)
        
        # Plot entry/exit points
        entry_label_long = None
        exit_label_long = None
        entry_label_short = None
        exit_label_short = None
        
        for trade in trades:
            if trade['position'] > 0:  # Long trade
                label = 'Long Entry' if entry_label_long is None else None
                plt.scatter(trade['entry_time'], trade['entry_price'], 
                          color='green', marker='^', s=100, label=label)
                if label: entry_label_long = True
                
                if trade['exit_price']:
                    label = 'Long Exit' if exit_label_long is None else None
                    plt.scatter(trade['exit_time'], trade['exit_price'], 
                              color='red', marker='v', s=100, label=label)
                    if label: exit_label_long = True
            else:  # Short trade
                label = 'Short Entry' if entry_label_short is None else None
                plt.scatter(trade['entry_time'], trade['entry_price'], 
                          color='red', marker='v', s=100, label=label)
                if label: entry_label_short = True
                
                if trade['exit_price']:
                    label = 'Short Exit' if exit_label_short is None else None
                    plt.scatter(trade['exit_time'], trade['exit_price'], 
                              color='green', marker='^', s=100, label=label)
                    if label: exit_label_short = True
        
        plt.title('Trading Performance')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show() 