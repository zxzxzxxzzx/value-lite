import matplotlib.pyplot as plt
try:
    import seaborn as sns
    try:
        plt.style.use('seaborn')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass
    sns.set_palette("husl")
except ImportError:
    plt.style.use('ggplot')

import pandas as pd
import numpy as np
import os
from datetime import datetime

class HDBVisualizer:
    def __init__(self):
        self.output_dir = 'graphs'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_prediction_summary_visuals(self, model, inputs, prediction, contributions):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._create_feature_contribution_chart(contributions, timestamp)

        self._create_price_comparison_graph(model, inputs, prediction, timestamp)

        self._create_market_analysis_heatmap(model, inputs, timestamp)

        print("📊 Generated 3 visualizations in {}/ folder".format(self.output_dir))
        return [
            "{}/feature_contributions_{}.png".format(self.output_dir, timestamp),
            "{}/price_comparison_{}.png".format(self.output_dir, timestamp),
            "{}/market_heatmap_{}.png".format(self.output_dir, timestamp)
        ]

    def _create_feature_contribution_chart(self, contributions, timestamp):
        plt.figure(figsize=(12, 8))

        features = list(contributions.keys())
        values = list(contributions.values())

        try:
            colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(features)))
        except AttributeError:
            colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        bars = plt.barh(features, values, color=colors)

        plt.title('Feature Contributions to HDB Price Prediction', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Contribution Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)

        for i, (bar, value) in enumerate(zip(bars, values)):
            plt.text(value + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                     '{:.1f}'.format(value), ha='left', va='center', fontweight='bold')

        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        plt.savefig("{}/feature_contributions_{}.png".format(self.output_dir, timestamp), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_price_comparison_graph(self, model, inputs, prediction, timestamp):
        plt.figure(figsize=(12, 8))

        towns = model.get_available_towns()[:10]
        predictions = []

        for town in towns:
            try:
                temp_inputs = inputs.copy()
                temp_inputs['town'] = town
                pred, _ = model.predict_price(temp_inputs)
                predictions.append(pred)
            except:
                predictions.append(0)

        bars = plt.bar(range(len(towns)), predictions, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(towns))))

        current_town_idx = None
        try:
            current_town_idx = towns.index(inputs['town'])
            bars[current_town_idx].set_color('red')
            bars[current_town_idx].set_alpha(0.8)
        except (ValueError, KeyError):
            pass

        plt.title('HDB Price Comparison Across Different Towns', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Towns', fontsize=12)
        plt.ylabel('Predicted Price (SGD)', fontsize=12)
        plt.xticks(range(len(towns)), towns, rotation=45, ha='right')

        for i, (bar, value) in enumerate(zip(bars, predictions)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(predictions) * 0.01,
                     '${:,.0f}'.format(value), ha='center', va='bottom', fontsize=10, fontweight='bold')

        if current_town_idx is not None:
            plt.axhline(y=prediction, color='red', linestyle='--', alpha=0.7, 
                       label='Your Prediction: ${:,.0f}'.format(prediction))
            plt.legend()

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        plt.savefig("{}/price_comparison_{}.png".format(self.output_dir, timestamp), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_market_analysis_heatmap(self, model, inputs, timestamp):
        plt.figure(figsize=(14, 10))

        flat_types = model.get_available_flat_types()
        towns_sample = model.get_available_towns()[:12]

        price_matrix = []

        for town in towns_sample:
            town_prices = []
            for flat_type in flat_types:
                try:
                    temp_inputs = inputs.copy()
                    temp_inputs['town'] = town
                    temp_inputs['flat_type'] = flat_type
                    pred, _ = model.predict_price(temp_inputs)
                    town_prices.append(pred)
                except:
                    town_prices.append(np.nan)
            price_matrix.append(town_prices)

        price_matrix = np.array(price_matrix)

        try:
            import seaborn as sns
            sns.heatmap(price_matrix, 
                       xticklabels=flat_types,
                       yticklabels=towns_sample,
                       annot=True, 
                       fmt='.0f',
                       cmap='RdYlBu_r',
                       cbar_kws={'label': 'Predicted Price (SGD)'})
        except ImportError:
            plt.imshow(price_matrix, cmap='RdYlBu_r', aspect='auto')
            plt.colorbar(label='Predicted Price (SGD)')
            plt.xticks(range(len(flat_types)), flat_types, rotation=45)
            plt.yticks(range(len(towns_sample)), towns_sample)

        plt.title('HDB Price Heatmap: Towns vs Flat Types', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Flat Types', fontsize=12)
        plt.ylabel('Towns', fontsize=12)

        if inputs['town'] in towns_sample and inputs['flat_type'] in flat_types:
            current_town_idx = towns_sample.index(inputs['town'])
            current_flat_idx = flat_types.index(inputs['flat_type'])
            plt.scatter(current_flat_idx, current_town_idx, s=200, c='red', marker='X', 
                       label='Your Selection', linewidths=2, edgecolors='white')
            plt.legend()

        plt.tight_layout()

        plt.savefig("{}/market_heatmap_{}.png".format(self.output_dir, timestamp), 
                   dpi=300, bbox_inches='tight')
        plt.close()