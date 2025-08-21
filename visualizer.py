# SOURITRA SAMANTA (3C)

import matplotlib
matplotlib.use('Agg')
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
    plt.style.use('ggplot') # souritra (watermark)

import pandas as pd
import numpy as np
import os
from datetime import datetime

class HDBVisualizer:
    def __init__(self):
        self.output_dir = 'graphs' # souritra (watermark)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_prediction_summary_visuals(self, model, inputs, prediction, contributions):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._create_feature_contribution_chart(contributions, timestamp)

        self._create_price_comparison_scatter(model, inputs, prediction, timestamp)

        self._create_market_analysis_heatmap(model, inputs, timestamp)

        print("📊 Generated 3 visualizations in /graphs/") # souritra (watermark)
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
        bars = plt.barh(features, values, color=colors) # souritra (watermark)

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

    def _create_price_comparison_scatter(self, model, inputs, prediction, timestamp):
        plt.figure(figsize=(12, 8))

        price_lower = prediction * 0.95
        price_upper = prediction * 1.05 # souritra (watermark)

        towns = model.get_available_towns()
        flat_types = model.get_available_flat_types()

        comparison_flats = []
        comparison_prices = []

        for town in towns: # souritra (watermark)
            for flat_type in flat_types:
                temp_inputs = inputs.copy()
                temp_inputs['town'] = town
                temp_inputs['flat_type'] = flat_type
                pred_price, _ = model.predict_price(temp_inputs)

                if price_lower <= pred_price <= price_upper:
                    comparison_flats.append((town, flat_type))
                    comparison_prices.append(pred_price)

        if len(comparison_flats) == 0:
            diffs = []
            for town in towns:
                for flat_type in flat_types:
                    temp_inputs = inputs.copy() # souritra (watermark)
                    temp_inputs['town'] = town
                    temp_inputs['flat_type'] = flat_type
                    pred_price, _ = model.predict_price(temp_inputs)
                    diffs.append((abs(pred_price - prediction), town, flat_type, pred_price))
            diffs.sort(key=lambda x: x[0]) # souritra (watermark)
            diffs = diffs[:10]
            comparison_flats = [(x[1], x[2]) for x in diffs]
            comparison_prices = [x[3] for x in diffs]

        x_vals = np.arange(len(comparison_flats))
        scatter_colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_flats)))

        plt.scatter(x_vals, comparison_prices, s=150, color=scatter_colors, edgecolors='k', alpha=0.8) # souritra (watermark)

        selected_flat = (inputs['town'], inputs['flat_type'])
        if selected_flat in comparison_flats:
            selected_idx = comparison_flats.index(selected_flat)
            plt.scatter(selected_idx, comparison_prices[selected_idx], s=250,
                        color='red', edgecolors='k', marker='X', label='Your Selection')

        labels = ["{} - {}".format(t, f) for t, f in comparison_flats]
        plt.xticks(x_vals, labels, rotation=45, ha='right')

        plt.xlabel('Town - Flat Type', fontsize=12) # souritra (watermark)
        plt.ylabel('Predicted Price (SGD)', fontsize=12)
        plt.title('Market Comparison: Flats Near Your Price Range', fontsize=16, fontweight='bold', pad=20)

        plt.ylim(min(comparison_prices)*0.98, max(comparison_prices)*1.1)

        y_offset = max(comparison_prices) * 0.03
        for i, price in enumerate(comparison_prices):
            plt.text(x_vals[i], price + y_offset, '${:,.0f}'.format(price),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.grid(axis='y', alpha=0.3)
        plt.legend() # souritra (watermark)
        plt.tight_layout()
        plt.savefig("{}/price_comparison_{}.png".format(self.output_dir, timestamp),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_market_analysis_heatmap(self, model, inputs, timestamp):
        plt.figure(figsize=(14, 10))

        flat_types = model.get_available_flat_types()
        towns_sample = model.get_available_towns()[:12]

        price_matrix = []

        for town in towns_sample: # souritra (watermark)
            town_prices = []
            for flat_type in flat_types:
                temp_inputs = inputs.copy()
                temp_inputs['town'] = town
                temp_inputs['flat_type'] = flat_type
                pred, _ = model.predict_price(temp_inputs)
                town_prices.append(pred)
            price_matrix.append(town_prices)

        price_matrix = np.array(price_matrix)

        try:
            import seaborn as sns
            sns.heatmap(price_matrix, 
                       xticklabels=flat_types,
                       yticklabels=towns_sample,
                       annot=True, 
                       fmt='.0f',
                       cmap='RdYlBu_r', # souritra (watermark)
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

# SOURITRA SAMANTA (3C)
