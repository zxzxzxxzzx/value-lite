import os
import pandas as pd

from hdb_polynomial_model import HDBPolynomialPriceModel
from data_processor import HDBDataProcessor
from visualizer import HDBVisualizer

try: 
    from colorama import init, Fore, Back, Style
    init(autoreset=True) 
    COLORAMA_AVAILABLE = True 
except ImportError:
    COLORAMA_AVAILABLE = False

    class DummyColor: 
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = BRIGHT = RESET_ALL = ""

    Fore = Back = Style = DummyColor()


class SimplifiedHDBCalculatorCLI:

    def __init__(self):
        self.model = HDBPolynomialPriceModel()
        self.processor = HDBDataProcessor()
        self.visualizer = HDBVisualizer()
        self.session_predictions = []

    def clear_screen(self):
        import os
        import sys

        try:
            if os.name == 'nt':
                os.system('cls')
            else:
                os.system('clear')
        except:
            print('\n' * 100, end='')
            try:
                sys.stdout.flush()
            except:
                pass

    def print_colored(self, text, color=None, style=None, end='\n'):
        if COLORAMA_AVAILABLE and color:
            color_code = getattr(Fore, color.upper(), '')
            style_code = getattr(Style, style.upper(), '') if style else ''
            print("{}{}{}{}".format(style_code, color_code, text,
                                    Style.RESET_ALL),
                  end=end)
        else:
            print(text, end=end)

    def print_rainbow(self, text, end='\n'):
        if COLORAMA_AVAILABLE:
            colors = [
                Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.BLUE,
                Fore.MAGENTA
            ]
            rainbow_text = ""
            for i, char in enumerate(text):
                if char != ' ':
                    color = colors[i % len(colors)]
                    rainbow_text += "{}{}{}".format(color, char,
                                                    Style.RESET_ALL)
                else:
                    rainbow_text += char
            print(rainbow_text, end=end)
        else:
            print(text, end=end)

    def display_banner(self):
        banner = """
 _   _ ____  ____   __     __    _             _   _             
| | | |  _ \\| __ \\  \\ \\   / /_ _| |_   _  __ _| |_(_) ___  _ __  
| |_| | | | |  _ \\   \\ \\ / / _` | | | | |/ _` | __| |/ _ \\| '_ \\ 
|  _  | |_| | |_) |   \\ V / (_| | | |_| | (_| | |_| | (_) | | | |
|_| |_|____/|____/     \\_/ \\__,_|_|\\__,_|\\__,_|\\__|_|\\___/|_| |_|
 / ___|__ _| | ___ _   _| | __ _| |_ ___  _ __                   
| |   / _` | |/ __| | | | |/ _` | __/ _ \\| '__|                  
| |__| (_| | | (__| |_| | | (_| | || (_) | |                     
 \\____\\__,_|_|\\___|\\__,_|_|\\__,_|\\__\\___/|_|                     
        """
        self.print_colored(banner, 'cyan', 'bright')

    def show_main_menu(self):
        self.display_banner()
        print("\n" + "=" * 50)
        self.print_colored("                MAIN MENU", 'yellow', 'bright')
        print("=" * 50)
        self.print_colored("1. Calculate HDB Price", 'green')
        self.print_colored("2. View Results History", 'blue')
        self.print_colored("3. Exit", 'red')
        print("=" * 50)

    def load_and_train_model(self):
        self.print_colored("Loading data and training polynomial model...",
                           'yellow')
        self.model.load_data('sample_data.csv')
        if self.model.df is not None:
            cleaned_data = self.processor.clean_data(self.model.df)
            self.model.df = cleaned_data
        self.model.train_model()
        metrics = self.model.get_model_metrics()
        polynomial_info = self.model.get_polynomial_equation_info()

        print("\n" + "=" * 60)
        self.print_colored("              MODEL TRAINING SUMMARY", 'green',
                           'bright')
        print("=" * 60)
        print("Polynomial Degree:     {}".format(polynomial_info['degree']))
        print("Training Samples:      {:,}".format(metrics['n_samples']))
        print("Original Features:     {}".format(metrics['n_features']))
        print("Polynomial Features:   {:,}".format(
            polynomial_info['n_features']))
        print("Training R²:           {:.4f}".format(metrics['train_r2']))
        print("Testing R²:            {:.4f}".format(metrics['test_r2']))
        print("Accuracy %:            {:.2f}%".format(metrics['test_r2'] *
                                                      100))
        print("=" * 60)

        import time
        time.sleep(3)
        self.clear_screen()

        self.print_colored("Model trained successfully!", 'green', 'bright')
        self.print_colored("\n🎉 Ready to use! Starting interface...", 'cyan')

        print("")
        self.print_rainbow("souritra coded this")
        self.print_colored("hopefully this one doesn't crash :)", 'magenta',
                           'bright')

        time.sleep(2.5)
        self.clear_screen()

    def predict_price(self):
        if not self.model.is_trained:
            self.print_colored(
                "Error: Model not trained. Please train the model first.",
                'red')
            return
        self.print_colored("\n🏠 Calculate HDB Price", 'cyan', 'bright')
        inputs = self.collect_user_inputs()
        if inputs:
            is_valid, errors = self.processor.validate_input_data(inputs)
            if not is_valid:
                for error in errors:
                    self.print_colored("Error: {}".format(error), 'red')
                return
            prediction, contributions = self.model.predict_price(inputs)
            self.clear_screen()
            self.display_prediction_results(inputs, prediction, contributions)

            self.visualizer.generate_prediction_summary_visuals(
                self.model, inputs, prediction, contributions)

            self.session_predictions.append({
                'inputs': inputs.copy(),
                'prediction': prediction,
                'timestamp': pd.Timestamp.now()
            })

            input("\nPress Enter to continue to main menu...")

    def collect_user_inputs(self):
        inputs = {}
        towns = self.model.get_available_towns()
        if not towns:
            self.print_colored("Error: No towns available", 'red')
            return None

        print("\nTowns ({} available):".format(len(towns)))
        for i, town in enumerate(towns, 1):
            print("  {}. {}".format(i, town))
        self.print_colored(
            "\n💡 You can type the town number (1-{}) or town name".format(
                len(towns)), 'yellow')

        while True:
            town_input = input("Select town: ").strip()
            if town_input.isdigit():
                town_idx = int(town_input) - 1
                if 0 <= town_idx < len(towns):
                    inputs['town'] = towns[town_idx]
                    break
                else:
                    self.print_colored(
                        "❌ Invalid town number. Please select 1-{}".format(
                            len(towns)), 'red')
            else:
                town_upper = town_input.upper()
                if town_upper in towns:
                    inputs['town'] = town_upper
                    break
                else:
                    self.print_colored(
                        "❌ Invalid town name. Please check the list above.",
                        'red')

        flat_types = self.model.get_available_flat_types()
        print("Flat Types:")
        for i, flat_type in enumerate(flat_types, 1):
            print("  {}. {}".format(i, flat_type))

        while True:
            flat_input = input("Select flat type [{}]: ".format(
                "/".join([str(i) for i in range(1, len(flat_types) + 1)]))).strip()
            if flat_input.isdigit():
                flat_idx = int(flat_input) - 1
                if 0 <= flat_idx < len(flat_types):
                    inputs['flat_type'] = flat_types[flat_idx]
                    break
            self.print_colored(
                "❌ Invalid selection. Please select 1-{}".format(
                    len(flat_types)), 'red')

        self.print_colored("💡 Typical ranges: 2-room (34-64 sqm), 3-room (60-90 sqm), 4-room (80-120 sqm)", 'yellow')
        while True:
            try:
                area = float(input("Enter floor area (sqm): "))
                if 30 <= area <= 300:
                    inputs['floor_area_sqm'] = area
                    break
                else:
                    self.print_colored(
                        "❌ Floor area must be between 30-300 sqm", 'red')
            except ValueError:
                self.print_colored("❌ Please enter a valid number", 'red')

        storey_ranges = self.model.get_available_storey_ranges()
        print("Storey Ranges:")
        for i, storey in enumerate(storey_ranges, 1):
            print("  {}. {}".format(i, storey))

        while True:
            storey_input = input("Select storey range [{}]: ".format(
                "/".join([str(i) for i in range(1, len(storey_ranges) + 1)]))).strip()
            if storey_input.isdigit():
                storey_idx = int(storey_input) - 1
                if 0 <= storey_idx < len(storey_ranges):
                    inputs['storey_range'] = storey_ranges[storey_idx]
                    break
            self.print_colored(
                "❌ Invalid selection. Please select 1-{}".format(
                    len(storey_ranges)), 'red')

        flat_models = self.model.get_available_flat_models()
        
        # Simplify flat models display to show only common ones
        common_models = ['MODEL A', 'IMPROVED', 'NEW GENERATION', 'PREMIUM APARTMENT', 'STANDARD', 'APARTMENT']
        display_models = []
        for model in common_models:
            if model in flat_models:
                display_models.append(model)
        
        # Add any remaining models not in common list
        for model in flat_models:
            if model not in display_models:
                display_models.append(model)
        
        print("Flat Models:")
        for i, model in enumerate(display_models[:6], 1):  # Show only first 6 for cleaner display
            display_name = model.replace('MODEL A', 'Model A').replace('NEW GENERATION', 'New Generation').replace('PREMIUM APARTMENT', 'Premium Apartment')
            print("  {}. {}".format(i, display_name))

        while True:
            model_input = input("Select flat model [{}]: ".format(
                "/".join([str(i) for i in range(1, min(6, len(display_models)) + 1)]))).strip()
            if model_input.isdigit():
                model_idx = int(model_input) - 1
                if 0 <= model_idx < min(6, len(display_models)):
                    inputs['flat_model'] = display_models[model_idx]
                    break
            self.print_colored(
                "❌ Invalid selection. Please select 1-{}".format(
                    min(6, len(display_models))), 'red')

        self.print_colored("💡 Most HDB flats have 50-90 years remaining lease", 'yellow')
        while True:
            try:
                remaining_lease = int(input("Enter remaining lease (years): "))
                if 40 <= remaining_lease <= 99:
                    inputs['remaining_lease'] = remaining_lease
                    break
                else:
                    self.print_colored(
                        "❌ Remaining lease must be between 40-99 years", 'red')
            except ValueError:
                self.print_colored("❌ Please enter a valid number", 'red')

        return inputs

    def display_prediction_results(self, inputs, prediction, contributions):
        print("\n" + "=" * 60)
        self.print_colored("              PREDICTION RESULTS", 'green',
                           'bright')
        print("=" * 60)
        self.print_colored(
            "Predicted HDB Price: SGD ${:,.2f}".format(prediction), 'cyan',
            'bright')
        print("=" * 60)
        self.print_colored("Input Summary:", 'yellow')
        print("-" * 30)
        for key, value in inputs.items():
            display_key = key.replace('_', ' ').title()
            print("{:<20}: {}".format(display_key, value))
        print("=" * 60)

    def view_history(self):
        if not self.session_predictions:
            self.print_colored("⚠️ No previous predictions available",
                               'yellow')
            return

        print("\n" + "=" * 80)
        self.print_colored("                    SESSION PREDICTIONS HISTORY",
                           'green', 'bright')
        print("=" * 80)
        print("{:<20} {:<15} {:<10} {:<12} {:<15}".format(
            'Timestamp', 'Town', 'Flat Type', 'Area (sqm)', 'Price (SGD)'))
        print("-" * 80)

        for record in self.session_predictions:
            timestamp = str(record['timestamp'])[:19]
            town = record['inputs'].get('town', '')[:14]
            flat_type = record['inputs'].get('flat_type', '')[:9]
            area = str(record['inputs'].get('floor_area_sqm', ''))
            price = "${:,.0f}".format(record['prediction'])
            print("{:<20} {:<15} {:<10} {:<12} {:<15}".format(
                timestamp, town, flat_type, area, price))
        print("=" * 80)
        input("\nPress Enter to continue to main menu...")

    def run(self):
        self.clear_screen()
        self.display_banner()
        self.load_and_train_model()

        self.print_colored("\n🎉 Ready to use! Starting interface...", 'green')

        while True:
            self.clear_screen()
            self.show_main_menu()
            choice = input("Select an option (1/2/3): ")
            if choice == "1":
                self.clear_screen()
                self.predict_price()
            elif choice == "2":
                self.clear_screen()
                self.view_history()
            elif choice == "3":
                self.print_colored(
                    "\n👋 Thank you for using HDB Polynomial Price Calculator!",
                    'green', 'bright')
                break
            else:
                self.print_colored(
                    "❌ Invalid choice. Please select 1, 2, or 3.", 'red')
                input("Press Enter to continue...")
