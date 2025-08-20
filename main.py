import sys
import os
from cli_interface import SimplifiedHDBCalculatorCLI

def main():
    print("🔄 Starting HDB Polynomial Price Calculator...")
    
    try:
        import pandas, sklearn, numpy
        print("✅ All required packages available")
    except ImportError as e:
        print("❌ Missing required package: {}".format(e))
        return
    
    if not os.path.exists('sample_data.csv'):
        print("⚠️ Main data file not found - please ensure sample_data.csv is available")
        return
    
    calculator = SimplifiedHDBCalculatorCLI()
    calculator.run()

if __name__ == "__main__":
    main()