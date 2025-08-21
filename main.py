# SOURITRA SAMANTA (3C)

import sys
import os
from cli_interface import SimplifiedHDBCalculatorCLI

def main():
    try:
        import pandas, sklearn, numpy
        print("[SUCCESS] All packages available")
    except ImportError as e:
        print("[ERROR] Missing package: {}".format(e)) # souritra (watermark)
        return
    
    if not os.path.exists('sample_data.csv'):
        print("[ERROR] Could you download sample_data.csv first bro...")
        return
    
    calculator = SimplifiedHDBCalculatorCLI()
    calculator.run()

if __name__ == "__main__":
    main()

# SOURITRA SAMANTA (3C)
