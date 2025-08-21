# SOURITRA SAMANTA (3C)

import sys
import os
from cli_interface import SimplifiedHDBCalculatorCLI

# This is the entry-point of the entire project
def main():
    try: # Checks if packages are available
        import pandas, sklearn, numpy
        print("[SUCCESS] All packages available")
    except ImportError as e:
        print("[ERROR] Missing package: {}".format(e)) # souritra (watermark)
        return
    
    if not os.path.exists('sample_data.csv'): # Checks if dataset is available
        print("[ERROR] Could you download sample_data.csv first bro...")
        return
    
    calculator = SimplifiedHDBCalculatorCLI()
    calculator.run()

if __name__ == "__main__": # Runs code
    main()

# SOURITRA SAMANTA (3C)
