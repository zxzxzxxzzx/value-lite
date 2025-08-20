# 🚀 HDB Valuation Calculator (LITE) - Instructions

## Overview
The **HDB Valuation Calculator (LITE)** is designed to run on Python IDLE executor with full Python 3.5 compatibility. It functions similarly to the original version that I made, with additional changes to make it easier to run on the IDLE executor with Python 3.5. 

The original full-feature version of this can be found in https://github.com/zxzxzxxzzx/value
## System Requirements

### Python Version
- **Python 3.5 or higher** (I specifically tested this for _Python 3.5_)
- Compatible with IDLE Python executor

### Required Libraries
The following libraries are automatically installed:
- `pandas`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `seaborn`
- `colorama` 
  
In the event that the library cannot be automatically installed, you may need to run the installation process manually in the shell.

## How to Run on IDLE Executor

### Step 1: Start the Application
1. Open your IDLE Python environment
2. Navigate to the project directory
3. Run the main application:
   ```python
   python main.py
   ```
   OR simply click the "Run" button for `main.py`

### Step 2: Initial Setup (Automatic)
The application will automatically:
1. Load the complete HDB dataset (37,153 records)
2. Clean and preprocess the data
3. Train the 4th degree polynomial regression model
4. Display a model training summary with accuracy metrics shortly after**¹**

**Expected Output:**
```
============================================================
              MODEL TRAINING SUMMARY
============================================================
Polynomial Degree:     4
Training Samples:      36,761
Original Features:     6
Polynomial Features:   209
Training R²:           0.7317
Testing R²:            0.7293
Accuracy %:            72.93%
============================================================
```

After setup, you'll see:
```
==================================================
                MAIN MENU
==================================================
1. Calculate HDB Price
2. View Results History
3. Exit
==================================================
```

## File Structure
```
├── main.py                   # Main entry point 
├── cli_interface.py          # CLI Engine
├── hdb_polynomial_model.py   # Machine Learning Model
├── data_processor.py         # Data Cleaning Engine
├── visualizer.py             # Visualisation Engine
├── sample_data.csv           # HDB dataset (37,153 records)
└── /graphs/                  # Generated visualization files
```

## IDLE Executor Specific Features

### Console Clearing
- Uses cross-platform console clearing
- Works with Windows (`cls`) and Unix (`clear`)
- Handling for IDLE environment

### Colored Output
- Uses `colorama` library for INSTEAD of `rich` for cross-platform support
- Auto-fallback to non-color if colorama fails
- Works in most environments

### Input Validation
- Input handling for IDLE executor
- Clear error messages and prompts
- Handles invalid inputs better

## Troubleshooting

### Common Issues

**1. Import Errors**
```
ModuleNotFoundError: No module named 'pandas'
```
**Solution**: Libraries should auto-install. If not, manually install with `pip install (module)`.


**2. Data File Not Found**
```
FileNotFoundError: sample_data.csv
```
**Solution**: Ensure `sample_data.csv` is in the root directory.

**3. Colors Not Showing**
- Colors may not display in some IDLE versions
- Application works normally without colors
  
**Solution**: The program will automatically fallback to plain text

**4. Slow Performance**
- First run may be slower due to model training
- Subsequent predictions are faster
- This is mainly because there are 37,153 records to process

## Performance Notes
- **Training Time**: ~10-15 seconds on first run
- **Prediction Time**: <1 second per prediction
- **Memory Usage**: ~50-100MB for complete dataset
- **Accuracy**: ~73% (R² score)**¹**

## Advanced Usage

### Viewing Generated Charts
Charts are automatically saved to `graphs/` folder: **²**
- `prediction_summary_[timestamp].png` - Feature contribution analysis
- `price_comparison_[timestamp].png` - Price comparisons
- `market_analysis_[timestamp].png` - Market trend analysis

### Session Management
- All predictions stored in current session
- View complete history with timestamps
- Session resets on application restart

## Technical Details

### Machine Learning Model
- **Algorithm**: 4th degree polynomial regression
- **Features**: 6 original features → 209 polynomial features
- **Validation**: Train/test split with cross-validation
- **Metrics**: R² score, accuracy percentage  
I will be releasing a notebook with a deconstruction of how I made this soon!
---

**Note**: This application is optimized for educational purposes and local development. For production use, consider additional validation and security measures. 
**Ps**: Hopefully this version doesn't crash 🔥 

¹ `sample_data.csv`  is split into **training** and **testing** data (_80% training & 20% testin_g). In **testing**, the price is hidden and calculated by the model on its own. After which, it is compared with the ACTUAL values from the original dataset. **The R² value is derived from that**. This form of data validation is known as **hold-out cross-validation**

² don't worry, if `/graphs/` doesn't exist, the application will create it on its own in the same directory that the application is in

THIS IS **NOT** THE OFFICIAL USER DOCUMENTATION
> souritra samanta (3C)