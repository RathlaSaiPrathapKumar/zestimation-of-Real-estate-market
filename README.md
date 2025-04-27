# Real Estate Market Analysis and Prediction System

## Overview
This project is a comprehensive real estate market analysis and prediction system that helps analyze and predict property counts using machine learning techniques. The system provides a user-friendly GUI interface for data analysis, visualization, and prediction.

## Features
- **Data Upload and Preprocessing**
  - CSV file upload functionality
  - Data cleaning and preprocessing
  - Handling missing values and duplicates
  - Label encoding for categorical variables

- **Exploratory Data Analysis (EDA)**
  - KDE plots for price distribution
  - Histograms for numerical features
  - Correlation heatmaps
  - Violin plots for price distribution
  - Line plots for time series analysis
  - Density plots
  - Categorical analysis
  - Geographic data visualization

- **Machine Learning Models**
  - K-Nearest Neighbors (KNN) Regressor
  - Extra Trees Regressor
  - Model persistence (saving/loading trained models)

- **Model Evaluation**
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²) score
  - Visual comparison of model performance

## Requirements
- Python 3.x
- Required Python packages:
  ```
  tkinter
  numpy
  pandas
  matplotlib
  seaborn
  scikit-learn
  joblib
  ```

## Installation
1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main application:
   ```bash
   python Main.py
   ```

2. Using the GUI:
   - Click "Upload Dataset" to load your CSV file
   - Use "Preprocessing" to clean and prepare the data
   - Click "EDA" to explore data visualizations
   - Use "Data Splitting" to split the dataset
   - Choose between "KNN Regression" or "Extra Trees Regression" for prediction
   - Use "Predict" to make predictions on new data

## Project Structure
- `Main.py`: Main application file containing the GUI and all functionality
- `Dataset/`: Directory for storing input data files
- Model files:
  - `KNeighborsRegressor.pkl`: Saved KNN model
  - `ExtraTrees_model.pkl`: Saved Extra Trees model

## Data Requirements
The input CSV file should contain the following columns:
- Regionname
- Price
- Distance
- Type
- Lattitude
- Longtitude
- Propertycount
- Date
- And other relevant real estate features

## Model Performance
The system provides comprehensive metrics to evaluate model performance:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²) score

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Built using Python's scientific computing stack
- Uses scikit-learn for machine learning capabilities
- Implements matplotlib and seaborn for visualization 