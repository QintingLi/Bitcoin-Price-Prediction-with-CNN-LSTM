# Time Series Forecasting of Bitcoin Prices Using Hybrid CNN-LSTM Neural Networks

This project focuses on forecasting Bitcoin prices using a hybrid neural network model that combines Convolutional Neural Networks (CNN) and Long Short-Term Memory Networks (LSTM). The goal is to leverage both the spatial feature extraction capabilities of CNNs and the temporal sequence learning ability of LSTMs to predict future Bitcoin prices based on historical data.

## Table of Contents
- [Background](#background)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Background

Time series forecasting is a crucial task in financial markets, where predicting the future prices of assets like Bitcoin can help investors make informed decisions. Traditional statistical methods may not capture complex patterns in financial data. Therefore, this project utilizes a hybrid CNN-LSTM model to enhance prediction accuracy by combining the strengths of both CNN and LSTM networks.

## Dataset

The dataset used in this project is historical Bitcoin price data (`BTC-USD.csv`), which contains the following columns:

- `Date`: The date of the record.
- `Open`: The opening price of Bitcoin on that day.
- `High`: The highest price of Bitcoin on that day.
- `Low`: The lowest price of Bitcoin on that day.
- `Close`: The closing price of Bitcoin on that day.
- `Adj Close`: The adjusted closing price for the day.
- `Volume`: The trading volume of Bitcoin on that day.

The dataset is loaded from Google Drive using Google Colab.

## Methodology

1. **Data Preprocessing**: 
   - Load the Bitcoin price data from a CSV file.
   - Extract the 'Close' prices for modeling.
   - Normalize the data using MinMaxScaler for improved model training.

2. **Data Preparation**:
   - Split the data into training (80%) and testing (20%) sets.
   - Create sequences of 5 days (look-back period) for LSTM input.

3. **Model Architecture**:
   - **CNN Layer**: A 1D Convolutional layer to extract spatial features from time-series data.
   - **MaxPooling Layer**: To reduce the spatial dimensions of the feature maps.
   - **Flatten Layer**: To prepare the data for fully connected layers.
   - **Dense Layers**: Fully connected layers with ReLU activation for learning complex relationships.
   - **Output Layer**: A Dense layer with a single neuron to predict the next day's Bitcoin price.

4. **Model Training**:
   - The model is compiled with Adam optimizer and Mean Squared Error (MSE) loss function.
   - Early stopping is implemented to prevent overfitting.

5. **Evaluation and Visualization**:
   - The model's performance is evaluated using Mean Squared Error (MSE).
   - Predictions are made on both training and testing data, and the results are visualized.

## Requirements

To run this project, you need the following libraries:

- Python 3.x
- TensorFlow
- Pandas
- Numpy
- Scikit-learn
- Matplotlib

You can install the required libraries using pip:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/bitcoin-price-forecasting.git
   ```
   
2. **Open the notebook in Google Colab**:
   - Upload `Time_Series_Forecasting_of_Bitcoin_Prices_Using_Hybrid_CNN_LSTM_Neural_Networks.ipynb` to Google Colab.
   
3. **Mount Google Drive**:
   - Run the code block to mount Google Drive where your dataset (`BTC-USD.csv`) is stored.

4. **Run the Notebook**:
   - Execute the cells in the notebook sequentially to train the model and visualize the results.

## Results

- The model's predictions are plotted against the actual Bitcoin prices, showing both training and test predictions.
- The results demonstrate the effectiveness of the CNN-LSTM hybrid model in capturing both spatial and temporal patterns in the data.

