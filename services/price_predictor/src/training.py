import os
import pickle
from argparse import ArgumentParser
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from comet_ml import Experiment
from loguru import logger as logging
from sklearn.metrics import mean_absolute_error

from src.baseline_model import BaselineModel
from src.data_preprocessing import create_target_metric, interpolate_missing_candles
from src.feature_engineering import FeatureEngineering
from src.model_factory import fit_xgboost_regressor
from src.utils import get_model_name
from tools.ohlc_data_reader import OHLCDataReader


def train(
    feature_view_name: str,
    feature_view_version: int,
    ohlc_window_sec: int,
    product_id: str,
    last_n_days_to_fetch_from_store: int,
    last_n_days_to_test_model: int,
    prediction_window_sec: int,
    hyper_param_search_trials: Optional[int] = 0,
) -> None:
    """
    Trains model by following these steps:
    1.  Fetch OHLC data from the feature store
    2. Split data into training and training sets
    3. Preprocess data. In this case, missing values imputation
    4. Create the target metric as a new column in the dataset. This will be the metric to be predicted
    5. Train the model

    Args:
        feature_view_name (str): Name of the feature view
        feature_view_version (int): Version of the feature view
        ohlc_window_sec (int): Window in seconds to fetch OHLC data
        product_id (str): Product ID
        last_n_days_to_fetch_from_store (int): Number of days to fetch data from the feature store
        last_n_days_to_test_model (int): Number of days to use for testing the model
        prediction_window_sec (int): Size of the prediction window in seconds
        hyper_param_search_trials (int, optional): Number of hyperparameter search trials. Defaults to 0.

    Returns:
        None
        The model artifacts are saved to the model registry
    """
    # Create an experiment to log metadata to Comet_ML platform
    experiment = Experiment(
        api_key=os.environ['COMET_ML_API_KEY'],
        project_name=os.environ['COMET_ML_PROJECT_NAME'],
        workspace=os.environ['COMET_ML_WORKSPACE'],
    )

    # Log all the input parameters to the training function
    experiment.log_parameters(
        {
            'feature_view_name': feature_view_name,
            'feature_view_version': feature_view_version,
            'ohlc_window_sec': ohlc_window_sec,
            'product_id': product_id,
            'last_n_days_to_fetch_from_store': last_n_days_to_fetch_from_store,
            'last_n_days_to_test_model': last_n_days_to_test_model,
            'prediction_window_sec': prediction_window_sec,
            'hyper_param_search_trials': hyper_param_search_trials,
        }
    )

    # Step 1: Fetch OHLC data from the feature store
    ohlc_data_reader = OHLCDataReader(
        feature_view_name=feature_view_name,
        feature_view_version=feature_view_version,
        ohlc_window_sec=ohlc_window_sec,
    )
    logging.info('Fetching OHLC data from the feature store')
    ohlc_data = ohlc_data_reader.read_from_offline_store(
        product_id=product_id, last_n_days=last_n_days_to_fetch_from_store
    )

    # Add a human readable datetime column to the dataset using the timestamp in milliseconds column
    ohlc_data['datetime'] = pd.to_datetime(ohlc_data['timestamp'], unit='ms')

    # Log a dataset hash to track the dataset
    experiment.log_dataset_hash(ohlc_data)

    # Step 2: Split data into training and testing sets
    logging.info('Splitting data into training and testing sets')
    ohlc_train, ohlc_test = split_train_test(
        ohlc_data=ohlc_data, last_n_days_to_test_model=last_n_days_to_test_model
    )

    # Log the number of rows in the training and testing sets
    n_rows_train_original = ohlc_train.shape[0]
    n_rows_test_original = ohlc_test.shape[0]
    experiment.log_metric('n_rows_train', n_rows_train_original)
    experiment.log_metric('n_rows_test', n_rows_test_original)

    # Step 3: Preprocess data. Interpolate missing candlesticks
    logging.info('Interpolating missing candlesticks for training data')
    ohlc_train = interpolate_missing_candles(ohlc_train, ohlc_window_sec)
    logging.info('Interpolating missing candlesticks for testing data')
    ohlc_test = interpolate_missing_candles(ohlc_test, ohlc_window_sec)

    # Log the number of rows interpolated due to missing candlesticks
    n_interpolated_rows_train = ohlc_train.shape[0] - n_rows_train_original
    n_interpolated_rows_test = ohlc_test.shape[0] - n_rows_test_original
    experiment.log_metric('n_interpolated_rows_train', n_interpolated_rows_train)
    experiment.log_metric('n_interpolated_rows_test', n_interpolated_rows_test)

    # Step 4: Create the target metric as a new column in the dataset for training and testing sets
    logging.info('Creating the target metric')
    ohlc_train = create_target_metric(
        ohlc_train, ohlc_window_sec, prediction_window_sec
    )
    ohlc_test = create_target_metric(ohlc_test, ohlc_window_sec, prediction_window_sec)

    # Log a plot of the target metric distribution
    output_dir = './target_metric_histogram.png'
    plot_target_metric_histogram(ohlc_train['target'], output_dir, n_bins=100)

    # Log the target metric histogram plot to Comet_ML
    experiment.log_image(output_dir, name='target_metric_histogram')

    # Split the features and target
    X_train = ohlc_train.drop(columns=['target'])
    y_train = ohlc_train['target']
    X_test = ohlc_test.drop(columns=['target'])
    y_test = ohlc_test['target']

    # Check for NaN values
    nan_summary_train = pd.DataFrame(
        {
            'nan_count': np.sum(X_train.isna()),
            'nan_percentage': np.sum(X_train.isna()) / X_train.shape[0] * 100,
        }
    )

    nan_summary_test = pd.DataFrame(
        {
            'nan_count': np.sum(X_test.isna()),
            'nan_percentage': np.sum(X_test.isna()) / X_test.shape[0] * 100,
        }
    )

    logging.info(f'NaN summary for training data:\n{nan_summary_train}')
    logging.info(f'NaN summary for testing data:\n{nan_summary_test}')

    # Step 5: Train the model Using the BaselineModel class
    logging.info('Training the model')
    model = BaselineModel(
        n_candles_into_the_future=prediction_window_sec // ohlc_window_sec
    )
    y_test_predictions = model.predict(X_test)
    baseline_test_mae = evaluate_model(
        predictions=y_test_predictions,
        actuals=y_test,
        description='Baseline model on Test data',
    )
    y_train_predictions = model.predict(X_train)
    baseline_train_mae = evaluate_model(
        predictions=y_train_predictions,
        actuals=y_train,
        description='Baseline model on Training data',
    )
    # Log the model evaluation metrics of both the training and testing sets to Comet_ML
    experiment.log_metric('baseline_train_mae', baseline_train_mae)
    experiment.log_metric('baseline_test_mae', baseline_test_mae)

    # Step 6: Build a more sophisticated model
    # Add more features to the dataset
    feature_eng_pipeline = FeatureEngineering(
        n_candles_into_future=prediction_window_sec // ohlc_window_sec
    )
    X_train = feature_eng_pipeline.fit_transform(X_train)
    X_test = feature_eng_pipeline.transform(X_test)

    # Keep only observations for which we have all the features for the training and testing sets
    nan_mask_train = X_train.isna().any(axis=1)
    X_train = X_train[~nan_mask_train]
    y_train = y_train[~nan_mask_train]

    nan_mask_test = X_test.isna().any(axis=1)
    X_test = X_test[~nan_mask_test]
    y_test = y_test[~nan_mask_test]

    # log the shapes of X_train, y_train, X_test, y_test
    experiment.log_metric('X_train_shape', X_train.shape)
    experiment.log_metric('y_train_shape', y_train.shape)
    experiment.log_metric('X_test_shape', X_test.shape)
    experiment.log_metric('y_test_shape', y_test.shape)

    # Log the list of features
    experiment.log_parameter('features_to_use', X_train.columns.tolist())

    # # Train a lasso regression model
    # model = fit_lasso_regressor(X_train, y_train, hyper_param_search_trials=hyper_param_search_trials)
    # test_mae = evaluate_model(
    #     predictions=model.predict(X_test),
    #     actuals=y_test,
    #     description='Lasso Regression model on Test data',
    # )
    # train_mae = evaluate_model(
    #     predictions=model.predict(X_train),
    #     actuals=y_train,
    #     description='Lasso Regression model on Training data',
    # )

    # # Log the MAE of the Lasso Regression model on the training and testing sets
    # experiment.log_metric('lasso_model_test_mae', test_mae)
    # experiment.log_metric('lasso_model_train_mae', train_mae)

    # # Save the model as a pickle file
    # with open('lasso_model.pkl', 'wb') as f:
    #     logging.info('Saving the model as a pickle file')
    #     pickle.dump(model, f)

    # model_name = get_model_name(product_id=product_id)
    # experiment.log_model(name=model_name, file_or_folder='./lasso_model.pkl')

    # Train an XGBoost model
    model = fit_xgboost_regressor(
        X_train, y_train, hyper_param_search_trials=hyper_param_search_trials
    )
    test_mae = evaluate_model(
        predictions=model.predict(X_test),
        actuals=y_test,
        description='XGBoost model on Test data',
    )
    train_mae = evaluate_model(
        predictions=model.predict(X_train),
        actuals=y_train,
        description='XGBoost model on Training data',
    )

    with open('./xgboost_model.pkl', 'wb') as f:
        logging.info('Saving the model as a pickle file')
        pickle.dump(model, f)

    # Log the MAE of the XGBoost model on the training and testing sets
    experiment.log_metric('xgboost_model_test_mae', test_mae)
    experiment.log_metric('xgboost_model_train_mae', train_mae)

    model_name = get_model_name(product_id=product_id)
    experiment.log_model(name=model_name, file_or_folder='./xgboost_model.pkl')

    # Last step in the training pipeline: Push the model to the model registry
    if test_mae < baseline_test_mae:
        # Push the model to the model registry
        experiment.register_model(model_name=model_name)
        logging.info(
            '***** The XGBoost model performs better than the baseline model *****'
        )
        logging.info('***** Pushing the XGBoost model to the model registry *****')
        experiment.register_model(model_name=model_name)

    else:
        logging.info(
            '***** The baseline model performs better than the XGBoost model *****'
        )
        logging.info('***** Not pushing the XGBoost model to the model registry *****')


def split_train_test(
    ohlc_data: pd.DataFrame,
    last_n_days_to_test_model: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the OHLC data into training and testing sets

    Args:
        ohlc_data (pd.DataFrame): OHLC data
        last_n_days_to_test_model (int): Number of days to use for testing the model

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing sets
    """
    # Calculate the cutoff date for splitting the data
    cutoff_date = ohlc_data['datetime'].max() - pd.Timedelta(
        days=last_n_days_to_test_model
    )

    # Split the data into training and testing sets
    ohlc_train = ohlc_data[ohlc_data['datetime'] < cutoff_date]
    ohlc_test = ohlc_data[ohlc_data['datetime'] >= cutoff_date]

    return ohlc_train, ohlc_test


def plot_target_metric_histogram(target: pd.Series, output_dir: str, n_bins: int = 30):
    """
    Plots the histogram of the target metric

    Args:
        target (pd.Series): Target metric
        output_dir (str): Output directory for the plot
        n_bins (int, optional): Number of bins in the histogram. Defaults to 30.
    """
    # Create a histogram of the contnuous variable ohlc_train["target"] and save it to an object
    plt.figure(figsize=(10, 6))
    plt.hist(target, bins=30, alpha=0.75, color='blue', edgecolor='black')
    plt.title('Histogram of Price Change')
    plt.xlabel('Price Change')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(output_dir, format='png')


def evaluate_model(
    predictions: pd.Series,
    actuals: pd.Series,
    description: Optional[str] = 'Model evaluation',
):
    """
    Evaluates the model using the mean absolute error metric

    Args:
        predictions (pd.Series): Predictions
        actuals (pd.Series): Actual values
        description (str, optional): Description of the evaluation. Defaults to "Model evaluation".

    Returns:
        None
    """
    logging.info('**********' + description + '**********')
    mae = mean_absolute_error(actuals, predictions)
    logging.info(f'Mean Absolute Error: {mae:.4f}')

    return mae


if __name__ == '__main__':
    # Add a CLI (Command Line Interface) argument called hyper_param_search_trials using argparse
    parser = ArgumentParser()
    parser.add_argument(
        '--hyper_param_search_trials',
        type=int,
        default=0,
        help='Number of hyperparameter search trials',
    )
    args = parser.parse_args()

    train(
        feature_view_name='ohlc_feature_view',
        feature_view_version=1,
        ohlc_window_sec=60,
        product_id='BTC/USD',
        last_n_days_to_fetch_from_store=120,
        last_n_days_to_test_model=7,
        prediction_window_sec=60 * 5,
        hyper_param_search_trials=args.hyper_param_search_trials,
    )
