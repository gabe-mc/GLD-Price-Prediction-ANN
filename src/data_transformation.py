"""All necessary data transformation functions for preprocessing input data"""

import pandas as pd
import math

import src.constants as constants

def normalize(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the array data into values between 0 and 1.

    Args:
        data: Panda DataFrame object storing numerical data.

    Returns:
        A new Panda DataFrame object, with all data normalized to between 1 and 0.
    """
    result = pd.DataFrame(columns=data.columns)
    for column in data:
        loop_index = 0
        for value in data[column].values:
            loop_index += 1
            column_max = data[column].max()
            column_min = data[column].min()
            new_value = (float(value) - float(column_min)) / \
                (float(column_max) - float(column_min))
            result.at[loop_index, column] = new_value

    return result

def denormalize(normalized_value: float)-> float:
    """
    Reverses the normalization process to give the single_entry parameter in the same scale as it was given in the 
    origial real-world data.

    Args:
        train_max: The maximum value within the test features or target dataframe which was used in normalization.
        train_min: The minmum value within the test features or target dataframe which was used in normalization.
        normalized_value: A float which is the result of our ANN predictor, which needs to be rescaled up (denormalized).

    Returns:
        A denormalized float, which is the predicted gold price.
    """
    return normalized_value * (constants.TRAIN_MAX - constants.TRAIN_MIN) + constants.TRAIN_MIN

def normalize_inputs(WPM_prev, WPM, silver_prev, silver, palladium, oil, treasury_bill, month, GLD_prev)-> list:
    """
    Normalizes a single value according to the input max and min. Meant to be used when passing new values into the predictor.
    
    Args:
        WPM_prev: The previous previous trading day's closing price for Wheaton Precious Metals Corp. ($WPM).
        WPM: The previous trading day's closing price for $WPM.
        silver_prev: The previous previous trading day's closing spot price for Silver.
        silver: The previous trading day's closing spot price for Silver.
        palladium: silver: The previous trading day's closing spot price for Palladium.
        oil: The previous trading day's closing Crude Oil spot price
        treasury_bill: The previous trading day's closing return on 4 Week US Treasury Bills .
        month: The Current month.
        GLD_prev: The previous closing price of $GLD.

    Returns:
        A list of the input values, normalized.
    """

    return [
        (WPM_prev - constants.WPM_PREV_MIN) / (constants.WPM_PREV_MAX - constants.WPM_PREV_MIN),
        (WPM - constants.WPM_MIN) / (constants.WPM_MAX - constants.WPM_MIN),
        (silver_prev - constants.SILVER_PREV_MIN) / (constants.SILVER_PREV_MAX - constants.SILVER_PREV_MIN),
        (silver - constants.SILVER_MIN) / (constants.SILVER_MAX - constants.SILVER_MIN),
        (palladium - constants.PALLADIUM_MIN) / (constants.PALLADIUM_MAX - constants.PALLADIUM_MIN),
        (oil - constants.OIL_MIN) / (constants.OIL_MAX - constants.OIL_MIN),
        (treasury_bill - constants.TREASURY_MIN) / (constants.TREASURY_MAX - constants.TREASURY_MIN),
        (month - constants.MONTH_MIN) / (constants.MONTH_MAX - constants.MONTH_MIN),
        (GLD_prev - constants.GLD_LTD_MIN) / (constants.GLD_LTD_MAX - constants.GLD_LTD_MIN),
        ]

def split(data: pd.DataFrame, percent_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the array data into a new training DataFrame and a new test DataFrame.

    Args:
        data: Panda DataFrame object storing numerical data.
        split: A float storing the decimal percentage of the desired training split.

    Returns:
        A tuple storing the two new DataFrames made, the first being the training and the second being test.
    """
    num_rows = data.shape[0]
    row_split = math.floor(num_rows*percent_split)

    return (data[0:row_split], data[row_split:])
