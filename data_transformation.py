"""All necessary data transformation functions for preprocessing input data"""

import pandas as pd
import math

def min_max_scaler(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the array data into values between 0 and 1.

    Args:
        data: Panda DataFrame object storing numerical data.
    
    Returns:
        Returns a new Panda DataFrame object, with all data normalized to between 1 and 0.
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

def split(data: pd.DataFrame, percent_split: float) -> tuple[pd.Dataframe, pd.DataFrame]:
    """
    Splits the array data into a new training DataFrame and a new test DataFrame.

    Args:
        data: Panda DataFrame object storing numerical data.
        split: A float storing the decimal percentage of the desired training split.
    
    Returns:
        Returns a tuple storing the two new DataFrames made, the first being the training and the second being test.
    """
    num_rows = data.shape[0]
    row_split = math.floor(num_rows*percent_split)

    return (data[0:row_split], data[row_split:])