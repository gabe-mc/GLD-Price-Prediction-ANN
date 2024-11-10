"""Runs the ANN model to predict prices based on input vales."""

import torch

from src.model import GLDPredictor
from data_transformation import denormalize, normalize_inputs


def predict(WPM_prev, WPM, silver_prev, silver, palladium, oil, treasury_bill, month, GLD_prev)-> float:
    """
    Uses the GLDPredictor ANN model to predict the closing price of the SPDR Gold Trust ETF ($GLD).
    
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
        The predicted next-trading-day price of $GLD in USD.
    """
    # Instantiate model
    model = GLDPredictor()
    model.load_state_dict(torch.load("models/GLDPredictor_model.plt", weights_only=False))
    model.eval()

    # Creating an input tensor
    input_tensor = torch.tensor(normalize_inputs(WPM_prev, WPM, silver_prev, silver, palladium, oil, treasury_bill, month, GLD_prev), dtype=torch.double)
    print(input_tensor.dtype)
    # Predict
    with torch.no_grad():
        prediction = model(input_tensor)
    
    return denormalize(prediction.item())

print("Predicted GLD spot price: $" + str(predict(50.06, 50, 22.17, 22.1, 1002.5, 71.71, 5.31, 12, 192.34)) + " ...it should have been $190.99")
