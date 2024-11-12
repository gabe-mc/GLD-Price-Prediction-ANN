"""Runs the prediction on the current day's data"""

from datetime import datetime

from data_gathering import get_price
from prediction import predict


def main():
    """
    Predicts today's price for GLD based on yesterday's closing price

    Returns:
        A float that is the prediction for today's GLD price.
    """

    WPM_prev = get_price("WPM", True)
    WPM = get_price("WPM")
    silver_prev = get_price("SIUSD", True)
    silver = get_price("SIUSD")
    palladium = get_price("PAUSD")
    oil = get_price("CLUSD")
    treasury_bill = 4.51
    month = int(datetime.now().strftime("%m"))
    GLD_prev = get_price("GLD", True)

    print("Today's GLD Price prediction:", round(predict(WPM_prev, WPM,
          silver_prev, silver, palladium, oil, treasury_bill, month, GLD_prev), 2))


if __name__ == "__main__":
    main()
