import matplotlib.pyplot as plt
from vollib.black_scholes_merton import bsm_call

def get_risk_neutral_price(market_prices, model_volatility, stock_price, strike_prices, dte):
    """
    Returns the Risk Neutral Price
    """

    # Set Black Scholes Parameters
    S = stock_price
    t = dte

    model_prices = []

    # Iterate Thru Volatility Array
    index = 0
    for ivol in model_volatility:    

        K = strike_prices[index]
        q = .05
        r = 0.1
        sigma = ivol

        call_opt = bsm_call(S, K, q, t, r, sigma)
        model_prices.append(call_opt)

        index+=1

    return 0