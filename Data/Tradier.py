import requests
import numpy as np
import matplotlib.pyplot as plt

def get_tradier_data():
    """
    Returns Tradier Implied Volatility
    """

    response = requests.get(

        'https://api.tradier.com/v1/markets/options/expirations',

        params = {
            'symbol': 'SPY', 
            'includeAllRoots': 'true', 
            'strikes': 'false',
            },

        headers = {
            'Authorization': 'Bearer GQZRHrCRsbRqE43Z0k6rKcDj73oU', 
            'Accept': 'application/json',
            }

    )

    json_response = response.json()
    expiry = json_response["expirations"]["date"]

    response = requests.get(
        
        'https://api.tradier.com/v1/markets/options/chains',

        params = {
            'symbol': 'SPY', 
            'expiration': expiry[7], 
            'greeks': 'true',
            },

        headers = {
            'Authorization': 'Bearer GQZRHrCRsbRqE43Z0k6rKcDj73oU', 
            'Accept': 'application/json',
            }

    )

    json_response = response.json()
    data = json_response["options"]["option"]

    midpoint = []
    strikes = []
    ivol = []
    delta = []

    for d in data:

        K = d["strike"]

        if K % 5 == 0:

            if d["option_type"] == "call":

                bid = d["bid"]
                ask = d["ask"]
                mid = (bid + ask) / 2.0
                midpoint.append(mid)

                strikes.append(d["strike"])
                ivol.append(d["greeks"]["mid_iv"])
                delta.append(d["greeks"]["delta"])

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle("Market Data")

    ax1.plot(strikes, ivol)
    ax2.plot(strikes, midpoint)
    plt.show()

    return np.array(midpoint).reshape(-1, 1), np.array(ivol)