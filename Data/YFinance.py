import numpy as np
import yahoo_fin.options as ops
import matplotlib.pyplot as plt

ticker = "QQQ"

expiration_dates = ops.get_expiration_dates(ticker)
print(expiration_dates)

puts = ops.get_puts(ticker, expiration_dates[5]) 
print(puts.head())

calls = ops.get_calls(ticker, expiration_dates[5]) 
print(calls.head())

strike = []
ivol = []
midpoint = []

for i in range(len(puts)):
    strike.append(puts['Strike'][i])
    ivol.append(float(puts['Implied Volatility'][i].strip('%')))
    midpoint.append((puts["Bid"][i] + puts["Ask"][i]) / 2.0)

print(ivol)

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle("Volatility Curve and Pricing")

ax1.plot(strike, ivol, linestyle = "dashdot", label = "Market Volatility")
ax1.legend(loc = "best")

ax2.plot(strike, midpoint, linestyle = "dashdot", label = "Market Price")
ax2.legend(loc = "best")

plt.show()