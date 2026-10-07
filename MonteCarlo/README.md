# Monte Carlo Stock Price Simulation

## 1. Project Overview

This project explores Monte Carlo simulation for modelling possible future stock prices using **Geometric Brownian Motion (GBM)**.

The project was developed in two stages:

* **V1:** A baseline Monte Carlo simulation using manually specified stock parameters.

* **V2:** An extension using historical **Apple (AAPL)** data to estimate the model parameters from observed market data.

The project focuses on understanding how random sampling, probability, volatility and drift can be combined to model a range of possible future stock-price outcomes.

---

## 2. What is Monte Carlo Simulation?

Monte Carlo simulation is a computational method that uses repeated random sampling to estimate possible outcomes of a process.

Instead of trying to predict one exact future stock price, the simulation generates many possible future price paths.

For example, if we simulate:

```text
10,000 possible future paths
```

we can examine the distribution of the resulting final prices and calculate quantities such as:

* Average simulated price

* Standard deviation

* Percentiles

* Probability of finishing above or below a given price

The larger the number of simulations, the more stable the estimated distribution generally becomes.

---

## 3. Geometric Brownian Motion

The stock-price simulation is based on the Geometric Brownian Motion model:

$$
S_{t+\Delta t}
=
S_t
\exp
\left[
\left(\mu-\frac{1}{2}\sigma^2\right)\Delta t
+
\sigma Z\sqrt{\Delta t}
\right]
$$

where:

* \(S_t\) = current stock price

* \(\mu\) = expected annual return (drift)

* \(\sigma\) = annual volatility

* \(Z\) = random draw from a standard normal distribution

* \(\Delta t\) = time step

The model contains two main components:

### Drift

$$
\left(\mu-\frac{1}{2}\sigma^2\right)\Delta t
$$

This represents the deterministic component of the price movement.

### Random shock

$$
\sigma Z\sqrt{\Delta t}
$$

This introduces randomness into each simulated price movement.

---

## 4. Why Use GBM?

GBM is commonly used as a simple mathematical model for stock prices because it produces:

* Continuous price paths

* Positive stock prices

* Normally distributed log returns

* Lognormally distributed price levels

It is also relatively straightforward to implement using NumPy, making it useful for understanding the mechanics behind Monte Carlo simulation.

However, real stock prices do not perfectly follow GBM, which is discussed later in the limitations section.

---

# 5. V1 — Baseline Monte Carlo Simulation

The first version uses manually specified values for the initial price, drift and volatility.

The simulation uses:

```text
252 trading days

10,000 simulations
```

Each simulation represents one possible one-year stock-price path.

## Creating the Price Matrix

The simulated prices are stored in a NumPy matrix:

```python
price = np.zeros((n_steps + 1, n_simulations))
```

With:

```python
n_steps = 252

n_simulations = 10000
```

the matrix has shape:

```text
(253, 10000)
```

There are 253 rows because the simulation contains:

* 1 initial price

* 252 future trading days

Each column represents one complete simulated price path.

---

# 6. Setting the Initial Price

The initial stock price is placed in the first row:

```python
price[0] = S0
```

This sets the starting point for all simulations.

Therefore, every simulation begins with the same initial price, but the paths diverge as different random shocks are generated.

---

# 7. Generating Random Shocks

At each time step, the simulation generates random values from a standard normal distribution:

```python
z = rng.standard_normal(n_simulations)
```

Since:

```python
n_simulations = 10000
```

this produces:

```text
[z₀, z₁, z₂, z₃, ..., z₉₉₉₉]
```

Each value represents the random shock for one simulation at that particular time step.

For example:

* \(z_0\) → simulation 1

* \(z_1\) → simulation 2

* ...

* \(z_{9999}\) → simulation 10,000

This means that every simulation receives its own random shock at each time step.

If only:

```python
rng.standard_normal()
```

were used, a single random number would be generated. NumPy would then broadcast that same value across all simulations, causing the paths to move identically rather than independently.

---

# 8. Understanding the GBM Formula in Code

The GBM equation is implemented as:

```python
price[t + 1] = price[t] * np.exp(
    (mu - 0.5 * sigma ** 2) * dt
    + sigma * z * np.sqrt(dt)
)
```

This corresponds directly to:

$$
S_{t+\Delta t}
=
S_t
\exp
\left[
\left(\mu-\frac{1}{2}\sigma^2\right)\Delta t
+
\sigma Z\sqrt{\Delta t}
\right]
$$

The different parts of the equation are therefore represented directly in the Python code.

---

# 9. Vectorisation

The simulation uses NumPy vectorisation rather than calculating each simulation individually with a separate inner loop.

At a particular time step:

```text
z = z₀ to z₉₉₉₉
```

Therefore:

```text
σ × z
```

produces:

```text
[σz₀, σz₁, σz₂, ..., σz₉₉₉₉]
```

Each simulation therefore receives its own scaled random shock.

NumPy performs these calculations across the entire array at once rather than requiring a Python loop over all 10,000 simulations.

This makes the simulation substantially more efficient.

---

# 10. How Many Random Numbers Are Generated?

There are:

```text
252 time steps
```

and:

```text
10,000 simulations
```

At every time step, 10,000 random shocks are generated.

Therefore:

$$
252 \times 10,000
=
2,520,000
$$

random numbers are generated in total.

Each random number contributes to one simulation at one particular time step.

---

# 11. Extracting the Final Prices

After all 252 time steps have been simulated:

```python
final_price = price[n_steps]
```

Since:

```python
n_steps = 252
```

this extracts the final row of the matrix.

The result contains the final simulated price from each of the 10,000 simulations.

This array can then be used to analyse the distribution of possible final prices.

---

# 12. Boolean Comparisons and Probability

NumPy can compare every value in an array against a condition:

```python
final_price > 80
```

This produces a Boolean array where:

```text
True = 1
False = 0
```

The proportion of simulations satisfying the condition can then be calculated using `np.mean()`:

```python
np.mean(final_price > 80)
```

This estimates the probability of the final price exceeding 80:

$$
\frac{\text{number of simulations above 80}}{\text{total simulations}}
$$

With 10,000 simulations, this is the number of simulations above 80 divided by 10,000.

This allows Monte Carlo simulations to estimate probabilities directly from the simulated outcomes.

The same approach can be used for different conditions.

For example:

```python
np.mean(final_price > 100)
```

estimates the probability of finishing below 80.

A range can also be examined:

```python
np.mean((final_price > 80) & (final_price < 100))
```

This estimates the proportion of simulations finishing between 80 and 100.

---

# 13. Standard Deviation

The standard deviation of the simulated final prices can be calculated using:

```python
np.std(final_price)
```

Standard deviation measures the spread of the simulated final prices around their mean.

A larger standard deviation indicates that the simulated outcomes are more widely dispersed.

A smaller standard deviation indicates that the simulated outcomes are more concentrated around the mean.

---

# 14. Percentiles

Percentiles can be calculated using:

```python
np.percentile(final_price, 5)
```

This gives the price below which approximately 5% of simulated outcomes fall.

Similarly:

```python
np.percentile(final_price, 50)
```

gives the median simulated final price.

And:

```python
np.percentile(final_price, 95)
```

gives the price below which approximately 95% of simulated outcomes fall.

Percentiles are useful for understanding the distribution without relying only on the mean.

---

# 15. Visualising the Simulations

The simulated price paths can be visualised using Matplotlib.

Plotting the paths makes it easier to see how the simulations evolve over time.

A typical plot contains:

* Time on the x-axis

* Stock price on the y-axis

* One line for each simulated path

The resulting graph shows a large collection of possible future price trajectories.

---

# 16. Why the Paths Spread Out

All simulations start from the same initial price:

```python
price[0] = S0
```

However, each simulation receives different random shocks.

At every time step, these differences accumulate.

Therefore, the paths gradually diverge from one another.

This produces the characteristic spreading pattern seen in Monte Carlo stock-price simulations.

Higher volatility generally produces greater dispersion between the paths.

---

# 17. V2 — Historical AAPL Data

The second version extends the model by using historical Apple stock-price data instead of manually specifying the model parameters.

Historical data is downloaded using the `yfinance` library.

The key code is:

```python
data = yf.download("AAPL", start="2020-01-01", end="2025-12-31")

prices = data["Close"]["AAPL"]

returns = prices.pct_change().dropna()

mu_daily = returns.mean()

sigma_daily = returns.std()

mu = mu_daily * 252

sigma = sigma_daily * np.sqrt(252)
```

This allows the simulation to use estimates derived from observed AAPL price movements.

---

# 18. Downloading Historical Data

Historical AAPL data is downloaded using:

```python
data = yf.download(
    "AAPL",
    start="2020-01-01",
    end="2025-12-31"
)
```

This retrieves historical market data for Apple between the specified dates.

The downloaded data contains information such as:

* Opening price

* High price

* Low price

* Closing price

* Adjusted closing price

* Trading volume

The project focuses on the closing price.

---

# 19. Selecting the Closing Prices

The closing prices are selected using:

```python
prices = data["Close"]["AAPL"]
```

This produces a series containing the historical AAPL closing prices.

These prices are then used to calculate historical returns.

---

# 20. Understanding `data["Close"]["AAPL"]`

The expression:

```python
data["Close"]["AAPL"]
```

uses chained indexing.

First:

```python
data["Close"]
```

selects the `Close` column from the downloaded data.

The result contains AAPL's closing-price data.

Then:

```python
["AAPL"]
```

selects the AAPL series from that result.

Therefore:

```python
data["Close"]["AAPL"]
```

can be understood as:

```text
Data → Close prices → AAPL 

OR

data[column][subcolumn]
```

---

# 21. Calculating Historical Returns

The historical percentage returns are calculated using:

```python
returns = prices.pct_change().dropna()
```

`pct_change()` calculates the percentage change between consecutive prices.

Conceptually:

$$
R_t
=
\frac{P_t-P_{t-1}}{P_{t-1}}
$$

For example, if a stock moves from 100 to 105:

$$
\frac{105-100}{100}
=
0.05
$$

which represents a 5% return.

---

# 22. Why Does `pct_change()` Produce a Missing Value?

The first price has no previous price to compare against.

For example:

```text
Price

100

105

103
```

The first observation cannot have a return because there is no earlier price.

Therefore:

```python
prices.pct_change()
```

produces something conceptually like:

```text
NaN

0.05

-0.0190...
```

`dropna()` removes this missing observation:

```python
returns = prices.pct_change().dropna()
```

leaving only valid returns.

---

# 23. Historical Mean Return

The average daily historical return is calculated using:

```python
mu = returns.mean()
```

This estimates the average daily percentage return of AAPL over the selected historical period.

---

# 24. Historical Volatility

Historical daily volatility is calculated using:

```python
sigma_daily = returns.std()
```

This measures the historical standard deviation of AAPL's daily returns.

The daily volatility is then annualised using:

```python
sigma = sigma_daily * np.sqrt(252)
```

The square-root-of-time scaling comes from the assumption that returns are independent over time.

---

# 25. Historical Data Does Not Predict the Future

Using historical data makes the model more realistic than simply choosing arbitrary values, but it does not mean the historical parameters will accurately predict future stock behaviour.

The simulation assumes that the estimated historical:

* Mean return

* Volatility

remain relevant for the future simulation period.

In reality, these quantities can change significantly over time.

Therefore, V2 should be viewed as a historical-data-based modelling exercise rather than a guaranteed prediction of future prices.

---

# 26. V1 vs V2

| Feature         | V1                               | V2                                     |
| --------------- | -------------------------------- | -------------------------------------- |
| Initial price   | Manually specified               | Historical AAPL price                  |
| Drift           | Manually specified               | Estimated from historical returns      |
| Volatility      | Manually specified               | Estimated from historical returns      |
| Stock           | Generic                          | AAPL                                   |
| Historical data | No                               | Yes                                    |
| Purpose         | Understand Monte Carlo mechanics | Apply the model using real market data |

V1 focuses primarily on understanding how the simulation works.

V2 extends this by calibrating the model using historical market data.

---

# 27. Important NumPy Concepts Used

### `np.zeros()`

Creates an array filled with zeros, used to initialise the price matrix.

### `np.exp()`

Calculates the exponential function element-wise, used because GBM is expressed using an exponential function.

### `np.sqrt()`

Calculates the square root of dt

### `np.mean()`

Calculates the average of an array:

### `np.std()`

Calculates standard deviation:

### `np.percentile()`

Calculates a specified percentile

### `rng.standard_normal()`

Generates random values from a standard normal distribution.

# 28. Important Pandas Concepts Used

### `pct_change()`

Calculates percentage changes between consecutive observations

### `dropna()`

Removes missing values

### `mean()`

Calculates the average:

### `std()`

Calculates standard deviation:

These functions allow the historical AAPL data to be transformed into the parameters required by the Monte Carlo model.

---

# 29. Key Statistical Ideas

The project demonstrates several important statistical concepts:

### Random Sampling

Each simulated path is generated using random draws from a standard normal distribution.

### Expected Value

The mean of the simulated final prices provides an estimate of the expected simulated price.

### Variance and Volatility

The dispersion of the simulated outcomes reflects the effect of volatility.

### Probability

Conditions such as:

```python
final_price > 80
```

can be converted into estimated probabilities using Boolean arrays and `np.mean()`.

### Percentiles

Percentiles allow the distribution of simulated outcomes to be analysed without relying only on the mean.

---

# 30. Why More Simulations Help

Monte Carlo estimates are based on random sampling.

With a small number of simulations, the results can vary substantially depending on the particular random numbers generated.

For example:

```text
100 simulations
```

will generally provide a less stable estimate than:

```text
10,000 simulations
```

Increasing the number of simulations generally makes the estimated distribution more stable and reduces sampling noise.

This is an example of the **Law of Large Numbers**: as the number of observations increases, sample estimates tend to become closer to their underlying expected values.

---

# 31. Model Assumptions

The GBM model relies on several assumptions, including:

* Returns follow a normal distribution

* Volatility remains constant

* Expected return remains constant

* Returns are independent across time

* Prices evolve continuously

* Extreme market events are not explicitly modelled

---

# 32. Limitations

Although the model is useful for learning Monte Carlo methods, it does not capture every feature of real financial markets.

Some limitations include:

### Constant Volatility

Real market volatility changes over time.

### Constant Drift

Expected returns are unlikely to remain constant indefinitely.

### Normal Returns

Real financial returns can exhibit:

* Fat tails

* Skewness

* Extreme movements

which are not fully captured by a normal distribution.

### No Market Jumps

The model does not explicitly account for sudden price jumps caused by events such as:

* Earnings announcements

* Economic news

* Geopolitical events

* Market crashes

### Historical Parameters

V2 assumes that historical estimates of return and volatility remain relevant to future simulations.

---

# 33. Future Development

Possible future extensions include:

* Comparing simulated prices with actual historical prices

* Backtesting the model

* Implementing Value at Risk (VaR)

* Calculating confidence intervals

* Using log returns instead of simple returns

* Testing different historical periods

* Simulating multiple stocks

* Introducing correlations between assets

* Exploring alternative stochastic models

* Improving parameter calibration

* Comparing simulated distributions with actual market distributions

These extensions would allow the project to move from a basic Monte Carlo implementation toward a more realistic financial modelling framework.

---

# 34. Key Learning Outcomes

Through this project, I developed practical experience with:

* Monte Carlo simulation

* Geometric Brownian Motion

* NumPy vectorisation

* Random number generation

* Probability estimation

* Statistical analysis

* Historical financial data

* Return and volatility calculations

* Data manipulation with pandas

* Financial modelling assumptions

* Visualisation using Matplotlib

The project also helped connect mathematical and statistical concepts with practical financial applications.

---

# 35. Technologies

* Python

* NumPy

* pandas

* Matplotlib

* yfinance

* Jupyter Notebook

---

# 36. Disclaimer

This project is for educational and research purposes only.

The simulated prices are mathematical estimates based on model assumptions and historical data. They should not be interpreted as financial advice or reliable predictions of future stock prices.

---

# 37. Project Summary

### V1 — Baseline Simulation

```text
Initial Price
     │
     ▼
GBM Parameters
     │
     ▼
Random Normal Shocks
     │
     ▼
10,000 Simulated Paths
     │
     ▼
Final Price Distribution

     │

     ├── Mean

     ├── Standard Deviation

     ├── Percentiles

     └── Probability Estimates
```

### V2 — Historical AAPL Simulation

```text
Historical AAPL Prices
          │
          ▼
     Daily Returns
          │
          ▼
 Mean Return + Volatility
          │
          ▼
     GBM Simulation
          │
          ▼
 10,000 Future Price Paths
          │
          ▼
 Final Price Distribution
```


