import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt
from scipy.stats import norm, t

def BSM(tau, S, K, vol, r):
    d1 = (np.log(S/K) + tau * (r + vol ** 2 / 2)) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    return (S * norm.cdf(d1) - np.exp(- r * tau) * K * norm.cdf(d2)).squeeze()

def get_data(stocks, start, end, columnns=['Close']):
    yf.pdr_override()
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData[columnns]
    returns = stockData.pct_change().dropna()
    return stockData, returns

def get_statistics(returns):
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix


def MC_simulation(n_sims, T, weights, meanReturns, covMatrix, portfoliovalue = 1, method='daily_normal'):

    if len(np.shape(meanReturns)) == 1:
        meanReturns = np.expand_dims(meanReturns, axis=0)
    if len(np.shape(covMatrix)) == 2:
        covMatrix = np.expand_dims(covMatrix, axis=0)
    meanReturns = np.expand_dims(meanReturns, axis=2)


    if method == 'daily_normal':
        Z = np.random.normal(size=(T, len(weights), n_sims))
        L = np.linalg.cholesky(covMatrix)
        dailyReturns = meanReturns + L @ Z
        portfolio_simulation = np.cumprod((weights @ dailyReturns) + 1, axis=0)
        return portfolio_simulation * portfoliovalue, dailyReturns
    
    elif method[:-1] == 't':
        pass
    

def plot_simulation(portfolio_simulation, nplot):
    plt.plot(portfolio_simulation[:, :nplot])
    plt.xlabel('Time')
    plt.ylabel('portfolio value')
    plt.show()

def MC_VaR(returns, alpha=0.05):
    VaR = np.quantile(returns, alpha)
    CVaR = np.mean(returns[returns<=VaR])
    return VaR, CVaR

def mc_european_call(simulation, K, r, tau):
    T, N = simulation.shape
    finals = simulation[-1, :]
    values = np.exp(- r * tau) * np.maximum(finals-K, 0)
    se = np.std(values/np.sqrt(N))
    return np.mean(values), se, values
        
    