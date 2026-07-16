"""
y, 2020.9.13
statsmodels - coint - pair trading.py
https://amunategui.github.io/concepts-in-pair-trading/index.html
"""

import glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pathlib

# find the data directory and extract each CSV file
path = pathlib.Path('data')
np_array_list = []
for file_ in path.glob('*.csv'):
    df = pd.read_csv(file_, index_col=None, header=0)
    # get symbol name from file
    df['Symbol'] = file_.stem
    # pull only needed fields
    df = df[['Symbol', 'Date', 'Adj Close']]
    np_array_list.append(df.values)

# stack all arrays and tranfer it into a data frame
comb_np_array = np.vstack(np_array_list)
# simplify column names
stock_data_raw = pd.DataFrame(comb_np_array, columns=['Symbol', 'Date', 'Close'])
# fix datetime data
stock_data_raw['Date'] = pd.to_datetime(stock_data_raw['Date'], infer_datetime_format=True)
stock_data_raw['Date'] = stock_data_raw['Date'].dt.date

# check for NAs
stock_data_raw = stock_data_raw.dropna(axis=1, how='any')

# quick hack to get the column names (i.e. whatever stocks you loaded)
stock_data_tmp = stock_data_raw.copy()

# make symbol column header
stock_data_raw = stock_data_raw.pivot('Date', 'Symbol')
stock_data_raw.columns = stock_data_raw.columns.droplevel()
# collect correct header names (actual stocks)
column_names = list(stock_data_raw)

print(stock_data_raw.tail())

# hack to remove mult-index stuff
stock_data_raw = stock_data_tmp[['Symbol', 'Date', 'Close']]
stock_data_raw = stock_data_raw.pivot('Date', 'Symbol')
stock_data_raw.columns = stock_data_raw.columns.droplevel(-1)
stock_data_raw.columns = column_names

# replace NaNs with previous value
stock_data_raw.fillna(method='bfill', inplace=True)

print(stock_data_raw.tail())

stock_data = stock_data_raw.copy()

# Plot paired stocks on different axes
plt.figure(figsize=(12, 5))
ax1 = stock_data['FDX'].plot(color='green', grid=True, label='FDX')
ax2 = stock_data['UPS'].plot(color='purple', grid=True, secondary_y=True, label='UPS')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
plt.legend(h1 + h2, l1 + l2, loc=2)
plt.show()

plt.figure(figsize=(12, 5))
ax1 = stock_data['KO'].plot(color='green', grid=True, label='KO')
ax2 = stock_data['PEP'].plot(color='purple', grid=True, secondary_y=True, label='PEP')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
plt.legend(h1 + h2, l1 + l2, loc=2)
plt.show()


# Percent Change And Cumsum
def normalize_and_accumulate_series(data):
    # take tail to drop head NA
    return data.pct_change(fill_method=None).cumsum()


stock_data['FDX'] = normalize_and_accumulate_series(stock_data['FDX'])
stock_data['UPS'] = normalize_and_accumulate_series(stock_data['UPS'])
stock_data['KO'] = normalize_and_accumulate_series(stock_data['KO'])
stock_data['PEP'] = normalize_and_accumulate_series(stock_data['PEP'])

# remove first row with NAs
stock_data = stock_data.tail(len(stock_data) - 1)
print(stock_data.head())

# Plot paired stocks on same axes
fig, ax = plt.subplots(figsize=(12, 5))
plt.plot(stock_data['FDX'], color='green', label='FDX')
plt.plot(stock_data['UPS'], color='purple', label='UPS')
ax.grid(True)
plt.legend(loc=2)
plt.show()

fig, ax = plt.subplots(figsize=(12, 5))
plt.plot(stock_data['KO'], color='purple', label='KO')
plt.plot(stock_data['PEP'], color='green', label='PEP')
ax.grid(True)
plt.legend(loc=2)
plt.show()

# Getting some statistical measurements

# pip install scipy
# pip install statsmodels
from statsmodels.tsa.stattools import coint


def corr(data1, data2):
    """data1 & data2 should be numpy arrays."""
    mean1 = data1.mean()
    mean2 = data2.mean()
    std1 = data1.std()
    std2 = data2.std()
    corr = ((data1 * data2).mean() - mean1 * mean2) / (std1 * std2)
    return corr


stock_name_1 = 'KO'
stock_name_2 = 'PEP'

score, pvalue, _ = coint(stock_data[stock_name_1], stock_data[stock_name_2])
correlation = corr(stock_data[stock_name_1], stock_data[stock_name_2])

print('Correlation between %s and %s is %f' % (stock_name_1, stock_name_2, correlation))
print('Cointegration between %s and %s is %f' % (stock_name_1, stock_name_2, pvalue))

stock_name_1 = 'UPS'
stock_name_2 = 'FDX'

score, pvalue, _ = coint(stock_data[stock_name_1], stock_data[stock_name_2])
correlation = corr(stock_data[stock_name_1], stock_data[stock_name_2])

print('Correlation between %s and %s is %f' % (stock_name_1, stock_name_2, correlation))
print('Cointegration between %s and %s is %f' % (stock_name_1, stock_name_2, pvalue))

# Measuring separatation and spikes highlights
fig, ax = plt.subplots(figsize=(12,5))
plt.plot(stock_data['FDX'] - stock_data['UPS'], color='purple', label='Diff FDX minus UPS')
ax.grid(True)
ax.axhline(y=0, color='black', linestyle='-')
plt.legend(loc=2)
plt.show()

fig, ax = plt.subplots(figsize=(12,5))
plt.plot(stock_data['KO'] - stock_data['PEP'], color='purple', label='Diff KO minus PEP')
ax.grid(True)
ax.axhline(y=0, color='black', linestyle='-')
plt.legend(loc=2)
plt.show()

# Designing spike thresholds

# get the original data set
stock_data = stock_data_raw.copy()

def normalize_series(data):
    # take tail to drop head NA
    return data.pct_change(fill_method=None)

stock_data['FDX'] = normalize_series(stock_data['FDX'])
stock_data['UPS'] = normalize_series(stock_data['UPS'])
stock_data['KO'] = normalize_series(stock_data['KO'])
stock_data['PEP'] = normalize_series(stock_data['PEP'])

# remove first row with NAs
stock_data = stock_data.tail(len(stock_data)-1)
print(stock_data.head())

fig, ax = plt.subplots(figsize=(12,5))
plt.plot(stock_data['FDX'] - stock_data['UPS'], color='purple', label='Diff FDX minus UPS')
ax.grid(True)
ax.axhline(y=0, color='black', linestyle='-')
ax.axhline(y=0.02, color='red', linestyle='-')
ax.axhline(y=-0.02, color='red', linestyle='-')
plt.legend(loc=2)
plt.show()

fig, ax = plt.subplots(figsize=(12,5))
plt.plot(stock_data['KO'] - stock_data['PEP'], color='purple', label='Diff KO minus PEP')
ax.grid(True)
ax.axhline(y=0, color='black', linestyle='-')
ax.axhline(y=0.02, color='red', linestyle='-')
ax.axhline(y=-0.02, color='red', linestyle='-')
plt.legend(loc=2)
plt.show()
