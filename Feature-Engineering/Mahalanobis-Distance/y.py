# -*- coding: utf-8 -*-

"""
Created on Wed Nov 5 17:24:06 2014
@author: yRocket
"""

InFolder = 'D:\\Code\\Python'
InData = 'ydata2.txt'
InDataRows = ['Lt', 'Wf']
InDataCols = ['X', 'Y']
InDataValues = ['RADIUS', 'VALUE']
InDataCols4MD = InDataValues
InDataCols4Sort = ['TIME'] + InDataRows
InDataColumns = InDataRows + InDataCols + InDataValues + InDataCols4Sort  # All
OutData = 'y.out'
OutFolder = InFolder

import datetime
import pandas as pd
import numpy as np
# import yFunctions as yy
from yFunctions import (MahalanobisDistanceArray, yLSRwithMDCut)
import matplotlib.pyplot as pl

StartTime = datetime.datetime.now()

df = pd.read_table(InFolder + '\\' + InData, header=0, usecols=InDataColumns)
df.sort(columns=InDataCols4Sort, ascending=[1] * len(InDataCols4Sort), inplace=True)
df = pd.pivot_table(df, rows=InDataRows, cols=InDataCols,
                    values=InDataValues, aggfunc='last')
out = df.ix[:, 0:4]
out.columns = ['median', 'slope', 'intercept', 'r2']
for i in range(len(df)):
    npa = df.iloc[i].values
    npa = np.array_split(npa, 2)
    x = npa[0]  # RADIUS
    y = npa[1]  # VALUE
    md = MahalanobisDistanceArray(x, y)
    slope, intercept, r_value = yLSRwithMDCut(x, y, md, 2)
    out.values[i, 0] = np.mean(y, dtype=np.float64)
    out.values[i, 1] = slope
    out.values[i, 2] = intercept
    out.values[i, 3] = r_value ** 2
out.to_csv(OutFolder + '\\' + OutData, index=True, sep='\t')

print(out.shape)
pl.subplot(211)
pl.title("Radial Regression")
pl.plot(out['median'], out['slope'], 'r+')
pl.xlabel("V")
pl.ylabel("Slope")
pl.subplot(212)
pl.plot(out['median'], out['intercept'], 'r+')
pl.xlabel("V")
pl.ylabel("Intercept")
pl.show()

EndtTime = datetime.datetime.now()
TotalTime = EndtTime - StartTime
print("Total execution time is :", TotalTime)
