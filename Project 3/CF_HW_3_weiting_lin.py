'''
@project       : Queens College CSCI 365/765 Computational Finance
@Instructor    : Dr. Alex Pang

@Student Name  : weiting lin

@Date          : Spring 2021

Technical Indicators

'''
import enum
import calendar
import math
import pandas as pd
import numpy as np

from datetime import date
from scipy.stats import norm

from math import log, exp, sqrt

from stock import *

class SimpleMovingAverages(object):
    '''
    On given a OHLCV data frame, calculate corresponding simple moving averages
    '''
    def __init__(self, ohlcv_df, periods):
        #
        self.ohlcv_df = ohlcv_df
        self.periods = periods
        self._sma = {}

    def _calc(self, period, price_source):
        '''
        for a given period, calc the SMA as a pandas series from the price_source
        which can be  open, high, low or close
        '''
        result = None
        #TODO: implement details here
        # hint: use rolling method from pandas
        result = self.ohlcv_df[price_source].rolling(period,min_periods=1).mean()
        #end TODO
        return(result)
        
    def run(self, price_source = 'close'):
        '''
        Calculate all the simple moving averages as a dict
        '''
        for period in self.periods:
            self._sma[period] = self._calc(period, price_source)
    
    def get_series(self, period):
        return(self._sma[period])

    
class ExponentialMovingAverages(object):
    '''
    On given a OHLCV data frame, calculate corresponding simple moving averages
    '''
    def __init__(self, ohlcv_df, periods):
        #
        self.ohlcv_df = ohlcv_df
        self.periods = periods
        self._ema = {}

    def _calc(self, period):
        '''
        for a given period, calc the SMA as a pandas series
        '''
        result = None
        #TODO: implement details here
        result = self.ohlcv_df['close'].ewm(span=period,adjust=False).mean()
        #end TODO
        return(result)
        
    def run(self):
        '''
        Calculate all the simple moving averages as a dict
        '''
        for period in self.periods:
            self._ema[period] = self._calc(period)

    def get_series(self, period):
        return(self._ema[period])

class VWAP(object):

    def __init__(self, ohlcv_df):
        self.ohlcv_df = ohlcv_df
        self.vwap = None

    def get_series(self):
        return(self.vwap)

    def run(self):
        '''
        '''
        #TODO: implement details here
        price = (self.ohlcv_df['high'] + self.ohlcv_df['low'] + self.ohlcv_df['close']) / 3
        self.vwap = ((self.ohlcv_df['volume'] * price).cumsum()) / self.ohlcv_df['volume'].cumsum()
        #end TODO
        
        return(self.vwap)


class RSI(object):

    def __init__(self, ohlcv_df):
        self.ohlcv_df = ohlcv_df
        self.rsi = None

    def get_series(self):
        return(self.rsi)

    def run(self):
        '''
        '''
        #TODO: implement details here
        diff = self.ohlcv_df['close'].diff()
        
        gain = diff.copy()
        gain[diff<=0]=0.0
        
        loss = abs(diff.copy())
        loss[diff>0]=0.0
        
        avg_gain = gain.ewm(com=13,adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(com=13,adjust=False, min_periods=14).mean()
        
        try:
            rs = abs(avg_gain/avg_loss)
            self.rsi= 100-100/(1+rs)
        except ZeroDivisionError:
            print ("Can not divide by zero")
        #end TODO
        
        return(self.rsi)


def _test():
    symbol = 'AAPL'
    stock = Stock(symbol)
    start_date = datetime.date(2019, 1, 1)
    end_date = datetime.date.today()

    stock.get_daily_hist_price(start_date, end_date)

    periods = [9, 20, 50, 100, 200]
    smas = SimpleMovingAverages(stock.ohlcv_df, periods)
    smas.run()
    s1 = smas.get_series(9)
    print(s1.index)
    print(s1)

if __name__ == "__main__":
    _test()

