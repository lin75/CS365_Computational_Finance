'''
@project       : Queens College CSCI 365/765 Computational Finance
@Instructor    : Dr. Alex Pang

@Student Name  : weiting lin

@Date          : May 2021

Discounted Cash Flow Model with Financial Data from Yahoo Financial

https://github.com/JECSand/yahoofinancials


'''
import enum
import calendar
import math
import pandas as pd
import numpy as np

import datetime 
from scipy.stats import norm

from math import log, exp, sqrt

from yahoofinancials import YahooFinancials 

class MyYahooFinancials(YahooFinancials):

    def __init__(self, symbol, freq = 'annual'):
        YahooFinancials.__init__(self, symbol)
        self.freq = freq

    def get_operating_cashflow(self):
        return self._financial_statement_data('cash', 'cashflowStatementHistory', 'totalCashFromOperatingActivities', self.freq)

    def get_capital_expenditures(self):
        return self._financial_statement_data('cash', 'cashflowStatementHistory', 'capitalExpenditures', self.freq)

    def get_long_term_debt(self):
        return self._financial_statement_data('balance', 'balanceSheetHistory', 'longTermDebt', self.freq)

    def get_account_payable(self):
        return self._financial_statement_data('balance', 'balanceSheetHistory', 'accountsPayable', self.freq)

    def get_total_current_liabilities(self):
        return self._financial_statement_data('balance', 'balanceSheetHistory', 'totalCurrentLiabilities', self.freq)

    def get_other_current_liabilities(self):
        return self._financial_statement_data('balance', 'balanceSheetHistory', 'otherCurrentLiab', self.freq)

    def get_cash(self):
        return self._financial_statement_data('balance', 'balanceSheetHistory', 'cash', self.freq)

    def get_short_term_investments(self):
        return self._financial_statement_data('balance', 'balanceSheetHistory', 'shortTermInvestments', self.freq)


class Stock(object):
    '''
    Stock class for getting financial statements
    default freq is annual
    '''
    def __init__(self, symbol):
        self.symbol = symbol
        self.yfinancial = MyYahooFinancials(symbol, 'annual')

        
    def get_total_debt(self):
        '''
        compute total_debt as long term debt + current debt 
        current debt = total current liabilities - accounts payables - other current liabilities (ignoring current deferred liabilities)
        '''
        result = None
        # TODO
        current_debt = self.yfinancial.get_total_current_liabilities() - self.yfinancial.get_account_payable() - self.yfinancial.get_other_current_liabilities()
        
        result = self.yfinancial.get_long_term_debt() + current_debt
        # end TODO
        return(result)

    def get_free_cashflow(self):
        '''
        get free cash flow as operating cashflow + capital expenditures (which will be negative)
        '''
        result = None
        # TODO
        result = self.yfinancial.get_operating_cashflow() + self.yfinancial.get_capital_expenditures()
        # end TODO
        return(result)

    def get_cash_and_short_term_investments(self):
        '''
        Return cash plus short term investment 
        '''
        result = None
        # TODO
        result = self.yfinancial.get_cash() + self.yfinancial.get_short_term_investments()
        # end TODO
        return(result)

    def get_num_shares_outstanding(self):
        '''
        get current number of shares outstanding from Yahoo financial library
        '''
        result = None
        # TODO
        result = self.yfinancial.get_num_shares_outstanding()
        # end TODO
        return(result)

    def get_beta(self):
        '''
        get beta from Yahoo financial
        '''
        result = None
        # TODO
        result = self.yfinancial.get_beta()
        # end TODO
        return(result)

    def lookup_wacc_by_beta(self, beta):
        '''
        lookup wacc by using the table in Slide 15 of the DiscountedCashFlowModel lecture powerpoint
        '''
        result = None
        # TODO:
        if beta < 0.8:
            result = 0.05
        elif beta < 1.0:
            result = 0.06
        elif beta < 1.1:
            result = 0.065
        elif beta < 1.2:
            result = 0.07
        elif beta < 1.3:
            result = 0.075
        elif beta < 1.5:
            result = 0.08
        elif beta < 1.6:
            result = 0.085
        elif beta > 1.6:
            result = 0.09
        else:
            print("Wrong beta input")
        #end TODO
        return(result)
        



class DiscountedCashFlowModel(object):
    '''
    DCF Model:

    FCC is assumed to go have growth rate by 3 periods, each of which has different growth rate
           short_term_growth_rate for the next 5Y
           medium_term_growth_rate from 6Y to 10Y
           long_term_growth_rate from 11Y to 20thY
    '''

    def __init__(self, stock, as_of_date):
        self.stock = stock
        self.as_of_date = as_of_date

        self.short_term_growth_rate = None
        self.medium_term_growth_rate = None
        self.long_term_growth_rate = None


    def set_FCC_growth_rate(self, short_term_rate, medium_term_rate, long_term_rate):
        self.short_term_growth_rate = short_term_rate
        self.medium_term_growth_rate = medium_term_rate
        self.long_term_growth_rate = long_term_rate


    def calc_fair_value(self):
        '''
        calculate the fair_value using DCF model

        1. calculate a yearly discount factor using the WACC
        2. Get the Free Cash flow
        3. Sum the discounted value of the FCC for the first 5 years using the short term growth rate
        4. Add the discounted value of the FCC from year 6 to the 10th year using the medium term growth rate
        5. Add the discounted value of the FCC from year 10 to the 20th year using the long term growth rate
        6. Compute the PV as cash + short term investments - total debt + the above sum of discounted free cash flow
        7. Return the stock fair value as PV divided by num of shares outstanding

        '''
        #TODO 
        # hint check out the DiscountedCashFlowModel notebook, you can almost copy-and-paste the code from there
        FCC = self.stock.get_free_cashflow()
        current_cash = self.stock.get_cash_and_short_term_investments()
        WACC = self.stock.lookup_wacc_by_beta(self.stock.get_beta())
        EPS5Y = self.short_term_growth_rate
        EPS6To10Y = self.medium_term_growth_rate
        EPS10To20Y = self.long_term_growth_rate
        total_debt = self.stock.get_total_debt()
        shares = self.stock.get_num_shares_outstanding()
        # ...
        DF = 1/(1+ WACC)
        DCF = 0
        for i in range(1, 6):
            DCF += FCC * (1+ EPS5Y)**i * DF ** i
            
        CF5 = FCC * (1+EPS5Y)**5
        for i in range(1, 6):
            DCF += CF5 * (1+EPS6To10Y)**i * DF ** (i+5)

        CF10 = CF5 * (1+EPS6To10Y)**5
        for i in range(1, 11):
            DCF += CF10 * (1+EPS10To20Y)**i * DF **(i + 10)
            
        PV = current_cash - total_debt + DCF
        result = PV/shares
        #end TODO
        return(result)




def _test():
    symbol = 'AAPL'
    as_of_date = datetime.date(2021, 4, 19)

    stock = Stock(symbol)
    model = DiscountedCashFlowModel(stock, as_of_date)

    print("Shares ", stock.get_num_shares_outstanding())

    print("FCC ", stock.get_free_cashflow())
    beta = stock.get_beta()
    wacc = stock.lookup_wacc_by_beta(beta)
    print("Beta ", beta)
    print("WACC ", wacc)

    print("Total debt ", stock.get_total_debt())

    print("cash ", stock.get_cash_and_short_term_investments())

    # look up from Finviz
    eps5y = 0.14
    model.set_FCC_growth_rate(eps5y, eps5y/2, 0.04)

    model_price = model.calc_fair_value()
    print(model_price)

if __name__ == "__main__":
    _test()
