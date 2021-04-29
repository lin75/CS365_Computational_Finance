'''

@project       : Queens College CSCI 365/765 Computational Finance
@Instructor    : Dr. Alex Pang

@Student Name  : weiting lin

@Date          : March 2021

A Simplified Bond Class

'''

import math
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from bisection_method import *

import enum
import calendar

from datetime import date

class DayCount(enum.Enum):
    DAYCOUNT_30360 = "30/360"
    DAYCOUNT_ACTUAL_360 = "Actual/360"
    DAYCOUNT_ACTUAL_ACTUAL = "Actual/Actual"

class PaymentFrequency(enum.Enum):
    ANNUAL     = "Annual"
    SEMIANNUAL = "Semi-annual"
    QUARTERLY  = "Quarterly"
    MONTHLY    = "Monthly"
    CONTINUOUS = "Continuous"

    
class Bond(object):
    '''
    term is maturity term in years
    coupon is the coupon in percent
    maturity date will be issue_date + term
    '''
    def __init__(self, issue_date, term, day_count, payment_freq, coupon, principal = 100):
        self.issue_date = issue_date
        self.term = term
        self.day_count = day_count
        self.payment_freq = payment_freq
        self.coupon = coupon
        self.principal = principal
        # internal data structures
        self.payment_times_in_year = []
        self.payment_dates = []
        self.coupon_payment = []
        self._calc()

    def _add_months(self, dt, n_months):
        return(dt + relativedelta(months = n_months))

    def _calc(self):
        # calculate maturity date
        self.maturity_date = self._add_months(self.issue_date, 12 * self.term)

        # calculate all the payment dates
        dt = self.issue_date
        while dt < self.maturity_date:
            if self.payment_freq == PaymentFrequency.ANNUAL:
                next_dt = self._add_months(dt, 12)
            elif self.payment_freq == PaymentFrequency.SEMIANNUAL:
                next_dt = self._add_months(dt, 6)
            elif self.payment_freq == PaymentFrequency.QUARTERLY:
                next_dt = self._add_months(dt, 4)
            elif self.payment_freq == PaymentFrequency.MONTHLY:
                next_dt = self._add_months(dt, 1)
            else:
                raise Exception("Unsupported Payment frequency")
                
            if next_dt <= self.maturity_date:
                self.payment_dates.append(next_dt)
                
            dt = next_dt

        # calculate the future cashflow vectors
        if self.payment_freq == PaymentFrequency.ANNUAL:
            coupon_cf = self.principal * self.coupon 
        elif self.payment_freq == PaymentFrequency.SEMIANNUAL:
            coupon_cf = self.principal * self.coupon / 2
        elif self.payment_freq == PaymentFrequency.QUARTERLY:
            coupon_cf = self.principal * self.coupon / 4 
        elif self.payment_freq == PaymentFrequency.MONTHLY:
            coupon_cf = self.principal * self.coupon / 12 
        else:
            raise Exception("Unsupported Payment frequency")
            
        self.coupon_payment = [ coupon_cf for i in range(len(self.payment_dates))]
        
        # calculate payment_time in years
        if self.payment_freq == PaymentFrequency.ANNUAL:
            period = 1
        elif self.payment_freq == PaymentFrequency.SEMIANNUAL:
            period = 1/2
        elif self.payment_freq == PaymentFrequency.QUARTERLY:
            period = 1/4
        elif self.payment_freq == PaymentFrequency.MONTHLY:
            period = 1/12
        else:
            raise Exception("Unsupported Payment frequency")

        self.payment_times_in_year = [ period * (i+1) for i in range(len(self.payment_dates))]

    def get_next_payment_date(self, as_of_date):
        '''
        return the next payment date after as_of_date
        '''
        if as_of_date <= self.issue_date:
            return(self.payment_dates[0])
        elif as_of_date > self.payment_dates[-1]:
            return(None)
        else:
            i = 0
            while i < len(self.payment_dates):
                dt = self.payment_dates[i]
                if as_of_date <= dt:
                    return(dt)
                else:
                    i += 1
            return(None)


class BondCalculator(object):
    '''
    Bond Calculator. 
    '''

    def __init__(self, pricing_date):
        self.pricing_date = pricing_date

    def calc_one_period_discount_factor(self, bond, yld):
        '''
        Calculate the one period discount factor
        ''' 
        # TODO:
        if bond.payment_freq==PaymentFrequency.ANNUAL:
            df=1/(1+yld)
        elif bond.payment_freq==PaymentFrequency.SEMIANNUAL:
            df=1/(1+yld/2)
        elif bond.payment_freq == PaymentFrequency.QUARTERLY:
            df=1/(1+yld/4)
        elif bond.payment_freq == PaymentFrequency.MONTHLY:
            df=1/(1+yld/12)
        else:
            raise Exception("Unsupported Payment frequency")
          
        return(df)
       

    def calc_clean_price(self, bond, yld):
        '''
        Calculate the price of the bond as of the pricing_date for a given yield
        as a percentage
        '''
        # TODO:
        px = None
        one_period_factor = self.calc_one_period_discount_factor(bond, yld)
        DF = [math.pow(one_period_factor, i+1) for i in range(len(bond.coupon_payment))]
        CF = [c for c in bond.coupon_payment]
        CF[-1] += bond.principal
        PVs = [ CF[i] * DF[i] for i in range(len(bond.coupon_payment))]
        
        TotalPV=0
        for i in PVs:
            TotalPV = TotalPV + i
            
        px = TotalPV/bond.principal
        
        return(px*100)

    def calc_accrual_interest(self, bond, settle_date):
        '''
        calculate the accrual interest on given a settle_date
        by calculating the previous payment date first and use the date count
        from previous payment date to the settle_date
        '''
        def get_actual360_daycount_frac(start, end):
            day_in_year = 360
            day_count = (end - start).days
            return(day_count / day_in_year)
        
        def get_30360_daycount_frac(start, end):
            day_in_year = 360
            day_count = 360*(end.year - start.year) + 30*(end.month - start.month - 1) + max(0, 30 - start.day) + min(30, end.day)
            return(day_count / day_in_year )
        
        next_payment_date=bond.get_next_payment_date(settle_date)
        
        if bond.payment_freq == PaymentFrequency.ANNUAL:
            last_payment_date=next_payment_date-relativedelta(months = 12)
        elif bond.payment_freq == PaymentFrequency.SEMIANNUAL:
            last_payment_date=next_payment_date-relativedelta(months = 6)
        elif bond.payment_freq == PaymentFrequency.QUARTERLY:
            last_payment_date=next_payment_date-relativedelta(months = 4)
        elif bond.payment_freq == PaymentFrequency.MONTHLY:
            last_payment_date=next_payment_date-relativedelta(months = 1)
        
        if(bond.day_count == DayCount.DAYCOUNT_ACTUAL_360):
            return(get_actual360_daycount_frac(last_payment_date,settle_date)*bond.coupon*100)
        elif(bond.day_count == DayCount.DAYCOUNT_30360):
            return(get_30360_daycount_frac(last_payment_date,settle_date)*bond.coupon*100)
        
       

    def calc_macaulay_duration(self, bond, yld):
        '''
        time to cashflow weighted by PV
        '''
        one_period_factor = self.calc_one_period_discount_factor(bond, yld)
        # DF=Discount Factor
        DF = [math.pow(one_period_factor, i+1) for i in range(len(bond.coupon_payment))]
        # CF=Cash Flow
        CF = [c for c in bond.coupon_payment]
        CF[-1] += bond.principal
        # PV=Present value = DF*CF
        PVs = [ CF[i] * DF[i] for i in range(len(bond.coupon_payment))]

        # TODO:
        wavg = [bond.payment_times_in_year[i] * PVs[i] for i in range(len(bond.coupon_payment))]

        return( sum(wavg) / sum(PVs))

    def calc_modified_duration(self, bond, yld):
        '''
        calculate modified duration
        '''
        D = self.calc_macaulay_duration(bond, yld)
        # TODO:
        if bond.payment_freq == PaymentFrequency.ANNUAL:
            return(D/(1+yld))
        elif bond.payment_freq == PaymentFrequency.SEMIANNUAL:      
            return(D/(1+yld/2))
        elif bond.payment_freq == PaymentFrequency.QUARTERLY:
            return(D/(1+yld/4))
        elif bond.payment_freq == PaymentFrequency.MONTHLY:
            return(D/(1+yld/12))

    def calc_yield(self, bond, bond_price):
        '''
        Calculate the yield to maturity on given a bond price using bisection method
        '''

        def match_price(yld):
            calculator = BondCalculator(self.pricing_date)
            px=calculator.calc_clean_price(bond, yld)
            return(px - bond_price)

        # call the bisection method
        # TODO:
        yld, n_iteractions =bisection(match_price, 0, 10000, eps=1.0e-6)
        return(yld*100)

    def calc_convexity(self, bond, yld):
        '''
        Calculate the convexity
        '''
        #TODO





