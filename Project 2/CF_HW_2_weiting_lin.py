'''
@project       : Queens College CSCI 365/765 Computational Finance
@Instructor    : Dr. Alex Pang

@Student Name  : weiting lin

@Date          : April 2021

A Simple Option Pricer Class

'''
import enum
import calendar
import math
import pandas as pd
import numpy as np

from datetime import date
from scipy.stats import norm

from math import log, exp, sqrt

class Stock(object):
    '''
    mu is the expected return
    sigma is the volatility of the stock
    dividend_yield is the continous dividend yield paid by the stock
    '''
    def __init__(self, symbol, spot_price, sigma, mu, dividend_yield = 0):
        self.symbol = symbol
        self.spot_price = spot_price
        self.sigma = sigma
        self.mu = mu 
        self.dividend_yield = dividend_yield

class Option(object):
    '''
    time_to_expiry is the number of days till expiry_date expressed in unit of years
    underlying is the underlying stock object
    '''

    class Type(enum.Enum):
        CALL = "Call"
        PUT  = "Put"

    class Style(enum.Enum):
        EUROPEAN = "European"
        AMERICAN = "American"

    def __init__(self, option_type, option_style,  underlying, time_to_expiry, strike):
        self.option_type = option_type
        self.option_style = option_style
        self.underlying = underlying
        self.time_to_expiry = time_to_expiry
        self.strike = strike

class EuropeanCallOption(Option):
    def __init__(self, underlying, time_to_expiry, strike):
        Option.__init__(self, Option.Type.CALL, Option.Style.EUROPEAN,
                        underlying, time_to_expiry, strike)

class EuropeanPutOption(Option):
    def __init__(self, underlying, time_to_expiry, strike):
        Option.__init__(self, Option.Type.PUT, Option.Style.EUROPEAN,
                        underlying, time_to_expiry, strike)

class AmericanCallOption(Option):
    def __init__(self, underlying, time_to_expiry, strike):
        Option.__init__(self, Option.Type.CALL, Option.Style.AMERICAN,
                        underlying, time_to_expiry, strike)

class AmericanPutOption(Option):
    def __init__(self, underlying, time_to_expiry, strike):
        Option.__init__(self, Option.Type.PUT, Option.Style.AMERICAN,
                        underlying, time_to_expiry, strike)


class OptionPricer(object):
    '''
    Option Pricer for calculating option price using either Binomial or Black-Scholes Model
    '''

    def __init__(self, pricing_date, risk_free_rate):
        self.pricing_date = pricing_date
        self.risk_free_rate = risk_free_rate

    def _binomial_european_call(self, S_0, K, T, r, sigma, q, N):
        '''
        Calculate the price of an European call using Binomial Tree
        S_0 - stock price today
        K - strike price of the option
        T - time to expiry in unit of years
        r - risk-free interest rate
        sigma - the volatility of the stock
        q - the continous dividend yield of the stock
        N - number of periods in the tree
        '''
        dt = T/N
        u =  exp(sigma * sqrt(dt)) #
        d = 1/u
        prob = (exp((r-q) * dt) - d)/(u - d)
        df = exp(-r * dt) #1-period DF
        C = {}
        
        #TODO
        for m in range(0, N+1):
            S_T = S_0 * (u ** (2*m - N)) #yellow
            C[(N, m)] = max(S_T - K, 0) #green
            
        for k in range(N-1, -1, -1): #it start from last one and --
            for m in range(0,k+1):
                C[(k, m)]=df*(prob*C[(k+1, m+1)]+(1-prob)*C[(k+1, m)])
        
        # end TODO
        return C[(0,0)]        

    def _binomial_european_put(self, S_0, K, T, r, sigma, q, N):
        '''
        Calculate the price of an European put using Binomial Tree
        S_0 - stock price today
        K - strike price of the option
        T - time to expiry in unit of years
        r - risk-free interest rate
        sigma - the volatility of the stock
        q - the continous dividend yield of the stock
        N - number of steps in the model
        '''
        dt = T/N
        u =  exp(sigma * sqrt(dt))
        d = 1/u
        prob = (exp((r - q) * dt) - d)/(u - d)
        df = exp(-r * dt) #1-period DF
        P = {}
        
        # TODO：
        for m in range(0, N+1):
            S_T = S_0 * (u ** (2*m - N)) #yellow
            P[(N, m)] = max(K - S_T, 0) #green
            
        for k in range(N-1, -1, -1): #it start from last one and --
            for m in range(0,k+1):
                P[(k, m)]=df*(prob*P[(k+1, m+1)]+(1-prob)*P[(k+1, m)])
                
        # end TODO        
        return P[(0,0)]        

    def _binomial_american_call(self, S_0, K, T, r, sigma, q, N):
        '''
        Calculate the price of an American call using Binomial Tree
        S_0 - stock price today
        K - strike price of the option
        T - time until expiry of the option
        r - risk-free interest rate
        sigma - the volatility of the stock
        q - the continous dividend yield of the stock
        N - number of steps in the model
        '''
        dt = T/N
        u =  exp(sigma * sqrt(dt))
        d = 1/u
        prob = (exp((r-q) * dt) - d)/(u - d)
        df = exp(-r * dt)
        C = {}
        S_C = {}
        
        # TODO:
        for m in range(0, N+1):
            S_T = S_0 * (u ** (2*m - N))
            C[(N, m)] = max(S_T - K, 0)
            print("S_T: ",S_T)
            print(C[(N, m)])

        for k in range(N-1, -1, -1):
            for m in range(0, k+1):
                S_C[(k, m)] = S_0 * (u ** (2*m - k))
                C[(k, m)] = max(df * (prob * C[(k+1, m+1)] + (1-prob) * C[(k+1, m)]), S_C[(k, m)] - K)
                print(C[(k, m)])
        
        # end TODO
        return C[(0,0)]       

    def _binomial_american_put(self, S_0, K, T, r, sigma, q, N):
        '''
        Calculate the price of an American put using Binomial Tree
        S_0 - stock price today
        K - strike price of the option
        T - time to expiry in unit of years
        r - risk-free interest rate
        sigma - the volatility of the stock
        N - number of steps in the model
        '''
        dt = T/N
        u =  exp(sigma * sqrt(dt))
        d = 1/u
        prob = (exp((r-q) * dt) - d)/(u - d)
        df = exp(-r * dt)
        P = {}
        S_P = {}


        # TODO: implement details here
        for m in range(0, N+1):
            S_T = S_0 * (u ** (2*m - N))
            P[(N, m)] = max(K - S_T, 0)

        for k in range(N-1, -1, -1):
            for m in range(0, k+1):
                S_P[(k, m)] = S_0 * (u ** (2*m - k))
                P[(k, m)] = max(df * (prob * P[(k+1, m+1)] + (1-prob) * P[(k+1, m)]), K - S_P[(k, m)])
                
        # end TODO
        return P[(0,0)]       


    def calc_binomial_model_price(self, option, num_of_period):
        '''
        Calculate the price of the option using num_of_period Binomial Model 
        '''
        if option.option_type == Option.Type.CALL and option.option_style == Option.Style.EUROPEAN:
            px = self._binomial_european_call(option.underlying.spot_price, option.strike, option.time_to_expiry, self.risk_free_rate, 
                                              option.underlying.sigma, option.underlying.dividend_yield, num_of_period)
        elif option.option_type == Option.Type.PUT and option.option_style == Option.Style.EUROPEAN:
            px = self._binomial_european_put(option.underlying.spot_price, option.strike, option.time_to_expiry, self.risk_free_rate, 
                                             option.underlying.sigma, option.underlying.dividend_yield, num_of_period)
        elif option.option_type == Option.Type.CALL and option.option_style == Option.Style.AMERICAN:
            px = self._binomial_american_call(option.underlying.spot_price, option.strike, option.time_to_expiry, self.risk_free_rate, 
                                              option.underlying.sigma, option.underlying.dividend_yield, num_of_period)
        elif option.option_type == Option.Type.PUT and option.option_style == Option.Style.AMERICAN:
            px = self._binomial_american_put(option.underlying.spot_price, option.strike, option.time_to_expiry, self.risk_free_rate, 
                                             option.underlying.sigma, option.underlying.dividend_yield, num_of_period)

        return(px)


    def calc_parity_price(self, option, option_price):
        '''
        return the put price from Put-Call Parity if input option is a call
        else return the call price from Put-Call Parity if input option is a put
        '''
        result = None
        
        # TODO: implement details here
        if option.option_style == Option.Style.AMERICAN:
            raise Exception("The put-call parity price for American option not implemented yet")
        else:
            K = option.strike
            r = self.risk_free_rate
            S_0 = option.underlying.spot_price
            q = option.underlying.dividend_yield
            T = option.time_to_expiry

            if option.option_type == Option.Type.CALL:
                result = option_price + K * exp(-r * T) - S_0
            elif option.option_type == Option.Type.PUT:
                result = option_price + S_0 - K * exp(-r * T)
            else:
                raise Exception("Wrong type of Option")
        # end TODO
        return(result)

    def calc_black_scholes_price(self, option):
        '''
        Calculate the price of the call or put option using Black-Scholes model
   
        '''
        px = None
        
        #TODO:
        if option.option_style == Option.Style.AMERICAN:
            raise Exception("B\S price for American option not implemented yet")
        else:
            K = option.strike
            r = self.risk_free_rate
            S_0 = option.underlying.spot_price
            q = option.underlying.dividend_yield
            T = option.time_to_expiry
            sigma = option.underlying.sigma
        
            d1 = (log(S_0/K) + (r - q + (sigma**2)/2) * T) / (sigma * sqrt(T))
            d2 = d1-(sigma * sqrt(T))
            
            if option.option_type == Option.Type.CALL:
                px = S_0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
            elif option.option_type == Option.Type.PUT:
                px = K * exp(-r * T) * norm.cdf(-d2) - S_0 * norm.cdf(-d1)
            else:
                raise Exception("Wrong type of Option")
        # end TODO:        
        return(px)
                
    def calc_delta(self, option):
        result = None
        
        #TODO: implement
        K = option.strike
        r = self.risk_free_rate
        S_0 = option.underlying.spot_price
        q = option.underlying.dividend_yield
        T = option.time_to_expiry
        sigma = option.underlying.sigma
            
        d1 = (log(S_0/K) + (r - q + (sigma**2)/2) * T) / (sigma * sqrt(T))
        
        if option.option_style == Option.Style.AMERICAN:
            raise Exception("Greeks value for American option not implemented yet")
        else:
            if option.option_type == Option.Type.CALL:
                result = exp(-q * T) * norm.cdf(d1)
            elif option.option_type == Option.Type.PUT:
                result = exp(-q * T) *( norm.cdf(d1) - 1 )
            else:
                raise Exception("Wrong type of Option")
            
         # end TODO:    
        return(result)

    def calc_gamma(self, option):
        result = None
        
        # TODO: implement
        K = option.strike
        r = self.risk_free_rate
        S_0 = option.underlying.spot_price
        q = option.underlying.dividend_yield
        T = option.time_to_expiry
        sigma = option.underlying.sigma
        
        d1 = (log(S_0/K) + (r - q + (sigma**2)/2) * T) / (sigma * sqrt(T))
        
        if option.option_style == Option.Style.AMERICAN:
            raise Exception("Greeks value for American option not implemented yet")
        else:
            if option.option_type == Option.Type.CALL or option.option_type == Option.Type.PUT:
                result = (norm.pdf(d1) * exp(-q * T))/(S_0 * sigma * sqrt(T))
            else:
                raise Exception("Wrong type of Option")
        
        # end TODO:
        return(result)

    def calc_theta(self, option):
        result = None
        
        # TODO: implement
        K = option.strike
        r = self.risk_free_rate
        S_0 = option.underlying.spot_price
        q = option.underlying.dividend_yield
        T = option.time_to_expiry
        sigma = option.underlying.sigma
        
        
        d1 = (log(S_0/K) + (r - q + (sigma**2)/2) * T) / (sigma * sqrt(T))
        d2 = d1-(sigma * sqrt(T)) 
        
        if option.option_style == Option.Style.AMERICAN:
            raise Exception("Greeks value for American option not implemented yet")
        else:
            if option.option_type == Option.Type.CALL:
                result = (- S_0 * norm.pdf(d1) * sigma * exp(-q * T)) / (2 * sqrt(T)) + q * S_0 * norm.cdf(d1) * exp(-q * T) - r * K * exp(-r * T) * norm.cdf(d2)      
            elif option.option_type == Option.Type.PUT:
                result = (- S_0 * norm.pdf(d1) * sigma * exp(-q * T)) / (2 * sqrt(T)) - q * S_0 * norm.cdf(-d1) * exp(-q * T) + r * K * exp(-r * T) * norm.cdf(- d2)
            else:
                raise Exception("Wrong type of Option")
            
        # end TODO:
        return(result)

    def calc_vega(self, option):
        result = None
        
        # TODO: implement
        K = option.strike
        r = self.risk_free_rate
        S_0 = option.underlying.spot_price
        q = option.underlying.dividend_yield
        T = option.time_to_expiry
        sigma = option.underlying.sigma
        
        d1 = log(S_0/K) + (r - q + (sigma**2)/2) * T / (sigma * sqrt(T))
        
        if option.option_style == Option.Style.AMERICAN:
            raise Exception("Greeks value for American option not implemented yet")
        else:
            if option.option_type == Option.Type.CALL or option.option_type == Option.Type.PUT:
                result =S_0 * sqrt(T) * norm.pdf(d1) * exp(-q * T)
            else:
                raise Exception("Wrong type of Option")
        # end TODO:
        
        return(result)

    def calc_rho(self, option):
        result = None
        
        # TODO: implement
        K = option.strike
        r = self.risk_free_rate
        S_0 = option.underlying.spot_price
        q = option.underlying.dividend_yield
        T = option.time_to_expiry
        sigma = option.underlying.sigma
        
        d1 = (log(S_0/K) + (r - q + (sigma**2)/2) * T) / (sigma * sqrt(T))
        d2=d1-(sigma * sqrt(T))
        
        if option.option_style == Option.Style.AMERICAN:
            raise Exception("Greeks value for American option not implemented yet")
        else:
            if option.option_type == Option.Type.CALL:
                result = K * T * exp(-r * T) * norm.cdf(d2)
            elif option.option_type == Option.Type.PUT:
                result = K * T * exp(-r * T) * norm.cdf(-d2)
            else:
                raise Exception("Wrong type of Option")
            
        # end TODO:
        return(result)
    
    