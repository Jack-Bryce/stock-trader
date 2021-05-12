import alpaca_trade_api as tradeapi
import math
import numpy as np
from scipy.signal import argrelextrema

#This handles the Bollinger Band calculations.
class BollingerBands:
    """
    Purpose: This initializes the appropriate values for the Bollinger Bands calculations.
    Parameters: N - This is the size of the period we want to be looking at.
    """
    def __init__(self, N):
        self.period = []
        self.period_size = N

    """
    Purpose: This calculates the bollinger bands.
    Parameters: x - This is the closing value we are using to calculate the bollinger bands with.
    """
    def perform(self, x):
        #Maintain the size of the list.
        self.period.append(x)
        if len(self.period) == self.period_size+1:
            self.period.pop(0)

        #Find the standard deviations and return it with the bollinger bands.
        if len(self.period) == self.period_size:
            sma = sum(self.period)/len(self.period)

            variance = sum([pow(x - sma, 2) for x in self.period])
            sigma = math.sqrt(variance/(self.period_size-1))
            
            return {"upper": sma + 2*sigma, "lower": sma - 2*sigma, "sigma": sigma}

        return None

#This is all of the common functions extracted into a class.
class Miscellaneous:
    """
    Purpose: This instantiaties the instance variables.
    Parameters: api - This is an instance of the api object we will be using.
    """
    def __init__(self, api):
        self.api = api

    """
    Purpose: This retrieves 25% of the buying power of the account.
    Parameters: N/A
    """
    def get_acc_amount(self):
        acc = self.api.get_account()
        return 0.25 * float(acc.buying_power) #May be a good idea, to use cash instead.

    """
    Purpose: This retrieves the maximum amount of normalized closing values.
    Parameters: sym - This is the symbol of the stock that we wish to retrieve the data of.
    """
    def get_info(self, sym):
        #Retrieve the maximum amount of historical data and ensure that there is that much.
        temp = self.api.get_barset(sym, "day", limit=1000)[sym]
        if (len(temp) == 1000) is not True:
            return None

        #Return the closing values.
        return [bar.c for bar in temp]

    """
    Purpose: This identifies all of the turning points of the given array.
    Parameters: x - This is the dataset we would like to find the turning points for.
    """
    def find_extrema(self, x):
        #Identify all of the minima and maxima.
        x = np.array(x)
        maxIdx, minIdx = argrelextrema(x, np.greater), argrelextrema(x, np.less)
        idxs = []

        #Find the periods of higher highs.
        maxima = 0
        for idx in maxIdx[0]:
            if x[idx] >= maxima:
                maxima = x[idx]
            else:
                idxs.append(idx)
        
        maxIdx = idxs
        idxs = []

        #Find the periods of lower lows.
        minima = 0
        for i in range(1, len(minIdx[0])):
            idx = minIdx[0][i]
            if x[idx] <= minima:
                minima = x[idx]
            else:
                idxs.append(idx)

        minIdx = idxs
        idxs = []

        #Store the information into idxs as a dictionary.
        for idx in maxIdx:
            idxs.append(idx)

        for idx in minIdx:
            idxs.append(idx)

        idxs.sort()
        return idxs

    """
    Purpose: This submits an order given the RSI divergence strategy.
    Parameters: sym - This is the symbol of the stock we went to make an order on,
                amount - This is the number of stocks that we wish to buy,
                sigma - This is the value we are using for the trailing stop,
                gradSMA - This is the gradient of the sma trend line,
                gradRSI - This is the gradient oof the rsi trend line,
                isInPositions - This shows whether or not a stock is already bought or not,
                bearishType - This is the order type when there is a bearish divergence,
                bullishType - This is the order type when thee is a bullish divergence.
    """
    def submit_order(self, sym, amount, sigma, gradSMA, gradRSI, isInPositions, bearishType, bullishType):
        if gradSMA < 0:
            if gradRSI > 0 and isInPositions:
                self.submit_order_call(sym, amount, sigma, bearishType)
            elif gradRSI < 0 and isInPositions is not True:
                self.submit_order_call(sym, amount, sigma, bullishType)
        elif gradSMA > 0:
            if gradRSI < 0 and isInPositions:
                self.submit_order_call(sym, amount, sigma, bearishType)
            elif gradRSI > 0 and isInPositions is not True:
                self.submit_order_call(sym, amount, sigma, bullishType)
        elif gradSMA == 0:
            if gradRSI < 0 and isInPositions:
                self.submit_order_call(sym, amount, sigma, bearishType)
            elif gradRSI > 0 and isInPositions is not True:
                self.submit_order_call(sym, amount, sigma, bullishType)

    """
    Purpose: This submits an order.
    Parameters: sym - This is the symbol of the stock we went to make an order on,
                amount - This is the number of stocks that we wish to buy,
                sigma - This is the value we are using for the trailing stop,
                orderType - This is the order (buy or sell) we would like to submit.
    """
    def submit_order_call(self, sym, amount, sigma, orderType):
        #Depending on what we are planning to do with the order, change the amount we are handling
        if orderType == 'sell':
            #As we are selling, make sure we sell all of the stock.
            positions = self.api.list_positions()
            positions = [pos.symbol for pos in positions]

            if sym in positions:
                position = self.api.get_position(sym)
                amount = position.qty
            else:
                return None #This prevents us from selling stocks we do not have.
        else:
            #Make sure we are buying no more than 25 stocks at a time.
            if amount > 25:
                amount = 25

        #Submit the order.
        self.api.submit_order(
            symbol=sym,
            qty=amount,
            side=orderType,
            type='trailing_stop',
            trail_price=sigma,
            time_in_force='day'
        )

def trade_bot():
    #Generate the appropriate information.
    api = tradeapi.REST('<CLIENT_ID>', '<CLIENT_SECRET>', "https://paper-api.alpaca.markets", 'v2')
    misc = Miscellaneous(api)

    #This is our portfolio with the stocks we wish to buy.
    """
    The percentage is found by the following: 30% Dividend, 20% Growth, 5% Safe Haven, 5% Consumer, 20% Index/ETF, 20% Tech.
    After Identifying the stocks that you are looking for under these categorys, divided the percentage for the category by
    the number of stocks and that is the percentage per stock.
    """
    symbol = [{"token": "<symbol>", "percent": 0.0}]
    predictors = [{"symbol": sym['token'], "closing": m.get_info(sym['token']), "percent": sym['percent']} for sym in symbol]
    
    #Make new predictions.
    for i in range(len(predictors)):
        closing = predictors[i]['closing']
        stockSym, stockPercent = predictors[i]['symbol'], predictors[i]['percent']
        periodData, sma, bb, rsi = [], [], [], []
        
        bbPred = BollingerBands(20)
        for val in closing:
            #Calculate the sma.
            periodData.append(val)
            periodSubset = periodData[len(periodData)-14:]
            currentSMA = sum(periodSubset)/len(periodSubset)
            sma.append(currentSMA)
            
            #Handle the rsi calculation.
            n = 14
            if len(sma) >= n:
                deltas = np.diff(periodData)
                seed = deltas[:n+1]
                up, down = seed[seed >= 0].sum()/n, -seed[seed < 0].sum()/n
                rs = up/down
                rsi = np.zeros_like(periodData)
                rsi[:14] = 100. - 100./(1.+rs)

                for j in range(n, len(periodData)):
                    delta = deltas[j-1]  # The diff is 1 shorter

                    if delta > 0:
                        upval, downval = delta, 0.
                    else:
                        upval, downval = 0., -delta

                    up = (up*(n-1) + upval)/n
                    down = (down*(n-1) + downval)/n

                    rs = up/down
                    rsi[j] = 100. - 100./(1.+rs)

            #Handle the bollinger bands.
            bbY = bbPred.perform(val)
            if bbY is not None:
                #Store the values into the appropriate arrays.
                bb.append(bbY['sigma'])

        #Before identifying the extreme within sma, we need to make sure the dataset are the same length as each other.
        sma, rsi = sma[19:], rsi[6:]
        
        #Find the subset of rsi and sma data.
        idxs = misc.find_extrema(sma)
        idx = idxs[-1]
        sma, bb, rsi = sma[idx:], bb[idx:], rsi[idx:]

        #Find the angle between the subtrends (fixed onto the last element of the trend) and the overall trend of the subset. If the angle is increasing, then there is an increased likelihood of a trend reversal.
        delta = len(sma)-1
        m = {"sma": (sma[-1] - sma[0])/delta, "bb": (bb[-1] - bb[0])/delta, "rsi": (rsi[-1] - rsi[0])/delta}
        b = {"sma": sma[-1] - m['sma']*delta, "bb": bb[-1] - m['bb']*delta, "rsi": rsi[-1] - m['rsi']*delta}

        angles, recent = [], {"m": m, "b": b} #recent = overall
        for i in range(1, len(sma)-1):
            smaM = (sma[-1] - sma[i])/(len(sma)-1 - i)
            angles.append(math.atan(recent['m']['sma'] - smaM))

        anglesTrend = None
        if len(angles) > 1:
            anglesTrend = {"m": (angles[-1] - angles[0])/(len(angles)-1), "b": 0}
            anglesTrend['b'] = angles[-1] - anglesTrend['m']*(len(angles)-1)
        elif len(angles) == 1:
            anglesTrend = {"m": 0, "b": angles[-1]}
        elif len(angles) == 0:
            anglesTrend = {"m": 0, "b": 0}

        #Retrieve the orders we have made in our portfolio.
        positions = api.list_positions()
        positions = [pos.symbol for pos in positions]
        isInPositions = stockSym in positions

        #Make the trades that we would like.
        #Identify the amount that we would like to buy or sell.        
        amount = math.floor((misc.get_acc_amount() * stockPercent)/closing[-1])

        #Submit an order when one have an RSI divergence.
        #Only sell when we have something to sell and buy when we do not have any open filled orders.
        if anglesTrend['m'] <= 0:
            misc.submit_order(stockSym, amount, bb[-1], recent['m']['sma'], recent['m']['rsi'], isInPositions, 'sell', 'buy')
        else:
            misc.submit_order(stockSym, amount, bb[-1], recent['m']['sma'], recent['m']['rsi'], isInPositions, 'buy', 'sell')

if __name__ == '__main__':
    trade_bot()