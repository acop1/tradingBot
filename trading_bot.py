from datetime import datetime
import pandas as pd
import numpy as np
import os
from lumibot.backtesting.yahoo_backtesting import YahooDataBacktesting
from lumibot.brokers.alpaca import Alpaca
from lumibot.strategies.strategy import Strategy
from lumibot.traders.trader import Trader
from timedelta import Timedelta
from alpaca_trade_api.rest import REST
from finbert_utils import estimate_sentiment
from algorithimicTrading import msft_rf
from algorithimicTrading import lstm


Base_Url = "https://paper-api.alpaca.markets"
api_key = "Enter API Key"
api_secret =  "Enter API Secret Key"

ALPACA_CONFIG = {
    "API_KEY": api_key,
    "API_SECRET": api_secret,
    "PAPER": True,
    "ENDPOINT" : Base_Url
}

wd = os.getcwd()
start = datetime(2023,1,1)
end = datetime(2024,8,29)



class BuyHold(Strategy):
    symbol = "SPY"
    def initialize(self):
        self.sleeptime = "24H"

    def on_trading_iteration(self):
        if self.first_iteration:
            price = self.get_last_price(BuyHold.symbol)
            quantity = self.cash // price
            order = self.create_order(BuyHold.symbol, quantity, "buy")
            self.submit_order(order)



class MLTrader(Strategy):
    symbol = "SPY"
    cash_at_risk = 0.5

    def initialize(self):
        self.symbol = MLTrader.symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.cash_at_risk = MLTrader.cash_at_risk
        self.api = REST(base_url=Base_Url, key_id=api_key,secret_key=api_secret)

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity
    
    def get_dates(self):
        today = self.get_datetime()
        tri_day_news = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), tri_day_news.strftime('%Y-%m-%d')

    def get_sentiment(self):
        today, tri_day =  self.get_dates()
        news = self.api.get_news(symbol = self.symbol, start = tri_day, end = today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        if cash > last_price and quantity > 0:
            if sentiment == "positive" and probability > 0.99:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type = "bracket",
                    take_profit_price = last_price*1.20,
                    stop_loss_price = last_price*0.95
                )
                self.submit_order(order)
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > 0.99:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type = "bracket",
                    take_profit_price = last_price*0.8,
                    stop_loss_price = last_price*1.05
                )
                self.submit_order(order)
                self.last_trade = "sell"



#this will implement the randomForest model into the trading strategy to make decisions.
class RFTrader(Strategy):
    symbol = "SPY"
    cash_at_risk = 0.5
    predictions = []
    counter = 0

    def initialize(self):
        self.sleeptime = "1D"
        self.symbol = RFTrader.symbol
        self.last_trade = None
        self.cash_at_risk = RFTrader.cash_at_risk

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity
    
    def get_dates(self):
        today = self.get_datetime()
        return today.strftime('%Y-%m-%d')
    
    def is_closest_to_end(self):
        today = pd.Timestamp(self.get_dates())
        end_date = pd.Timestamp(end.strftime("%Y-%m-%d"))
        num_business_days = np.busday_count(today.date(), end_date.date())
        return (num_business_days <= 1)
    
    def get_data(self):
        today = self.get_dates()
        data, test = msft_rf.prep_data(RFTrader.symbol, today)
        return data, test

    def get_prediction(self):
        data, test = self.get_data()
        pred = msft_rf.predict_rf(test, RFTrader.symbol)
        return pred

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        pred_move = self.get_prediction()
        data, test = self.get_data()
        RFTrader.predictions.append(pred_move)

        if cash > last_price and quantity > 0:
            if pred_move == 0:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type = "bracket",
                    take_profit_price = last_price*1.25,
                    stop_loss_price = last_price*0.97
                )
                RFTrader.counter += 1
                self.submit_order(order)
                self.last_trade = "buy"

        #not sure if this works well
        if self.is_closest_to_end():
                length = len(RFTrader.predictions) * -1
                target = data.copy()['target'].iloc[length:]
                RFTrader.predictions = pd.Series(RFTrader.predictions, index=target.index)
                print("\n\nThe random forest model from " + start.strftime("%Y-%m-%d") + 
                      " to " + end.strftime("%Y-%m-%d") + " was: " 
                      + str(msft_rf.get_prediction_score(target, RFTrader.predictions)) 
                      + " percent correct")
                print("\nNumber of buy trades executed: " + str(RFTrader.counter))
                print("\n")      



#this will implement the randomForest model into the trading strategy to make decisions.
#It uses moving averages instead of financial data and just the close date to make decisions
class RFMATrader(Strategy):
    symbol = "SPY"
    cash_at_risk = 0.5
    predictions = []
    counter = 0

    def initialize(self):
        self.sleeptime = "1D"
        self.symbol = RFMATrader.symbol
        self.last_trade = None
        self.cash_at_risk = RFMATrader.cash_at_risk

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity
    
    def get_dates(self):
        today = self.get_datetime()
        return today.strftime('%Y-%m-%d')
    
    def is_closest_to_end(self):
        today = pd.Timestamp(self.get_dates())
        end_date = pd.Timestamp(end.strftime("%Y-%m-%d"))

        num_business_days = np.busday_count(today.date(), end_date.date())
        return (num_business_days <= 1)
    
    def get_data(self):
        today = self.get_dates()
        data, test, predictors = msft_rf.horizon_prep(RFMATrader.symbol, today)
        return data, test, predictors

    def get_prediction(self):
        data, test, predictors = self.get_data()
        pred = msft_rf.predict_rfma(test, RFMATrader.symbol, predictors)
        return pred

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        pred_move = self.get_prediction()
        data, test, predictors = self.get_data()
        RFMATrader.predictions.append(pred_move)

        if cash > last_price and quantity > 0:
            if pred_move == 0:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type = "bracket",
                    take_profit_price = last_price*1.25,
                    stop_loss_price = last_price*0.97
                )
                RFMATrader.counter += 1
                self.submit_order(order)
                self.last_trade = "buy"

        #not sure if this works well
        if self.is_closest_to_end():
                length = len(RFMATrader.predictions) * -1
                target = data.copy()['target'].iloc[length:]
                RFMATrader.predictions = pd.Series(RFMATrader.predictions, index=target.index)
                print("\n\nThe random forest model from " + start.strftime("%Y-%m-%d") 
                      + " to " + end.strftime("%Y-%m-%d") + " was: " 
                      + str(msft_rf.get_prediction_score(target, RFMATrader.predictions))
                      + " percent correct")
                print("\nNumber of buy trades executed: " + str(RFMATrader.counter))
                print("\n")



class LSTMTrader(Strategy):
    symbol = "SPY"
    cash_at_risk = 0.5
    predictions = []
    counter = 0

    def initialize(self):
        self.sleeptime = "1D"
        self.symbol = LSTMTrader.symbol
        self.last_trade = None
        self.cash_at_risk = LSTMTrader.cash_at_risk

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity
    
    def get_dates(self):
        today = self.get_datetime()
        return today.strftime('%Y-%m-%d')
    
    def is_closest_to_end(self):
        today = pd.Timestamp(self.get_dates())
        end_date = pd.Timestamp(end.strftime("%Y-%m-%d"))
        num_business_days = np.busday_count(today.date(), end_date.date())
        return (num_business_days <= 1)

    def get_data(self):
        today = self.get_dates()
        train_loader, test_loader, X_test, data = lstm.prep_lstm_data(LSTMTrader.symbol, today)
        return data

    def get_prediction(self):
        today = self.get_dates()
        prob, pred  = lstm.predict_lstm(LSTMTrader.symbol)
        return pred

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        pred_move = self.get_prediction()
        data = self.get_data()
        LSTMTrader.predictions.append(pred_move)

        if cash > last_price and quantity > 0:
            if pred_move == 0:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type = "bracket",
                    take_profit_price = last_price*1.25,
                    stop_loss_price = last_price*0.97
                )
                LSTMTrader.counter += 1
                self.submit_order(order)
                self.last_trade = "buy"

            if self.is_closest_to_end():
                length = len(LSTMTrader.predictions) * -1
                target = data.copy()['target'].iloc[length:]
                LSTMTrader.predictions = pd.Series(LSTMTrader.predictions, index=target.index)
                print("\n\nThe LSTM model from " + start.strftime("%Y-%m-%d") + 
                      " to " + end.strftime("%Y-%m-%d") + " was: " 
                      + str(msft_rf.get_prediction_score(target, LSTMTrader.predictions)) 
                      + " percent correct")
                print("\nNumber of buy trades executed: " + str(LSTMTrader.counter))
                print("\n")



#This trader will use the size of the movement and the direction to make a decision on the trade
class XGBDeltaTrader(Strategy):
    symbol = "SPY"
    cash_at_risk = 0.5
    predictions = []
    counter = 0
    delta_preds = []

    def initialize(self):
        self.sleeptime = "1D"
        self.symbol = XGBDeltaTrader.symbol
        self.last_trade = None
        self.cash_at_risk = XGBDeltaTrader.cash_at_risk

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity
    
    def get_dates(self):
        today = self.get_datetime()
        return today.strftime('%Y-%m-%d')
    
    def is_closest_to_end(self):
        today = pd.Timestamp(self.get_dates())
        end_date = pd.Timestamp(end.strftime("%Y-%m-%d"))
        num_business_days = np.busday_count(today.date(), end_date.date())
        return (num_business_days <= 1)
    
    def get_data_rf(self):
        today = self.get_dates()
        data, test = msft_rf.prep_data(XGBDeltaTrader.symbol, today)
        return data, test

    def get_prediction_rf(self):
        data, test = self.get_data_rf()
        pred = msft_rf.predict_rf(test, XGBDeltaTrader.symbol)
        return pred
    
    def get_data_xgb(self):
        today = self.get_dates()
        data, test = msft_rf.prep_data_delta(XGBDeltaTrader.symbol, today)
        return data, test
    
    def get_prediction_xgb(self):
        data, test = self.get_data_xgb()
        pred = msft_rf.predict_xgb_delta(test, XGBDeltaTrader.symbol)
        return pred

    #This strategy will use different values for the minimum price movement and the direction of the price to create a trade. I will try 1,2,3,5,10% movements in both
    #directions to find the best results. 
    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        pred_move = self.get_prediction_rf()
        pred_delta = self.get_prediction_xgb()
        data, test = self.get_data_rf()
        per_chng = ((((last_price - pred_delta) - last_price) / last_price)*100)
        
        XGBDeltaTrader.delta_preds.append(per_chng)
        XGBDeltaTrader.predictions.append(pred_move)

        if cash > last_price and quantity > 0:
            if per_chng > 0.2:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type = "bracket",
                    take_profit_price = last_price*(1 + (5*(per_chng/100))),
                    stop_loss_price = last_price*(1 - (per_chng/100))
                )
                XGBDeltaTrader.counter += 1
                self.submit_order(order)
                self.last_trade = "buy"

        #not sure if this works well
        if self.is_closest_to_end():
                length = len(XGBDeltaTrader.predictions) * -1
                target = data.copy()['target'].iloc[length:]
                XGBDeltaTrader.predictions = pd.Series(XGBDeltaTrader.predictions, index=target.index)
                print("\n\nThe random forest model from " + start.strftime("%Y-%m-%d") + 
                      " to " + end.strftime("%Y-%m-%d") + " was: " 
                      + str(msft_rf.get_prediction_score(target, XGBDeltaTrader.predictions)) 
                      + " percent correct")
                print("\nNumber of buy trades executed: " + str(XGBDeltaTrader.counter))
                print("\n")  

    

#Backtests the buy and hold strategy and the ml news strategy
#does not work when the market is not live but backtesting works all the time
def trade_test(strategy_type:str, trade:bool, symbol:str,cash_at_risk:float):
    if strategy_type == "bh":
        BuyHold.symbol = symbol
        if trade:
            broker = Alpaca(ALPACA_CONFIG)
            strategy = BuyHold(broker=broker)
            trader = Trader()
            trader.add_strategy(strategy)
            trader.run_all()
        else:
            #instance = BuyHold()
            BuyHold.backtest(
                YahooDataBacktesting,
                start,
                end
            )

    elif strategy_type == "mltrader":
        MLTrader.symbol = symbol
        MLTrader.cash_at_risk = cash_at_risk
        if trade:
            broker = Alpaca(ALPACA_CONFIG)
            strategy = MLTrader(broker=broker, 
                                parameters={"symbol":symbol, 
                                            "cash_at_risk":cash_at_risk})
            trader = Trader()
            trader.add_strategy(strategy)
            trader.run_all()
        else:
            MLTrader.backtest(
                YahooDataBacktesting,
                start,
                end,
                parameters={"symbol":symbol, 
                            "cash_at_risk":cash_at_risk}
            )

    elif strategy_type == "rftrader":
        RFTrader.symbol = symbol
        RFTrader.cash_at_risk = cash_at_risk
        if trade:
            broker = Alpaca(ALPACA_CONFIG)
            strategy = RFTrader(broker=broker, 
                                parameters={"symbol":symbol, 
                                            "cash_at_risk":cash_at_risk})
            trader = Trader()
            trader.add_strategy(strategy)
            trader.run_all()
        else:
            RFTrader.backtest(
                YahooDataBacktesting,
                start,
                end,
                parameters={"symbol":symbol, 
                            "cash_at_risk":cash_at_risk}
            )

    elif strategy_type == "rfmatrader":
        RFMATrader.symbol = symbol
        RFMATrader.cash_at_risk = cash_at_risk
        if trade:
            broker = Alpaca(ALPACA_CONFIG)
            strategy = RFMATrader(broker=broker, 
                                parameters={"symbol":symbol, 
                                            "cash_at_risk":cash_at_risk})
            trader = Trader()
            trader.add_strategy(strategy)
            trader.run_all()
        else:
            RFMATrader.backtest(
                YahooDataBacktesting,
                start,
                end,
                parameters={"symbol":symbol, 
                            "cash_at_risk":cash_at_risk}
            )

    elif strategy_type == "lstmtrader":
        LSTMTrader.symbol = symbol
        LSTMTrader.cash_at_risk = cash_at_risk
        if trade:
            broker = Alpaca(ALPACA_CONFIG)
            strategy = LSTMTrader(broker=broker, 
                                parameters={"symbol":symbol, 
                                            "cash_at_risk":cash_at_risk})
            trader = Trader()
            trader.add_strategy(strategy)
            trader.run_all()
        else:
            LSTMTrader.backtest(
                YahooDataBacktesting,
                start,
                end,
                parameters={"symbol":symbol, 
                            "cash_at_risk":cash_at_risk}
            )

    elif strategy_type == "xgbdelta":
        XGBDeltaTrader.symbol = symbol
        XGBDeltaTrader.cash_at_risk = cash_at_risk
        if trade:
            broker = Alpaca(ALPACA_CONFIG)
            strategy = XGBDeltaTrader(broker=broker, 
                                parameters={"symbol":symbol, 
                                            "cash_at_risk":cash_at_risk})
            trader = Trader()
            trader.add_strategy(strategy)
            trader.run_all()
        else:
            XGBDeltaTrader.backtest(
                YahooDataBacktesting,
                start,
                end,
                parameters={"symbol":symbol, 
                            "cash_at_risk":cash_at_risk}
            )
        pass

trade_test("xgbdelta",False,"CRM",0.25)

#create a masked self attention model to predict the daily price of stocks

"""
testing AMZN stock for 2023 calendar year
BH = 80% returns with monthly returns of 
jan - 18.37
feb - -7.93
mar - 9.68
apr - -.068
may - 18.94
jun - 7.27
jul - 3.15
aug - 4.42
sep - -8.07
oct - 4.49
nov - 8.06
dec - 5.76

ML = -.7.07% returns with monthly returns of 
jan - -2.5
feb - 2.38
mar - -4.76
apr - -2.9
may - 0
jun - 0
jul - 0
aug - -2.5
sep - 0
oct - 0
nov - 0.63
dec - 2.90

RF = 68.46% returns with monthly returns of 
jan - 11.94
feb - -4.48
mar - 6.11
apr - -0.69
may - 20.47
jun - 7.01
jul - 3.15
aug - 5.67
sep - -7.04
oct - -.17
nov - 7.89
dec - 5.52
Accuracy: 45.6
trades: 165

RFMA = 65.32% returns with monthly returns of 
jan - 11.94
feb - -4.48
mar - 6.11
apr - -0.69
may - 18.21
jun - 7.00
jul - 3.15
aug - 5.64
sep - -7.07
oct - -.07
nov - 7.89
dec - 5.52
Accuracy: 45.6
trades: 175

LSTM = 67.06% returns with monthly returns of 
jan - 10.85
feb - -3.02
mar - 5.34
apr - -0.73
may - 18.62
jun - 6.27
jul - 3.15
aug - 4.75
sep - -4.41
oct - -.24
nov - 7.74
dec - 5.44
Accuracy: 49.6
trades: 161
"""
