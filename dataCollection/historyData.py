import requests
import csv
import pandas as pd
import os

#'5WD48EGZB5ENKBZU'
#'EDKKIQSZOFAQEXQB',
API_KEY = 'EDKKIQSZOFAQEXQB'
workingDir = os.getcwd()
#INTERVAL = '1min'  # 1min, 5min, 15min, etc.

#gets the income sheet of the input stock 
def get_income_sheet(symbol):
    url = 'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol='+ symbol +'&apikey=EDKKIQSZOFAQEXQB'
    r = requests.get(url)
    incomesheet = r.json()
    incomesheet.pop('symbol')
    incomesheet.pop('annualReports')
    incomesheet = list(incomesheet.values())
    incomesheet = incomesheet[0]
    for x in incomesheet:

        rev = x.pop('totalRevenue')
        prof = x.pop('grossProfit')
        date = x.pop('fiscalDateEnding')
        x.clear()
        x['date'] = date
        x['totalRevenue'] = rev
        x['grossProfit'] = prof
        x['symbol'] = symbol
     
    
    with open(workingDir + '/bot/StockCSV/RevenueData/' + symbol + 'RevenueData.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['date','totalRevenue','grossProfit','symbol'])
        writer.writeheader()
        writer.writerows(incomesheet)
    
#gets the earnings data of the input stock
def get_earnings(symbol):
    url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol='+ symbol +'&apikey=EDKKIQSZOFAQEXQB'
    r = requests.get(url)
    earnings = r.json()
    earnings.pop('symbol')
    earnings.pop('annualEarnings')
    earnings = list(earnings.values())
    earnings = earnings[0]
    for x in earnings:
        date = x.pop('fiscalDateEnding')
        reportDate = x.pop('reportedDate')
        epsRep = x.pop('reportedEPS')
        epsEst = x.pop('estimatedEPS')
        sur = x.pop('surprisePercentage')
        x.clear()
        x['date'] = date
        x['reportDate'] = reportDate
        x['estimatedEPS'] = epsEst
        x['actualEPS'] = epsRep
        x['percentDif'] = sur
        x['symbol'] = symbol

    with open(workingDir + '/bot/StockCSV/EPSData/' + symbol + 'EPSData.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['date','reportDate','estimatedEPS','actualEPS','percentDif','symbol'])
        writer.writeheader()
        writer.writerows(earnings)

#get_income_sheet('AAPL')

def adjustedSplits(symbol):
    url = 'https://www.alphavantage.co/query?function=SPLITS&symbol=' + symbol +'&apikey={API_KEY}'
    r = requests.get(url)
    data = r.json()
    return (list(data.values())[1])


#gets the historical stock data with all info of the input stock
def get_stock_data(symbol):
    URL = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={API_KEY}&datatype=json'
    response = requests.get(URL)
    data = response.json()
    
    data = data.pop('Time Series (Daily)')
    days = list(data.keys())
    values = list(data.values())

    for x in range(len(values)):
        values[x].update({'date': days[x]})
        values[x]['open'] = values[x].pop('1. open')
        values[x]['high'] = values[x].pop('2. high')
        values[x]['low'] = values[x].pop('3. low')
        values[x]['close'] = values[x].pop('4. close')
        values[x]['volume'] = values[x].pop('5. volume')


    with open(workingDir + '/bot/StockCSV/HistData/' + symbol + 'HistData.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['date','open','high','low','close','volume'])
        writer.writeheader()
        writer.writerows(values)

    get_earnings(symbol)
    get_income_sheet(symbol)

    path3 = workingDir + "/bot/StockCSV/EPSData/" +symbol+"EPSData.csv"
    path4 = workingDir + "/bot/StockCSV/RevenueData/" +symbol+"RevenueData.csv"

    #reading it using pandas
    EPSData = pd.read_csv(path3)
    RevenueData = pd.read_csv(path4) 

    RevenueData.reset_index(inplace = True)  
    EPSData.reset_index(inplace = True) 
    finData = EPSData.merge(RevenueData, left_on=['index'], right_on=['index'])

    #path of the csv data file
    path2 = workingDir + "/bot/StockCSV/HistData/" + symbol + "HistData.csv"

    #reading it using pandas
    histData = pd.read_csv(path2)

    finData = finData.iloc[::-1]
    finData = finData.reset_index()

    finData = finData.drop(columns  = ['index','date_y','symbol_x','date_x','symbol_y','level_0'])
    finData = finData.rename(columns = {'reportDate':'date'})

    row = {
        'date':[histData.iat[0,0]],
        'estimatedEPS':[finData.iat[(len(finData)-1),1]],
        'actualEPS':[finData.iat[(len(finData)-1),2]],
        'percentDif':[finData.iat[(len(finData)-1),3]],
        'totalRevenue':[finData.iat[(len(finData)-1),4]],
        'grossProfit':[finData.iat[(len(finData)-1),5]]
    }
    df = pd.DataFrame(row)
    finData =  pd.concat([finData, df])

    histData = histData.iloc[::-1]

    histData = histData.reset_index()
    finData.set_index(pd.DatetimeIndex(finData.date),inplace=True)
    histData.set_index(pd.DatetimeIndex(histData.date),inplace=True)
    finData.pop('date')
    histData.pop('date')
    #fill the gaps
    finData['totalRevenue'] = pd.to_numeric(finData['totalRevenue'],errors = 'coerce')
    finData['grossProfit'] = pd.to_numeric(finData['grossProfit'],errors = 'coerce')
    finData = finData.resample('1D').mean().ffill()
    finData = histData.merge(finData, left_on=['date'], right_on=['date'])
    finData = finData.drop(columns = ['index'])
    finData.to_csv(workingDir + '/bot/StockCSV/FinHistData/' + symbol +'finHistData.csv')

    finData = pd.read_csv(workingDir + '/bot/StockCSV/FinHistData/' + symbol +'finHistData.csv')

    splits = adjustedSplits(symbol)
    keyVal = {}
    for x in range(len(splits)-1, -1, -1):
        keyVal[list(splits[x].values())[0]] = list(splits[x].values())[1]  

    keyVal[finData['date'].iloc[-1]] = 1 

    df2 = finData.copy()
    df2 = df2.iloc[0:0]

    for i in range((len(keyVal)-1)):
        dte1 = list(keyVal.keys())[i]
        dte2 = list(keyVal.keys())[i+1]
        splitVal  = (list(keyVal.values())[i+1])
        filtered_df = (finData.loc[(finData['date'] >= dte1)
                                        & (finData['date'] < dte2)])
        df2 = pd.concat([df2, filtered_df], ignore_index=True)
        tags = ['open','close','high','low']
        for x in tags:
            df2[x] = round((df2[x].div(float(splitVal))),3)
    df2 = pd.concat([df2, finData.tail(1)], ignore_index = True)
    df2.to_csv(workingDir + '/bot/StockCSV/FinHistData/Adjusted/' + symbol + 'adjFinHistData.csv')
    
get_stock_data('CRM')