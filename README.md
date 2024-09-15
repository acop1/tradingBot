# tradingBot
#StockCsv 
  #this file stores the historical stock data in separate files.
  #EPSData has the estimated and actual EPS data along with the percentage diff
  #FinHistData holds the open,close,high,low,and volume. it also appends the 
  #EPSData annd RevenueData to the price data
  #It also has the adjusted file which adjusts the prices for stock splits
  #Hist Data has the open,high,low,close,and volume price data.
  #RevenueData has quarterly total revenue and gross profit data.

#dataCollection
  #historyData.py
  #this file has the python program which gathers the data from an api. It cleans    
  #the information and stores it in the respective files mentioned above.

#algorithmicTrading
  #lstm.py
  #this file contains the specific data processing and training for the long short 
  #term memory neural network. 
  #msft_rf.py
  #this file contains the specific data processing and training on a random forest 
  #model using the values in AdjustedFinHistData. It also has a random forest model
  #which uses price moving averages to predict the directionality of the stock. 
  #It also contains the accuracy function which predicts how well the model 
  #performed. 
