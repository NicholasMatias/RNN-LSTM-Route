# About

  Aimed to create a model capable of effectively forecasting and comparing the performance of economic sectors between countries using their stock indexes while also identifying the key driving features of economic growth. 

  Chose to analyze the stock sectors of the United States and India. The data used to train the model is from Yahoo Finance. Specifically, the model is trained using the closing prices of each sector. The first sector we are looking to analyze is the technology sector for both the United States and India. 
  
  Used pandas and numpy to prepare the data for the model. Data was split into training, validation, and testing data. The split was 80%, 10%, and 10%. The data spans from January 1st, 2017 to November 5th, 2023, Utilized keras' sequential model along with LSTM layer along and three dense layers. Though using the mean absolute error metric, it can be concluded that the model has an accuracy of 96.53%. 

  
