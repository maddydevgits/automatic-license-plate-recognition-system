import pandas as pd

def readFromDatabase(k):
    data=pd.read_csv('dataset.csv')
    temp=data.iloc[:,:].values
    for i in temp:
        #print(i)
        for j in i:
            if(j==k):
                return (i)

#print(readFromDatabase('AP 27 Y 0778'))