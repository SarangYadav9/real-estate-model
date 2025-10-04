print('-------------------------Real estate price prediction model------------------------------')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv('/Users/dexter/Documents/python/RealEstate.csv')
x = df[['area_sqft']]
y = df['price']
model = LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1, random_state=42)
model.fit(x_train, y_train)
print('1-> Show Datafame\n2-> Describe Dataframe\n3-> Predict Prices\n4-> Plot area vs price graph')
while True:
    ch = int(input("Enter your choice(1,2,3,4 or 0 to exit):"))
    if ch ==1:
        print(df.head())
    elif ch==2:
        print(df.describe())
    elif ch==3:
        new_house_area = float(input("Enter Area(in sqft):"))
        df1=pd.DataFrame([[new_house_area]],columns = ['area_sqft'])
        pred = model.predict(df1)
        print(f"The price of house for the area {new_house_area} is :",pred[0])
    elif ch==4:
        plt.scatter(x,y,color = 'red', label = "Actual Price")
        plt.plot(x,model.predict(x),color = 'blue',label= "Predicted Line")
        plt.title("House Area vs Price")
        plt.xlabel("Area in square feet")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()
    elif ch==0:
        print("Exiting Program. Goodbye!")
        break
    else:
        print("Invalid Choice!")
