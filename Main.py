from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog

import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
import os
import joblib
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, classification_report)
main = tkinter.Tk()
main.title("Exploring the Impact of Zestimate on Real Estate Market Dynamics: A Case Study of Buyer, Seller, and Renter Perspectives")
main.geometry("1000x650")
main.config(bg='dark sea green')
title = tkinter.Label(main, text="Exploring the Impact of Zestimate on Real Estate Market Dynamics: A Case Study of Buyer, Seller, and Renter Perspectives",justify='center')

title.grid(column=0, row=0)
font=('times', 12, 'bold')
title.config(bg='midnight blue', fg='white')
title.config(font=font)
title.config(height=3,width=120)
title.place(x=50,y=5)

global filename
global x,y,x_train,x_test,y_train,y_test
global df

global path,labels,cnn_model

def upload():
    global path
    global df
    path  = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0',tkinter.END)
    df = pd.read_csv(path)
    text.insert(tkinter.END,path+'Loaded\n\n')
    text.insert(tkinter.END,df.head())

def preprocessing():
    global df,x,y,x_train,x_test,y_train,y_test,predict
    df.fillna(np.mean,inplace=True)
    df.drop_duplicates(inplace=True)
    # Create a count plot

# Plotting the count plot
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Regionname')
    plt.title('Count of number of regions')
    plt.xlabel('Region name')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
    le=LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = le.fit_transform(df[column].astype(str))
    x=df.iloc[:,0:20]
    y = df.iloc[:,-1]
    text.insert(tkinter.END,'\n\n----preprocessing------\n\n')
    text.insert(tkinter.END,'X-shape'+str(x.shape[0])+'  Y-shape'+str(y.shape[0]))
    
def splitting():
    global df,labels
    global x,y,x_train,x_test,y_train,y_test,predict
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state =42)
    text.insert(tkinter.END,'\n\n-------Splitting-----\n\n')
    text.insert(tkinter.END,"X-train"+str(x_train.shape)+ ", Y-train"+str(y_train.shape)) 
    text.insert(tkinter.END,"\n\n X-test"+str(x_test.shape)+ ", Y-test"+str(y_test.shape)) 

# Define empty lists to store metrics
#defining global variables to store accuracy and other metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to create a scatter plot
def scatter_plot(df, x_column, y_column):
    """Scatter plot for two numerical columns."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[x_column], y=df[y_column], alpha=0.6)
    plt.title(f'Scatter Plot: {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

# Function to create a KDE plot
def kde_plot(df, column):
    """KDE plot for numerical feature against target."""
    plt.figure(figsize=(8, 5))
    sns.kdeplot(df[column], fill=True)
    plt.title(f'KDE Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.show()

# Function to create a histogram
def histogram(df, column):
    """Histogram for numerical features."""
    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=30, alpha=0.7, color="blue")
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# Function to create a correlation heatmap
def correlation_heatmap(df):
    """Heatmap of correlation matrix for numerical columns."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

# Function to create a regression plot
def regression_plot(df, x_column, y_column):
    """Regression plot to see trend between numerical feature and target."""
    plt.figure(figsize=(8, 5))
    sns.regplot(x=df[x_column], y=df[y_column])
    plt.title(f'Regression Plot: {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

# Function to create a violin plot (for price distribution)
def violin_plot(df, column):
    """Violin plot for feature distribution."""
    plt.figure(figsize=(8, 5))
    sns.violinplot(y=df[column])
    plt.title(f'Violin Plot of {column}')
    plt.ylabel(column)
    plt.show()

# Function to create a line plot (trend over time)
def line_plot(df, x_column, y_column):
    """Line plot showing trends over time."""
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df[x_column], y=df[y_column])
    plt.title(f'Line Plot: {y_column} over {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.xticks(rotation=45)
    plt.show()

# Function to create a density plot
def density_plot(df, column):
    """Density plot for numerical feature."""
    plt.figure(figsize=(8, 5))
    sns.kdeplot(df[column], fill=True)
    plt.title(f'Density Plot of {column}')
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.show()

# Function to create a category vs mean property count bar plot
def category_vs_target(df, category_column, target_column):
    """Bar plot of mean target values for a categorical feature."""
    plt.figure(figsize=(8, 5))
    sns.barplot(x=df[category_column], y=df[target_column], estimator=lambda x: x.mean(), ci=None)
    plt.title(f'Bar Plot: {category_column} vs {target_column}')
    plt.xlabel(category_column)
    plt.ylabel(f'Average {target_column}')
    plt.xticks(rotation=45)
    plt.show()

# Function to create a longitude-latitude scatter plot (location analysis)
def location_scatter(df, lat_col, long_col, color_col):
    """Scatter plot for geographic data with color-coded property count."""
    plt.figure(figsize=(8, 5))
    plt.scatter(df[long_col], df[lat_col], c=df[color_col], cmap="coolwarm", alpha=0.5)
    plt.colorbar(label=color_col)
    plt.title(f'Geospatial Distribution of {color_col}')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def EDA():
    global path
    global df
    kde_plot(df, "Price")                                 # 2. KDE Plot
    histogram(df, "Distance")                             # 3. Histogram
    correlation_heatmap(df)                               # 4. Correlation Heatmap
    violin_plot(df, "Price")                              # 6. Violin Plot
    line_plot(df, "Date", "Propertycount")               # 7. Line Plot (Time Series)
    density_plot(df, "Lattitude")                        # 8. Density Plot
    category_vs_target(df, "Type", "Propertycount")      # 9. Categorical Bar Plot
    location_scatter(df, "Lattitude", "Longtitude", "Propertycount")  # 10. Location Scatter Plot


mae_list=[]
mse_list=[]
rmse_list=[]
r2_list=[]
def calculateMetrics(algorithm,predict, testY):
        # Regression metrics
        mae = mean_absolute_error(testY, predict)
        mse = mean_squared_error(testY, predict)
        rmse = np.sqrt(mse)
        r2 = r2_score(testY, predict)
        
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        
        print(f"{algorithm} Mean Absolute Error (MAE): {mae:.2f}")
        print(f"{algorithm} Mean Squared Error (MSE): {mse:.2f}")
        print(f"{algorithm} Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"{algorithm} R-squared (R²): {r2:.2f}")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=testY, y=predict, alpha=0.6)
        plt.plot([min(testY), max(testY)], [min(testY), max(testY)], 'r--', lw=2)  # Line of equality
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(algorithm)
        plt.grid(True)
        plt.show()
        text.insert(tkinter.END,'\n\n-------'+algorithm+'------------\n' )
        text.insert(tkinter.END,f"{algorithm} Mean Absolute Error (MAE): {mae:.2f}"+'\n')
        text.insert(tkinter.END,f"{algorithm} Mean Squared Error (MSE): {mse:.2f}"+'\n')
        text.insert(tkinter.END,f"{algorithm} Root Mean Squared Error (RMSE): {rmse:.2f}"+'\n')
        text.insert(tkinter.END,f"{algorithm} R-squared (R²): {r2:.2f}"+'\n')
    
# File path to save the model

def  KNN():
    if os.path.exists('KNeighborsRegressor.pkl'):
        # Load the trained model from the file
        knn = joblib.load('KNeighborsRegressor.pkl')
        print("Model loaded successfully.")
        predict = knn.predict(x_test)
        calculateMetrics("KNN Regressor", predict, y_test)
    else:
        # Train the model (assuming X_train and y_train are defined)
        knn = KNeighborsRegressor()
        knn.fit(x_train, y_train)
        # Save the trained model to a file
        joblib.dump(knn, 'KNeighborsRegressor.pkl')
        print("Model saved successfully.")
        predict = knn.predict(x_test)
        calculateMetrics("KNN Regressor", predict, y_test)

def ETR():
    if os.path.exists('ExtraTrees_model.pkl'):
        # Load the trained model from the file
        ex = joblib.load('ExtraTrees_model.pkl')
        print("Model loaded successfully.")
        predict = ex.predict(x_test)
        calculateMetrics("Extra Trees Regressor", predict, y_test)
    else:
        # Train the model (assuming X_train and y_train are defined)
        ex = ExtraTreesRegressor()
        ex.fit(x_train, y_train)
        # Save the trained model to a file
        joblib.dump(ex, 'ExtraTrees_model.pkl')
        print("Model saved successfully.")
        predict = ex.predict(x_test)
        calculateMetrics("Extra Trees Regressor", predict, y_test)
    
def compare():
#showing all algorithms performance values
    #showing all algorithms performance values
    columns = ["Algorithm Name","MSE","MAE","R2_Score"]
    values = []
    algorithm_names = ["KNN Regressor", "Extra Trees Regressor"]
    for i in range(len(algorithm_names)):
        values.append([algorithm_names[i],mse_list[i],mae_list[i],r2_list[i]])
        
    temp = pd.DataFrame(values,columns=columns)
    temp
    text.insert(tkinter.END,'\n\n----------------compare ata-------------\n\n')
    text.insert(tkinter.END,temp)
 
def predict():
    global path
    model_file = 'ExtraTrees_model.pkl'
    path  = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0',tkinter.END)
    test = pd.read_csv(path)
    newset= pd.read_csv(path)
    text.insert(tkinter.END,'\n\n----------------Test Data Loading-------------\n\n')
    text.insert(tkinter.END,str(test))
    test.fillna(np.mean,inplace=True)
    test.drop_duplicates(inplace=True)
    le=LabelEncoder()
    for column in test.columns:
        if test[column].dtype == 'object':
            test[column] = le.fit_transform(test[column].astype(str))
    etr=joblib.load(model_file)
    predict = etr.predict(test)
    text.insert(tkinter.END,'\n\n----------------Test Data output-------------\n\n')
    newset['Propertycount']=predict        
    text.insert(tkinter.END,str(newset.iloc[:,-2:])+'\n')
        
bgcolor='dark green'
textcolor='black'    
fg='white'
buttonwidth=200
font=('times', 15, 'bold')
uploadButton = Button(main, text="Upload Dataset",command=upload)
uploadButton.config(bg=bgcolor, fg=fg)
uploadButton.place(x=50,y=100,width=buttonwidth)
uploadButton.config(font=font)

uploadButton = Button(main, text="Preprocessing ",command=preprocessing)
uploadButton.config(bg=bgcolor, fg=fg)
uploadButton.place(x=50,y=180,width=buttonwidth)
uploadButton.config(font=font)

uploadButton = Button(main, text="EDA",command=EDA)
uploadButton.config(bg=bgcolor, fg=fg)
uploadButton.place(x=50,y=260,width=buttonwidth)
uploadButton.config(font=font)


uploadButton = Button(main, text="Data Splitting",command=splitting)
uploadButton.config(bg=bgcolor, fg=fg)
uploadButton.place(x=50,y=340,width=buttonwidth)
uploadButton.config(font=font)


uploadButton = Button(main, text="KNN Regression",command=KNN)
uploadButton.config(bg=bgcolor,fg=fg)
uploadButton.place(x=50,y=420,width=buttonwidth)
uploadButton.config(font=font)

uploadButton = Button(main, text="Extra Trees Regression",command=ETR)
uploadButton.config(bg=bgcolor,fg=fg)
uploadButton.place(x=50,y=500,width=buttonwidth)
uploadButton.config(font=font)

uploadButton = Button(main, text="predict",command=predict)
uploadButton.config(bg=bgcolor, fg=fg)
uploadButton.place(x=50,y=580,width=buttonwidth)
uploadButton.config(font=font)

font1 = ('times', 12, 'bold')
text=Text(main,height=26,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=300,y=100)
text.config(font=font1)

main.mainloop()
    