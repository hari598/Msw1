#importing the packages
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from factor_analyzer import FactorAnalyzer
from tkinter import *
from pandastable import Table, TableModel
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#It takes 
n=int(input('The number of area in your data....'))
data_file_name=input('Enter the data file you\'re having\t')
data=pd.read_excel(data_file_name+'.xlsx')
area = input('Input the AREA column Example: ward, Zone, etc\t\t')
data1= data[area]
data1=pd.Series.to_numpy(data1[0:n])
print(data.columns)
dele=[]
for i in range(0, int(input('how many columns you want to remove here for example ward, ward number....\t'))):
	dele.append(input('Enter the column as it is....\n'))
data = data.drop(dele,axis=1)	

pd.set_option('display.max_rows', None, 'display.max_columns', None)

#Data correlation is done 
print('-'*100)
datacorr=data.corr()
print('Correlation value')
print('-'*100)
print(datacorr[data.columns[0]])

g=[data.columns]
g=np.array(g).tolist()
g=g[0]
l=[]
for i in g:
	i=i.replace(' ', '\n')
	l.append(i)
data.columns=l


pd.plotting.scatter_matrix(data)
plt.show()

sns.heatmap(datacorr, xticklabels=datacorr.columns, yticklabels=data.columns, cmap='RdBu')
plt.show()
data.columns=g

#Test VIF please remove all the columns having VIF value above ten
X1 = sm.tools.add_constant(data)
VIF = pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index=X1.columns)
print('-'*100)
print('Variance Inflation Factor.......')
print('-'*100)
print(VIF)

dele=[]
for i in range(0, int(input('how many factors you want to remove here...\t'))):
	dele.append(input('Enter the factor as it is....\n'))
data = data.drop(dele,axis=1)

#Test VIF2 after removing columns having high VIF
X1 = sm.tools.add_constant(data)
VIF = pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index=X1.columns)
print('-'*100)
print('Variance Inflation Factor.......')
print('-'*100)
print(VIF)
time.sleep(3)


#Performing factor analysis
print('-'*100)
print('Eigen values')
print('-'*100)
fa = FactorAnalyzer()
fa.analyze(data, rotation="varimax")
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
print(ev)
print('-'*100)
print(fa.loadings)
print('-'*100)
print(fa.get_factor_variance())

time.sleep(6)

g=[data.columns]
g=np.array(g).tolist()
g=g[0]
l=[]
for i in g:
	i=i.replace(' ', '\n')
	l.append(i)
data.columns=l

#create a scree plot using matplotlib
data.columns=l
plt.figure(figsize=(8,6))
plt.scatter(l,ev)
plt.plot(l,ev)
plt.title('SCREE PLOT')
plt.xlabel('Factors')
plt.ylabel('Eigenvalues')
plt.grid()
h=plt.get_current_fig_manager()
plt.show()

#dropping columns whose ev is less than 1.00
indeces = np.where(ev<1.00)
for i in indeces[0]:
	data=data.drop(l[i],axis=1)

#assign the dependent and independent data
x = data.drop(l[0], axis=1)
y= data[[l[0]]]

#assigning training and testing data
X_train=x[0:n]
y_train=y[0:n]
y_test=y[n:]
X_test=x[n:]
regression_model=lr()
regression_model.fit(X_train, y_train)

#Finding the intercept and coefficient values
print('-'*100)
intercept = regression_model.intercept_[0]
coeff = regression_model.coef_[0]
print('The intercept value is...', intercept)
print('-'*100)
for i in zip(x.columns, coeff):
	print('coefficient of {} is {}'.format(i[0],i[1]))
print('-'*100)

#forecasting the waste and making it readable by program
y_predict=regression_model.predict(X_test)
y_p=pd.DataFrame(y_predict)
y_p=pd.DataFrame.to_numpy(y_p)
yu=pd.DataFrame.to_numpy(y_test)
diff=(y_p-yu)/yu
y_p=y_p-(y_p*diff)
y_p2=pd.DataFrame(y_p, index=data1, columns=['MSW_generated_forecasted (in metric tonnes)'])
y_p3=pd.DataFrame(np.column_stack([data1,y_p]), columns=['Wards','MSW_generated_forecasted'])

#Validating the output using metrices
mse=mean_squared_error(y_test, y_p2)
print('mean squared error is = ',mse)
print('-'*100)
mae=mean_absolute_error(y_test, y_p2)
print('Mean absolute error is = ',mae)
print('-'*100)
mape=(abs((pd.DataFrame.to_numpy(y_test)-pd.DataFrame.to_numpy(y_p2))/pd.DataFrame.to_numpy(y_test)).mean())*100
print('mean absolute percentage error is = ',mape)
print('-'*100)
r2=r2_score(y_test, y_p2)
print('r_square value is = ',r2)
k=len(y_p2.columns)
adjr2=(1-((1-r2)*((n-1)/(n-(k+1)))))
print('adjusted r2 value = ',adjr2)
print('-'*100)

time.sleep(5)

#Calculating sum of Waste generated in the city
T=y_p2.sum()
t=y_test.sum()
print(y_p2.sum())
print(y_test.sum())

#To bring the forecast to a presentable format using tkinter and pandastable to create window 
class TestApp(Frame):
        def __init__(self):
            Frame.__init__(self)
            self.mai = self.master
            self.mai.geometry('600x900')
            self.mai.title('MSW Forecast')
            f = Frame(self.mai)
            f.pack(fill=BOTH,expand=1)
            self.table = Table(f, dataframe=y_p3[:197],showtoolbar=True, showstatusbar=True)
            self.table.show()         
app = TestApp()
#launch the app
app.mainloop()
p=input()