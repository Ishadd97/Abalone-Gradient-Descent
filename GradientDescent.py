import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

log = open("log.txt", "a")

df = pd.read_csv('C:/Users/robo-/OneDrive/Documents/School/MachineLearning/HW1/GradientDescent/abalone/abalone.csv')
#print(df.head())
#df.info()
#print(df.describe())

df['Sex'].replace(['M','F',"I"], [0,1,2],inplace=True)
#print(df.head())

scaler = StandardScaler()
scaler.fit(df)
scaler.transform(df)

print(df.corr())
sns.scatterplot(data = df, x='Shell_weight',y='Rings')
plt.savefig('weightVsRings.png')
plt.clf()

X = df.drop(['Sex','Shucked_weight'], axis = 1)
X = np.c_[np.ones(X.shape[0]),X]
y = df['Rings']
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2,random_state=4)

def gradient_descent(x, y, m , theta, alpha):
    costList = []
    thetaList = []
    predList = []
    run = True
    costList.append(1e10)
    i=0
    while run:
        prediction = np.dot(x,theta)
        predList.append(prediction)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)
        costList.append(cost)
        theta = theta - (alpha*(1/m)*np.dot(x.T,error))
        thetaList.append(theta)
        #print(costList[i])
        #print(costList[i+1])
        if costList[i]-costList[i+1] < 1e-4:
            run = False
        i+=1
    costList.pop(0)
    return predList,costList,thetaList

# Parameters required for Gradient Descent
alpha = 0.0001    #learning rate
m = y.size  # no. of samples
np.random.seed(15)
theta = np.random.rand(X.shape[1])

prediction_list, cost_list, theta_list = gradient_descent(Xtrain, ytrain, m, theta, alpha)
theta = theta_list[-1]

ytestPredict = np.dot(Xtest, theta)
rmse = (np.sqrt(mean_squared_error(ytest, ytestPredict)))
r2 = r2_score(ytest, ytestPredict)

plt.title('Cost Function J', size = 30)
plt.xlabel('No. of iterations', size=20)
plt.ylabel('Cost', size=20)
plt.plot(cost_list)
plt.show()


#SGD Regressor
'''
model = SGDRegressor()
model.fit(Xtrain, ytrain)
ytestPredict = model.predict(Xtest)
rmse = (np.sqrt(mean_squared_error(ytest, ytestPredict)))
r2 = r2_score(ytest, ytestPredict)

log.write("Learning Rate: ")
log.write(str(alpha))
log.write("\n")
log.write("Theta: ")
log.write(str(theta))
log.write("\n")
log.write("MSE: ")
log.write(str(rmse))
log.write("\n")
log.write("r2: ")
log.write(str(r2))
log.write("\n")
print(r2)

log.write("\n")
'''
log.close()

