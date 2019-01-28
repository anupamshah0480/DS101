import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
data = pd.read_csv('Social_Network_Ads.csv')

X = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values

#splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train =sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

#Fitting the LR model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

#predicting the test data
y_pred = classifier.predict(X_test)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)

#Visualising
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start= X_set[:,0].min() - 1, stop= X_set[:,0].max() + 1, step=0.01),
                     np.arange(start= X_set[:,1].min() - 1, stop= X_set[:,1].max() + 1, step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), alpha=0.75,cmap=ListedColormap(('yellow','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c= ListedColormap(('red','green'))(i), label=j)
plt.title('LogisticRegression (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start= X_set[:,0].min() - 1, stop= X_set[:,0].max() + 1, step=0.01),
                     np.arange(start= X_set[:,1].min() - 1, stop= X_set[:,1].max() + 1, step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), alpha=0.75,cmap=ListedColormap(('yellow','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c= ListedColormap(('red','green'))(i), label=j)
plt.title('LogisticRegression (Testing Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()