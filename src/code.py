import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from scipy import stats 
from sklearn import metrics 
from sklearn.metrics import mean_squared_error,mean_absolute_error, make_scorer,classification_report,confusion_matrix,accuracy_score,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import math
from sklearn import svm
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
%matplotlib inline
ete= pd.read_csv('/Users/rashm/mlops/bhutan_landslide_data.csv')
ete.head(15)
ete.info()
ete.shape
ete.describe
sns.countplot(x ="Code", hue ="Code",data = ete)
sns.countplot(x ="Code", hue ="Lithology",data = ete)
sns.countplot(x ="Code", hue ="Altitude",data = ete)
sns.countplot(x ="Code", hue ="Slope",data = ete)
sns.countplot(x ="Code", hue ="Total curvature",data = ete)
sns.countplot(x ="Code", hue ="Aspect",data = ete)
sns.countplot(x ="Code", hue ="Distance to road",data = ete)
sns.countplot(x ="Code", hue ="Distance to stream",data = ete)
sns.countplot(x ="Code", hue ="Slope length",data = ete)
sns.countplot(x ="Code", hue ="TWI",data = ete)
sns.countplot(x ="Code", hue ="STI",data = ete)
ete['Code'].value_counts(normalize=True)
ete.isnull()
sns.histplot(ete['Code']);
print("skewness: %f" %ete["Code"].skew())
print("Kurtosis: %f" %ete["Code"].kurt())
sns.countplot(x ="Code", hue ="Type",data = ete)
ete['Type']=ete['Type'].astype('category')
ete.dtypes
ete = ete.drop(['Type'], axis=1)
ete.head()
features=['Code','Lithology','Altitude','Slope','Total curvature','Aspect','Distance to road','Distance to stream','Slope length','TWI','STI']
X=ete.loc[:,features]
y=ete.loc[:,['Code']]
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, shuffle = True, random_state = 8)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,test_size=0.1, random_state= 8)
print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))
print("X_val shape: {}".format(y_train.shape))
print("y val shape: {}".format(y_test.shape))
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,  
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)
    from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}
rf = RandomForestClassifier()
rand_search = RandomizedSearchCV(rf, param_distributions = param_dist,n_iter=5, 
                                 cv=5)
rand_search.fit(X_train, y_train)
best_rf = rand_search.best_estimator_
print('Best hyperparameters:',  rand_search.best_params_)
from sklearn.metrics import ConfusionMatrixDisplay
y_pred = best_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
from sklearn.metrics import precision_score,recall_score
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
from sklearn.metrics import f1_score
f1score = f1_score(y_test, y_pred)
print("F1-score:", f1score)
X_train.to_csv("train.csv", index=True)
X_test.to_csv("test.csv", index=True)

