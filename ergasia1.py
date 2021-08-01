#!/usr/bin/env python
# coding: utf-8

# # Data Challenge - Πρόβλεψη Πληρότητας Πτήσεων
# 
# Στα πλαίσια της εργασίας του μαθήματος "Εξόρυξη Γνώσης από Βάσεις Δεδομένων και τον Παγκόσμιο Ιστό", θα δουλέψετε πάνω σε ένα πρόβλημα κατηγοριοποίησης. Συγκεκριμένα, σας δίνεται ένα σύνολο δεδομένων το οποίο αποτελείται από μερικές χιλιάδες πτήσεις, όπου κάθε πτήση περιγράφεται απο ένα σύνολο μεταβλητών (αεροδρόμιο αναχώρησης, αεροδρόμιο άφιξης, κτλ). Κάθε πτήση χαρακτηρίζεται επίσης από μια μεταβλητή που σχετίζεται με τον αριθμό των επιβατών της πτήσης (π.χ. κάθε τιμή της μεταβλητής σχετίζεται με ενα εύρος πλήθους επιβατών). Για κάποιες πτήσεις, η τιμή της μεταβλητής  είναι γνωστή, ενώ για άλλες όχι. Στόχος σας είναι να προβλέψετε την τιμή της μεταβλητής για τις πτήσεις για τις οποίες δεν είναι διαθέσιμη.
# 
# ### Σύνολο Δεδομένων
# 
# Το αρχείο με όνομα `train.csv` περιέχει τα δεδομένα εκπαίδευσης (training set) του προβλήματος, ενώ το αρχείο `test.csv` περιέχει τα δεδομένα ελέγχου (test set) του προβλήματος. Κάθε γραμμή των δυο αυτών αρχείων αντιστοιχεί σε μια πτήση η οποία χαρακτηρίζεται από τις εξής μεταβλητές:
# 
# Μεταβλητή | Περιγραφή
# --- | --- 
# 0 DateOfDeparture | Ημερομηνία αναχώρησης
# 1 Departure | Κωδικός αεροδρομίου αναχώρησης
# 2 CityDeparture | Πόλη αναχώρησης   #το rfe δεν το επιλεγει αλλα ανεβαζει το f1 score
# 3 LongitudeDeparture 	 | Γεωγραφικό μήκος αεροδρομίου αναχώρησης
# 4 LatitudeDeparture 	 | Γεωγραφικό πλάτος αεροδρομίου αναχώρησης
# 5 Arrival | Κωδικός αεροδρομίου άφιξης
# 6 CityArrival | Πόλη άφιξης
# 7 LongitudeArrival | Γεωγραφικό μήκος αεροδρομίου άφιξης
# 8 LatitudeArrival | Γεωγραφικό πλάτος αεροδρομίου άφιξης
# 9 WeeksToDeparture | Πόσες εβδομάδες πριν την αναχώρηση της πτήσης κατά μέσο όρο έκλεισαν οι επιβάτες τα εισητήριά τους
# 10 std_wtd | Τυπική απόκλιση για το παραπάνω 
# 11 Το training set περιέχει μια επιπλέον μεταβλητή (`PAX`) η οποία έχει σχέση με τον αριθμό των επιβατών της πτήσης. Η μεταβλητή αυτή παίρνει 8 διαφορετικές τιμές (τιμές από 0 έως 7 οπότε 8 κατηγορίες συνολικά). Κάθε κατηγορία υποδηλώνει πόσοι περίπου επιβάτες χρησιμοποίησαν την πτήση. Οι αριθμοί στις κατηγορίες έχουν ανατεθεί με τυχαίο τρόπο, οπότε μην θεωρήσετε ότι η κατηγορία 0 υποδηλώνει πολύ λίγους επιβάτες ενώ η κατηγορία 7 πάρα πολλούς επιβάτες. Η μεταβλητή `PAX` λείπει από το test set καθώς πρόκειται για την μεταβλητή που πρέπει να προβλέψετε στα πλαίσια της παρούσας εργασίας.
# 12 DayOfDeparture
# 13 MonthOfDeparture
# 14 YearOfDeparture
# Παρακάτω σας δίνεται κώδικας ο οποίος φορτώνει τα δεδομένα εκπαίδευσης σε ένα DataFrame της βιβλιοθήκης Pandas και τυπώνει τις πρώτες 5 γραμμές. Οπότε μπορείτε να δείτε τις 12 μεταβλητές του προβλήματος.

# In[1]:


import pandas as pd


df_train = pd.read_csv('C:/Users/theodore/Desktop/train.csv')

df_train['DayOfDeparture'] = pd.to_datetime(df_train['DateOfDeparture']).dt.weekday
df_train['MonthOfDeparture'] = pd.to_datetime(df_train['DateOfDeparture']).dt.month
df_train['YearOfDeparture'] = pd.to_datetime(df_train['DateOfDeparture']).dt.year
df_train.head()

list(df_train)
# In[3]:


df_test = pd.read_csv('C:/Users/theodore/Desktop/test.csv')

df_test['DayOfDeparture'] = pd.to_datetime(df_test['DateOfDeparture']).dt.weekday
df_test['MonthOfDeparture'] = pd.to_datetime(df_test['DateOfDeparture']).dt.month
df_test['YearOfDeparture'] = pd.to_datetime(df_test['DateOfDeparture']).dt.year
#df_test.drop(['DateOfDeparture'], axis=1, inplace=True)
df_test.head()

list(df_test)
# %%

y_train = df_train[['PAX']]

from sklearn.model_selection import train_test_split
df_train, df_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.2, random_state=42)

df_train.drop(df_train.columns[[0,2,4,5,7,9,10,11,14]], axis=1, inplace=True)

list(df_train)

# %%

df_test.drop(df_test.columns[[0,2,4,5,7,9,10,11,14]], axis=1, inplace=True)
list(df_test)
#%%
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5)
for train_index, test_index in kf.split(df_train):
   
   X_train, X_test = df_train[train_index], df_train[test_index]
   y_train, y_test = y_train[train_index], y_train[test_index]
   print(X_train, X_test, y_train, y_test)
#for train_index, test_index in kf.split(df_train):
#X_train, X_test = df_train[train_index], df_train[test_index]
#y_train, y_test = y[train_index], y[test_index]

#%%
df_train.head()
# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df_train['Departure'])
df_train['Departure'] = le.transform(df_train['Departure'])
#df_train['Arrival'] = le.transform(df_train['Arrival'])
df_test['Departure'] = le.transform(df_test['Departure'])
#df_test['Arrival'] = le.transform(df_test['Arrival'])


lt = LabelEncoder()

lt.fit(df_train['CityArrival'])

#df_train['CityDeparture'] = lt.transform(df_train['CityDeparture'])
df_train['CityArrival'] = lt.transform(df_train['CityArrival'])

#df_test['CityDeparture'] = lt.transform(df_test['CityDeparture'])
df_test['CityArrival'] = lt.transform(df_test['CityArrival'])

lp = LabelEncoder()

#lp.fit(df_train['DateOfDeparture'])

#df_train['DateOfDeparture'] = lp.transform(df_train['DateOfDeparture'])
#df_test['DateOfDeparture'] = lp.transform(df_test['DateOfDeparture'])
#%%
import numpy as np

df_train['MonthOfDeparture'] = np.cbrt(df_train['MonthOfDeparture'])
df_test['MonthOfDeparture'] = np.cbrt(df_test['MonthOfDeparture'])
df_train['Departure_cbrt'] = np.cbrt(df_train['Departure'])
df_test['Departure_cbrt'] = np.cbrt(df_test['Departure'])
df_train['LongitudeDeparture'] = np.cbrt(df_train['LongitudeDeparture'])
df_test['LongitudeDeparture'] = np.cbrt(df_test['LongitudeDeparture'])
df_train['CityArrival'] = np.cbrt(df_train['CityArrival'])
df_test['CityArrival'] = np.cbrt(df_test['CityArrival'])
df_train['LatitudeArrival'] = np.cbrt(df_train['LatitudeArrival'])
df_test['LatitudeArrival'] = np.cbrt(df_test['LatitudeArrival'])
df_train['DayOfDeparture'] = np.cbrt(df_train['DayOfDeparture'])
df_test['DayOfDeparture'] = np.cbrt(df_test['DayOfDeparture'])

# %%

from sklearn.preprocessing import StandardScaler




sc = StandardScaler()
df_train_std=sc.fit_transform(df_train)
df_test_std=sc.transform(df_test)
# %%
from sklearn.preprocessing import MinMaxScaler

minmax= MinMaxScaler()
df_train=minmax.fit_transform(df_train)
df_test=minmax.transform(df_test)

#%%
from sklearn.preprocessing import Normalizer

nm=Normalizer()
df_train=nm.fit_transform(df_train)
df_test=nm.transform(df_test)
# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

X_train = df_train
X_test = df_test                                     #Αποδίδει καλύτερα με standardization
y_train = np.ravel(y_train)

#classifier=MLPClassifier(hidden_layer_sizes=(100,50,50),solver ='lbfgs' )                   #,solver ='lbfgs'

#classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=14,min_samples_leaf=3))
classifier = BaggingClassifier(DecisionTreeClassifier(max_depth=14,min_samples_leaf=3),n_estimators=12)
#classifier = DecisionTreeClassifier(max_depth=14,min_samples_leaf=3 )
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
y_pred.shape
#%%
from sklearn.metrics import f1_score
#y_test = np.loadtxt('test_labels.csv', delimiter=",", skiprows=1, usecols=[1])
f1_score(y_test, y_pred, average='micro')
# %%
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import numpy as np

# Load the digits dataset
digits = load_digits()

X_train = df_train
X_test = df_test                                     #Αποδίδει καλύτερα με standardization
y_train = np.ravel(y_train)

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X_train, y_train)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()

#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
import numpy as np
X_train = df_train
X_test = df_test                                     #Αποδίδει καλύτερα με standardization
y_train = np.ravel(y_train)

classifier = DecisionTreeClassifier()
# create the RFE model for the svm classifier 
# and select attributes
rfe = RFE(classifier, 3)
rfe = rfe.fit(X_train, y_train)
# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)
# %%

from sklearn.preprocessing import StandardScaler




sc = StandardScaler()
df_train_std=sc.fit_transform(df_train)
df_test_std=sc.transform(df_test)
#%%
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X_train = df_train
X_test = df_test                                     #Αποδίδει καλύτερα με standardization
y_train = np.ravel(y_train)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

classifier = MLPClassifier() 
 
# Use a grid over parameters of interest
param_grid = { 
           "hidden_layer_sizes" : [100,200,250,300,350,400,450,500]}
           
CV_rfc = GridSearchCV(estimator=classifier, param_grid=param_grid, cv= 10)
CV_rfc.fit(X_train, y_train)
print (CV_rfc.best_params_)

#%%
print(grid.best_params_)

#%%
print(grid.best_estimator_)


#%%
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

#train_sizes = [1, 100, 500, 1000,1500, 1780]
X_train = df_train
X_test = df_test                                     #Αποδίδει καλύτερα με standardization
y_train = np.ravel(y_train)
classifier=MLPClassifier(hidden_layer_sizes=(50))
#classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=14,min_samples_leaf=3))
#classifier = BaggingClassifier(DecisionTreeClassifier(max_depth=14,min_samples_leaf=3),n_estimators=12)
train_sizes, train_scores, validation_scores = learning_curve(classifier, X_train, y_train, cv = 5,scoring = 'accuracy', train_sizes=np.linspace(0.01, 1.0, 50))
#print('Training scores:\n\n', train_scores)
#%%
# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(validation_scores, axis=1)
test_std = np.std(validation_scores, axis=1)

#%%
import matplotlib.pyplot as plt
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

# %%

from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
#from xgboost import XGBClassifier
from sklearn import ensemble
import numpy as np

X_train = df_train
X_test = df_test                                     #Αποδίδει καλύτερα με standardization
y_train = np.ravel(y_train)
#classifier=XGBClassifier()
#classifier = ensemble.GradientBoostingClassifier()
#classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
#classifier = DecisionTreeClassifier()
#classifier=MLPClassifier(hidden_layer_sizes=(100,50,50),solver ='lbfgs' )
classifier = MLPClassifier()
#classifier = BaggingClassifier(DecisionTreeClassifier(max_depth=14,min_samples_leaf=3),n_estimators=12)

scores = cross_val_score(classifier, df_train, y_train, cv=10, scoring='accuracy')

print (scores.mean())

#%%
print (scores.mean())
#%%
print (scores.std())

#%%
from sklearn.neural_network import MLPClassifier
k_range = range(2,40)
k_scores=[]
classifier=MLPClassifier(hidden_layer_sizes=(100,50,50),solver ='lbfgs')
#for k in k_range:
#classifier = DecisionTreeClassifier(min_samples_leaf=k)
scores = cross_val_score(classifier, df_train, y_train, cv=10, scoring='accuracy')
k_scores.append(scores.mean())
    
print(k_scores)

#%%
import matplotlib.pyplot as plt
%matplotlib inline

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
# plt.plot(x_axis, y_axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
#%%
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X_train = df_train
X_test = df_test                                     #Αποδίδει καλύτερα με standardization
y_train = np.ravel(y_train)

classifier = DecisionTreeClassifier()


print (cross_val_score(classifier,X_train, y_train, cv=10, scoring='accuracy').mean())

# %%
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
X_train = df_train
X_test = df_test
y_train = np.ravel(y_train)

knn = KNeighborsClassifier(n_neighbors=28)

print (cross_val_score(knn,X_train, y_train, cv=10, scoring='accuracy').mean())


#%%
#feature_cols = ['DateOfDeparture', 'Departure', 'Arrival', 'WeeksToDeparture', 'std_wtd']

X=df_train[['DateOfDeparture', 'Departure','CityDeparture', 'LongitudeDeparture', 'LatitudeDeparture', 'Arrival', 'CityArrival', 'LongitudeArrival', 'LatitudeArrival', 'WeeksToDeparture', 'std_wtd' ]]

#y = df_train[['PAX']]

#%%
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=28)

scores = cross_val_score(classifier, X, y_train, cv=10, scoring='mean_squared_error')
print (scores)

#%%

mse_scores = -scores
#3.3207030975915615
rmse_scores = np.sqrt(mse_scores)

print (rmse_scores.mean())

# In[5]:
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np

y_train = df_train[['PAX']]

from sklearn.model_selection import train_test_split
df_train, df_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.2, random_state=42)

X_train = df_train
X_test = df_test
y_train = np.ravel(y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print (metrics.accuracy_score(y_test, y_pred))

#df_train.drop(df_train.columns[[2,3,4,5,7,8]], axis=1, inplace=True)
#df_train.drop(df_train.columns[[0,2,3,4,6,7,8,9,10,11]], axis=1, inplace=True)
#df_train.head()

#plt.scatter(df_train.values[:,0], df_train.values[:,1])

# Για να μπορεί να δουλέψει ένα αλγόριθμος ταξινόμησης είναι απαραίτητο το training και το test set να έχουν ακριβώς τον ίδιο αριθμό στηλών (ίδιες μεταβλητές). Συνεπώς, πρέπει και στο test set να διαγράψουμε όλες τις στήλες εκτός από τις Departure και Arrival. Αυτό γίνεται με τον παρακάτω κώδικα.


# In[6]:
from sklearn.model_selection import train_test_split

y_train = df_train[['PAX']]

df_train, df_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.2, random_state=42)

df_train.drop(df_train.columns[[2,3,4,6,7,8,11]], axis=1, inplace=True)

#df_test.drop(df_test.columns[[0,2,3,4,6,7,8]], axis=1, inplace=True)
df_test.drop(df_test.columns[[2,3,4,6,7,8]], axis=1, inplace=True)

df_test.head()


# Οι στήλες των training και test set περιέχουν κατηγορικές μεταβλητές των οποίων οι τιμές είναι αλφαριθμητικά. Οι αλγόριθμοι ταξινόμησης ωστόσο δουλεύουν μόνο με αριθμητικές τιμές. Χρησιμοποιούμε το αντικείμενο `LabelEncoder` του `scikit-learn` για να μετατρέψουμε τα αλφαριθμητικά σε αριθμητικές τιμές.

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np

X_train = df_train
X_test = df_test
y_train = np.ravel(y_train)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print (metrics.accuracy_score(y_test, y_pred))


# In[7]:



#plt.scatter(df_train.values[:,0], df_train.values[:,1])


# Έπειτα, εκπαιδεύουμε έναν ταξινομητή logistic regression για να προβλέψουμε τις κατηγορίες αριθμού επιβατών των δεδομένων ελέγχου. Επιπλέον, αποθηκεύουμε τις προβλέψεις μας στο αρχείο `y_pred.txt` στο δίσκο.
# %%



# %%
#  reduce the effect of outliers 
import numpy as np

df_train['WeeksToDeparture_cbrt'] = np.cbrt(df_train['WeeksToDeparture'])

df_train['Departure_cbrt'] = np.cbrt(df_train['Departure'])

df_train['Arrival_cbrt'] = np.cbrt(df_train['Arrival'])

#%%

from sklearn.ensemble import RandomForestRegressor 
import numpy as np

X_train = df_train
X_test = df_test
y_train = np.ravel(y_train)

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)

# In[8]:


from sklearn.linear_model import LogisticRegression
import numpy as np

X_train = df_train
X_test = df_test
y_train = np.ravel(y_train)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

X_train.shape

#print (metrics.accuracy_score(y_test, y_pred))
# In[9]:


import csv
with open('y_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Id', 'Label'])
    for i in range(y_pred.shape[0]):
        writer.writerow([i, y_pred[i]])


# In[10]:


from sklearn.metrics import f1_score
#y_test = np.loadtxt('test_labels.csv', delimiter=",", skiprows=1, usecols=[1])
f1_score(y_test, y_pred, average='micro')


# Υποβάλλουμε το αρχείο `y_pred.csv` στην πλατφόρμα και μας δίνει micro F1-score ίσο με 0.23.
# 
# Ένας εναλλακτικός τρόπος αναπαράστασης κατηγορικών μεταβλητών είναι το λεγόμενο one-hot encoding όπου υπάρχει διαθέσιμη μια μεταβλητή για κάθε πιθανή τιμή του χαρακτηριστικού και ανάλογα με την τρέχουσα τιμή του, μια από αυτές της μεταβλητές είναι 1, ενώ όλες οι άλλες παραμένουν 0. Για παράδειγμα, αν είχαμε κάποια μεταβλητή Weekday η οποία περιέγραφε τη μέρα που έγινε μια πτήση, θα είχαμε 7 μεταβλητές (π.χ. 1000000 για Monday, 0100000 για Tuesday κτλ.). Σημειώστε ότι με την one-hot encoding αναπαράσταση ο αριθμός των χαρακτηριστικών που προκύπτει είναι ίσος με τον αριθμό των διαφορετικών τιμών που παίρνει η μεταβλητή. Παρακάτω εφαρμόζουμε one-hot encoding στις μεταβλητές `Departure` και `Arrival`.

# In[11]:


from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False)
enc.fit(df_train)  
X_train = enc.transform(df_train)
X_test = enc.transform(df_test)
X_train.shape


# Βλέπουμε ότι ο αριθμός των στηλών αυξήθηκε από 2 σε 40. Αυτό συνέβη γιατί υπάρχουν 20 διαφορετικά αεροδρόμια και συνεπώς χρειαζόμαστε 20 μεταβλητές για να αναπαραστήσουμε κάθε μια από τις μεταβλητές `Departure` και `Arrival` χρησιμοποιώντας one-hot encoding.
# 
# Έπειτα, εκπαιδεύουμε ξανά έναν ταξινομητή logistic regression για να προβλέψουμε τις κατηγορίες αριθμού επιβατών των δεδομένων ελέγχου. Επιπλέον, αποθηκεύουμε τις προβλέψεις μας στο αρχείο `y_pred.csv` στο δίσκο.








# In[12]:


clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[13]:


with open('y_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Id', 'Label'])
    for i in range(y_pred.shape[0]):
        writer.writerow([i, y_pred[i]])


# In[14]:


f1_score(y_test, y_pred, average='micro')


# Υποβάλλουμε το αρχείο `y_pred.csv` στην πλατφόρμα και μας δίνει macro F1-score ίσο με 0.33. Στα πλαίσια της παρούσας εργασίας καλείστε να τροποποιήσετε τον παραπάνω κώδικα ώστε να προβλέψετε τις κατηγορίες αριθμού επιβατών των πτήσεων του test set. Μπορείτε να εφαρμόσετε κάποια μέθοδο επιλογής χαρακτηριστικών στα δεδομένα ώστε να κρατήσετε
# μόνο ένα υποσύνολο από τα χαρακτηριστικά. Μπορείτε επίσης να δημιουργήσετε νέα χαρακτηριστικά τα οποία
# ίσως βοηθήσουν στην κατηγοριοποίηση. Μπορείτε επιπλέον να πειραματιστείτε με κάποια μέθοδο μείωσης
# διάστασης και να διερευνήσετε αν η εφαρμογή της βελτιώνει το αποτέλεσμα της κατηγοριοποίησης. Επίσης,
# μπορείτε να χρησιμοποιήσετε θορυβώδη ή ανούσια χαρακτηριστικά για να παράγετε νέα χαρακτηριστικά που
# παρέχουν μεγαλύτερα ποσοστά πληροφορίας. Μπορείτε να χρησιμοποιήσετε διαφορετικούς ταξινομητές ή να συνδυάσετε τα αποτελέσματα περισσότερων από έναν ταξινομητές. 
# 
# ### Παράδοση Εργασίας
# 
# Η εργασία είναι είτε ατομική ή μπορεί να γίνει σε ομάδες το πολύ 3 ατόμων. Η τελική αξιολόγηση θα βασίζεται τόσο στο micro F1-score που θα επιτύχετε, όσο και στη συνολική προσέγγισή σας στο πρόβλημα. Στα πλαίσια της εργασίας, θα πρέπει να υποβληθούν τα εξής:
# <ul>
#     <li>Μια αναφορά 2 σελίδων, στην οποία θα πρέπει να περιγράψετε την προσέγγιση και τις μεθόδους που χρησιμοποιήσατε. Δεδομένου ότι πρόκειται για ένα πραγματικό πρόβλημα ταξινόμησης, μας ενδιαφέρει να γνωρίζουμε πώς αντιμετωπίσατε κάθε στάδιο του προβλήματος, π.χ. τι είδους αναπαράσταση δεδομένων χρησιμοποιήσατε, τι χαρακτηριστικά χρησιμοποιήσατε, εάν εφαρμόσατε τεχνικές μείωσης διάστασης, ποιούς αλγορίθμους ταξινόμησης δοκιμάσατε και γιατί, την απόδοση των μεθόδων σας (macro F1-score και χρόνο εκπαίδευσης), τυχόν προσεγγίσεις που τελικά δεν λειτούργησαν, αλλά
# είναι ενδιαφέρον να παρουσιάσετε, και γενικά, ότι νομίζετε ότι είναι ενδιαφέρον να αναφερθεί.</li>
#     <li>Ενα φάκελο με τον κώδικα της εφαρμογής σας.</li>
#     <li>Εναλλακτικά μπορείτε να συνδυάσετε τα δυο παραπάνω σε ένα αρχείο Ipython Notebook.</li>
#     <li>Δημιουργήστε ένα αρχείο .zip που περιέχει τον κώδικα και την αναφορά και υποβάλετέ τον στην πλατφόρμα e-class.</li>
#     <li>**Λήξη προθεσμίας**: 6 Ιανουαρίου 2019.</li>
# </ul>
