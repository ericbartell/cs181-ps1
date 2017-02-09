
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# In[2]:

"""
Read in train and test as Pandas DataFrames
"""
trainFile = "train.csv"
testFile = "test.csv"
df_train = pd.read_csv(trainFile)
df_test = pd.read_csv(testFile)


# In[3]:

df_train['smiles'][0]
df_train['smiles'][3]


# In[4]:

df_test.head()


# In[5]:

#store gap values
Y_train = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)


# In[6]:

#break data into train and validation set (80%,20%)
percent_validation = .001
rand_mask = np.random.rand(len(df_train)) > percent_validation 
broken_train = df_train[rand_mask]
broken_validation = df_train[~rand_mask]
print(len(broken_train))
print(len(broken_validation))

#set last value of train, first of test
validation_idx = broken_train.shape[0]

#Reset Y_train value
Y_validation = Y_train[validation_idx:]
Y_train = Y_train[:validation_idx]


# In[7]:

#break data in a function (can iterate):
def make_validation_sets(train_data,num_groups,validation_set_size=1):
    if num_groups <= validation_set_size:
        print("Validation set size is too high for the number of groups")
        return None
    chunked_data = []
    rand_array = np.random.randint(0,num_groups,len(train_data))
    for i in range(num_groups):
        mask = rand_array == i
        chunked_data.append(train_data[mask])
    train_indexes = range(num_groups)
    validation_indexes = []
    train_val_pairs = []
    for i in range(validation_set_size):
        train_val_pairs.append(pd.concat(),pd.concat())
        
    


# In[8]:

#DataFrame with all train and test examples so we can more easily apply feature engineering on
#modified to use broken up data
df_all = pd.concat( (broken_train, broken_validation, df_test), axis=0) #(df_train, df_test), axis=0)
df_all.head()


# In[9]:

"""
Example Feature Engineering

this calculates the length of each smile string and adds a feature column with those lengths
Note: this is NOT a good feature and will result in a lower score!
"""
#smiles_len = np.vstack(df_all.smiles.astype(str).apply(lambda x: len(x)))
#df_all['smiles_len'] = pd.DataFrame(smiles_len)


# In[10]:

#Drop the 'smiles' column
df_all = df_all.drop(['smiles'], axis=1)
vals = df_all.values
#modified to use validation set as well
X_train = vals[:validation_idx]
X_validation = vals[validation_idx:test_idx]
X_test = vals[test_idx:]
print ("Train features:", X_train.shape)
print ("Train gap:", Y_train.shape)
print ("Test features:", X_test.shape)
print ("Placeholder, say something about validation set")



# In[13]:

LR = LinearRegression()
scores = cross_val_score(LR,X_train, Y_train,cv=5,scoring='neg_mean_squared_error')
print("Linear Regression:")
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
RF = RandomForestRegressor()
scores = cross_val_score(RF,X_train, Y_train,cv=5,scoring='neg_mean_squared_error')
print("\nRandom Forest:")
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[14]:

#finds root mean squared error given array of predicted and actual vals
def RMSE(predictions,actual):
    if len(predictions) != len(actual):
        print("Predictions must be same length as actual, and ORDERED")
        return None
    n = len(predictions)
    return np.sqrt( ((predictions-actual)**2).sum() /n)
    


# In[27]:


from matplotlib import pyplot as plt
def plot_accuracies(list_models,X_train,Y_train):
    fig,ax = plt.subplots()
    counter = -1
    colors = ['r','c','y','g','b','m','k']
    for i in list_models:
        counter += 1
        scores = cross_val_score(i,X_train, Y_train,cv=5,scoring='neg_mean_squared_error')
        ax.plot(counter,-scores.mean(),colors[counter%len(colors)]+'o',label=str(i).split("(")[0])

    ax.set_xlim(0-counter/5.0,counter+counter/5.0) 
    ax.legend(bbox_to_anchor=(1.6, 1.1))
    plt.title("Accuracies of cross validation")
    plt.xlabel("Order of attempting")
    plt.ylabel("RMSE")
    plt.show()

    
#expected input is a list of [MODEL, XtrainData, YtrainData]
def plot_accuracies_diff_data(list_models_and_X_and_Y_data):
    fig,ax = plt.subplots()
    counter = -1
    colors = ['r','c','y','g','b','m','k']
    score_list = []
    for i in list_models:
        counter += 1
        scores = cross_val_score(i[0],i[1], i[2],cv=5,scoring='neg_mean_squared_error')
        ax.plot(counter,-scores.mean(),colors[counter%len(colors)]+'o',label=str(i).split("(")[0])
    	score_list.append(-scores.mean())

    ax.set_xlim(0-counter/5.0,counter+counter/5.0) 
    ax.legend(bbox_to_anchor=(1.6, 1.1))
    plt.title("Accuracies of cross validation")
    plt.xlabel("Order of attempting")
    plt.ylabel("RMSE")
    plt.show()
    return score_list


# In[28]:

models = [LR,LR,LR,LR,RF,RF,RF,RF,RF,RF,RF,RF]
list_of_scores = plot_accuracies(models,X_train,Y_train)
bestModelIndex = list_of_scores.index(min(list_of_scores))
bestModelScore = list_of_scores[bestModelIndex]
print("BEST MODEL SCORE IS " + str(bestModelScore))
bestModel = models[bestModelIndex]
best_model_pred = bestModel.predict(X_test)

#note: number of cross valids seems to make a huge difference


# In[11]:

LR = LinearRegression()
LR.fit(X_train, Y_train)
LR_train = LR.predict(X_train)
#LR_valid = LR.predict(X_validation)
LR_pred = LR.predict(X_test)


# In[12]:

RF = RandomForestRegressor()
RF.fit(X_train, Y_train)
RF_train = RF.predict(X_train)
#RF_valid = RF.predict(X_validation)
RF_pred = RF.predict(X_test)


# In[17]:

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")


# In[18]:
print("WRITING TO FILE")

write_to_file("bestModelPredictions.csv", best_model_pred)
write_to_file("sample1_1.csv", LR_pred)
write_to_file("sample2_1.csv", RF_pred)

