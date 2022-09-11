# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model


![nn_model](https://user-images.githubusercontent.com/77089743/189541792-560402aa-7860-4eb5-8fea-58b6923cff34.png)



## DESIGN STEPS

### STEP 1:
      We start by reading the dataset using pandas.

### STEP 2:
      The dataset is then preprocessed, i.e, we remove the features that don't contribute towards the result.

### STEP 3:
      The null values are removed aswell.
      
### STEP 4:
      The resulting data values are then encoded. We, ensure that all the features are of the type int, or float, 
      for the model to better process the dataset.

### STEP 5:
      Once the preprocessing is done, we split the available data into Training and Validation datasets.
      
### STEP 6:
      The Sequential model is then build using 1 input layer, 3 dense layers(hidden) and, output layer.
<br>
<br>
<br>
<br>

### STEP 7:
      The model is then complied and trained with the data. A call back method is also implemented to prevent the model from overfitting.
      
### STEP 8:
      Once the model is done training, we validate and use the model to predict values.

## PROGRAM

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt

### Load the dataset
customer_df = pd.read_csv('customers.csv')
customer_df.columns
customer_df.dtypes
customer_df.shape
customer_df.isnull().sum()
customer_df_cleaned = customer_df.dropna(axis=0)
customer_df_cleaned.isnull().sum()
customer_df_cleaned.shape
customer_df_cleaned.dtypes
customer_df_cleaned['Gender'].unique()
customer_df_cleaned['Ever_Married'].unique()
customer_df_cleaned['Graduated'].unique()
customer_df_cleaned['Profession'].unique()
customer_df_cleaned['Spending_Score'].unique()
customer_df_cleaned['Var_1'].unique()
customer_df_cleaned['Segmentation'].unique()
categories_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
enc = OrdinalEncoder(categories=categories_list)
customers_1 = customer_df_cleaned.copy()
customers_1[['Gender',
             'Ever_Married',
              'Graduated','Profession',
              'Spending_Score']] = enc.fit_transform(customers_1[['Gender',
                                                                 'Ever_Married',
                                                                 'Graduated','Profession',
                                                                 'Spending_Score']])
customers_1.dtypes
le = LabelEncoder()
customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])
customers_1.dtypes
customers_1 = customers_1.drop('ID',axis=1)
customers_1 = customers_1.drop('Var_1',axis=1)
customers_1.dtypes

### Calculate the correlation matrix
corr = customers_1.corr()

### Plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)
sns.pairplot(customers_1)
sns.distplot(customers_1['Age'])
plt.figure(figsize=(10,6))
sns.countplot(customers_1['Family_Size'])
plt.figure(figsize=(10,6))
sns.boxplot(x='Family_Size',y='Age',data=customers_1)
plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Spending_Score',data=customers_1)
plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Age',data=customers_1)
customers_1.describe()
customers_1['Segmentation'].unique()
X=customers_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values
y1 = customers_1[['Segmentation']].values
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y1.shape
y = one_hot_enc.transform(y1).toarray()
y.shape
y1[0]
y[0]
X.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)
X_train[0]
X_train.shape
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

### To scale the Age column
X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)

### Creating the model
ai_brain = Sequential([
    Dense(8,input_shape = (8,)),
    Dense(16,activation='relu'),
    Dense(32,activation='relu'),
    Dense(32,activation='relu'),
    Dense(4,activation='softmax'),
])
ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2)
ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=2000,batch_size=256,
             validation_data=(X_test_scaled,y_test),
             )
metrics = pd.DataFrame(ai_brain.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()

### Sequential predict_classes function is deprecated

### predictions = ai_brain.predict_classes(X_test)
x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))

### Saving the Model
ai_brain.save('customer_classification_model.h5')

### Saving the data
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le], fh)

### Loading the Model
ai_brain = load_model('customer_classification_model.h5')

### Loading the data
with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)
Prediction for a single input
x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))
```

## Dataset Information

![Screenshot 2022-09-11 233148](https://user-images.githubusercontent.com/77089743/189542413-ee139a31-495f-455d-beff-e37c2edc7019.png)


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2022-09-11 233406](https://user-images.githubusercontent.com/77089743/189542438-6a986f09-e6e5-4d56-a824-01c16d6b8781.png)

### Classification Report

![Screenshot 2022-09-11 233544](https://user-images.githubusercontent.com/77089743/189542453-24a0923a-5d4d-48c9-9833-e922295b0742.png)


### Confusion Matrix

![Screenshot 2022-09-11 233629](https://user-images.githubusercontent.com/77089743/189542480-e781076d-10d3-43d6-99b1-4a49155b6ab4.png)


### New Sample Data Prediction

![Screenshot 2022-09-11 233719](https://user-images.githubusercontent.com/77089743/189542512-710a1c48-379b-4f06-81d2-f3809490ee91.png)


## RESULT

      Thus, we have developed a neural network classification model for the given dataset.
