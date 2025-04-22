## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
NAME: DEEPIKA R
REG NO.: 212223230038
```
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/2491ed72-b002-4a5a-8747-25acc2b29a5d)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/eb5c5259-eb47-4014-880f-c913fd8a1983)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/10051c5f-c62c-45c4-b7ee-7a5cd9b98400)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/dc7585e6-727a-46d2-86f7-592844384763)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/0352a3bd-797a-4cb4-b6f4-5a599d02a83b)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/5a64570d-d2dc-4105-8d2f-7f9d71aa6ad3)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/50465de7-f699-468f-96bc-56e5b976bca7)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/8f4cda92-f29f-4ee8-b0e9-aad4217fa0ea)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/73265cc9-99b0-4f96-8a24-4d2a8f4696d3)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/6fa4d1ce-1816-4562-9ae3-3bb4dd4c91e0)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/1d8abbb9-a92a-4299-b3bb-b641b4eba64f)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/70131a1b-a74d-41e1-868a-4d123fa9ea27)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/03ea6262-ccf9-449d-a2e3-1106638f355e)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/c430e42c-c34c-4be1-b07e-97812ef53965)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/ead7669d-093e-45d9-8d4a-4dab496fc622)
```
from scipy import stats

df["Highly Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Highly Negative Skew"])

print(df.skew())
```
![image](https://github.com/user-attachments/assets/12cb45a9-a928-4b75-a4ff-40c07dba476b)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/29ed7382-acf9-4712-b6f7-5292a512378d)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/3b9d4e0e-4451-41c7-ae40-88ab2f77a322)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/644b5a21-5b40-471b-b57c-fc8628f35046)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/d36aee92-61c2-431e-afe9-1adad883f307)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')

plt.show()
```
![image](https://github.com/user-attachments/assets/2124ea17-3507-49e6-8f9e-ed8e02c5e317)
```
dt = pd.read_csv("titanic_dataset.csv")
dt = dt.dropna(subset=["Age"])
qt = QuantileTransformer(output_distribution='normal', n_quantiles=dt.shape[0])
dt["Age_1"] = qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'], line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/6af8b4e8-10dd-4f4c-aac7-67055eed4741)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/b3008e20-ed95-430c-af73-32e6b7ffb917)


# RESULT:
Feature Encoding, Transformation process and saving the data to a file is done successfully
       
