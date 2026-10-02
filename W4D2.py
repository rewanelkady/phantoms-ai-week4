import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv("cleaned_data.csv")

df_x = df[df['survived'] == 0]
df_y = df[df['survived'] == 1]

df_yd = df_y.sample(frac=0.1, random_state=42)

df_imbalanced = pd.concat([df_x, df_yd])

X = df_imbalanced[['pclass', 'age', 'sibsp', 'parch', 'fare']] 
y = df_imbalanced['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Normal Model Accuracy:", accuracy)
print("Normal Confusion Matrix:\n", cm)

smote = SMOTE()
Xtrain, ytrain = smote.fit_resample(X_train, y_train)

model_smote = LogisticRegression()
model_smote.fit(Xtrain, ytrain)
y_pred_smote = model_smote.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_smote)
cm = confusion_matrix(y_test, y_pred_smote)

print("SMOTE Model Accuracy:", accuracy)
print("SMOTE Confusion Matrix:\n", cm)