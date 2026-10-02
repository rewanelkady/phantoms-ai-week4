import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier         
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np


df = pd.read_csv("cleaned_data.csv")
print(df.head())

X = df[['pclass', 'age', 'sibsp', 'parch', 'fare']] 
y = df['survived']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
model_lr = LogisticRegression()
scores_lr = cross_val_score(model_lr, X, y, cv=skf, scoring='accuracy')

model_dt = DecisionTreeClassifier(random_state=42)
scores_dt = cross_val_score(model_dt, X, y, cv=skf, scoring='accuracy')

mean_lr = np.mean(scores_lr)
std_lr = np.std(scores_lr)

mean_dt = np.mean(scores_dt)
std_dt = np.std(scores_dt)

print(f"Mean1 = {mean_lr:.4f}")
print(f"Standard Deviation1 = {std_lr:.4f}")
print(f"Mean2 = {mean_dt:.4f}")
print(f"Standard Deviation2 = {std_dt:.4f}")
