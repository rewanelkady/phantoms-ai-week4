from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("cleaned_data.csv")
print(df.head())

X = df[['pclass', 'age', 'sibsp', 'parch', 'fare']] 
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()

report = classification_report(y_test, y_pred)
print("Classification Report:\n")
print(report)


# FN

"""
annot=True ----> عشان تكتب الأرقام جوا المربعات.
fmt='d' -------> عشان تظهر كأرقام صحيحة (Integers).
cmap='Blues' --> بتدي تدرج ألوان جميل وخفيف (باللون الأزرق)."""