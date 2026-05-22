import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree

df = pd.read_csv(
    r"C:\Users\sagay\OneDrive\Desktop\prodigy_ds_02\bank-additional-full.csv",
    sep=';'
)
print(df.head())

# =====================
# CHARTS (BEFORE ENCODING)
# =====================

plt.figure()
df['age'].hist(bins=20)
plt.title("Age Distribution")
plt.show()

plt.figure()
df['job'].value_counts().plot(kind='bar')
plt.title("Job Distribution")
plt.show()

plt.figure()
df['marital'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Marital Status")
plt.ylabel("")
plt.show()

plt.figure()
df['education'].value_counts().plot(kind='bar')
plt.title("Education")
plt.show()

plt.figure()
df['housing'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Housing Loan")
plt.ylabel("")
plt.show()

plt.figure()
df['loan'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Personal Loan")
plt.ylabel("")
plt.show()

plt.figure()
df['euribor3m'].hist(bins=30)
plt.title("Euribor Rate Distribution")
plt.show()

plt.figure()
df['y'].value_counts().plot(kind='bar')
plt.title("Subscription Result")
plt.show()

# =====================
# ENCODING (FOR MODEL ONLY)
# =====================

for col in df.columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop('y', axis=1).values
y = df['y'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =====================
# MODEL
# =====================

model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =====================
# TREE VISUALIZATION
# =====================

plt.figure(figsize=(20,10))

tree.plot_tree(
    model,
    filled=True
)

plt.show()