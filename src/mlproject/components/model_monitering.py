import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/ipatel/Documents/Amit/Python/Practice/p1/notbook/raw.csv')

# Display basic info and first few rows
df.info(), df.head()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Step 1: Create the Family column and drop SibSp & Parch
df["Family"] = df["SibSp"] + df["Parch"]
df.drop(columns=["SibSp", "Parch"], inplace=True)

# Step 2: Handle missing values
df.loc[:, "Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.loc[:, "Age"] = df["Age"].fillna(df["Age"].median())

# Step 3: Encode categorical variables
label_encoders = {}
for col in ["Sex", "Embarked"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for reference

# Step 4: Select features and target
X = df[["Pclass", "Sex", "Age", "Fare", "Embarked", "Family"]]
y = df["Survived"]

# Step 5: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Standardize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Check the processed data shape
X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Find the best model
best_model = max(results, key=results.get)
best_model, results

print("Best Model: ", best_model)
print("Accuracy: ", results[best_model])
