import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==================================================
# LOAD DATASET
# ==================================================
df = pd.read_csv(
    "ai_student_impact_dataset (1).csv"
)

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)

# ==================================================
# INPUT AND OUTPUT
# ==================================================
target = "Burnout_Risk_Level"

X = df.drop(
    columns=[
        "Student_ID",
        target
    ]
)

y = df[target]

print("\nTarget Classes:")
print(y.unique())

# ==================================================
# TRAIN TEST SPLIT
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:")
print(X_train.shape)
print(y_train.shape)

print("\nTest Shape:")
print(X_test.shape)
print(y_test.shape)

# ==================================================
# NUMERICAL COLUMNS
# ==================================================
num_cols = X_train.select_dtypes(
    include=['int64', 'float64']
).columns

print("\nNumerical Columns:")
print(list(num_cols))

# ==================================================
# CATEGORICAL COLUMNS
# ==================================================
cat_cols = X_train.select_dtypes(
    include=['object', 'bool']
).columns

print("\nCategorical Columns:")
print(list(cat_cols))

# ==================================================
# NUMERICAL PIPELINE
# ==================================================
num_transformer = Pipeline([
    (
        'imputer',
        SimpleImputer(strategy='mean')
    ),
    (
        'scaler',
        StandardScaler()
    )
])

# ==================================================
# CATEGORICAL PIPELINE
# ==================================================
cat_transformer = Pipeline([
    (
        'imputer',
        SimpleImputer(strategy='most_frequent')
    ),
    (
        'onehot',
        OneHotEncoder(handle_unknown='ignore')
    )
])

# ==================================================
# COLUMN TRANSFORMER
# ==================================================
preprocessor = ColumnTransformer([
    (
        'num',
        num_transformer,
        num_cols
    ),
    (
        'cat',
        cat_transformer,
        cat_cols
    )
])

# ==================================================
# COMPLETE MODEL PIPELINE
# ==================================================
model = Pipeline([
    (
        'preprocessor',
        preprocessor
    ),
    (
        'classifier',
        LogisticRegression(
            max_iter=1000
        )
    )
])

# ==================================================
# TRAIN MODEL
# ==================================================
print("\nTraining Model...")

model.fit(
    X_train,
    y_train
)

print("Training Completed!")

# ==================================================
# PREDICTIONS
# ==================================================
y_pred = model.predict(
    X_test
)

# ==================================================
# ACCURACY
# ==================================================
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========================")
print("MODEL EVALUATION")
print("========================")

print(
    "\nAccuracy:",
    round(accuracy, 4)
)

# ==================================================
# CLASSIFICATION REPORT
# ==================================================
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ==================================================
# CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# ==================================================
# CONFUSION MATRIX VISUALIZATION
# ==================================================
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Student Burnout Prediction"
)

plt.show()

# ==================================================
# SAVE MODEL
# ==================================================
joblib.dump(
    model,
    "burnout_prediction_model.pkl"
)

print(
    "\nModel Saved Successfully!"
)

# ==================================================
# EXPECTED INPUT COLUMNS
# ==================================================
print(
    "\nExpected Input Columns:"
)

print(
    model.named_steps[
        "preprocessor"
    ].feature_names_in_
)