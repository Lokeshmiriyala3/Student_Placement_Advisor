import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as s
import tkinter
import warnings as w
w.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
# Load the Excel file into a DataFrame

data = pd.read_csv('Students_Dataset.csv')

# Display the first few rows of the DataFrame
print(data.head())
data_copy=data.copy()
data_copy.columns
data_copy.isnull().sum()
data_copy.head()
data_copy.drop(columns=['Full Name', 'Mobile Number','Department','Status','Have you been placed in any company?','Timestamp'], inplace=True)
for  i in data.columns:
  print(data[i].unique())
col=[]
for i in data_copy.columns:
  if data[i].dtypes=='object':
    col.append(i)
from sklearn.preprocessing import LabelEncoder
obj = LabelEncoder()
for i in col:
    data_copy[i] = data_copy[i].astype(str)  # Convert to strings
    data_copy[i] = obj.fit_transform(data_copy[i])
# Define the target variable
target_var = "What package were you offered during your placement, either on-campus or off-campus?"

# Separate features (X) and target variable (y)
x = data_copy.drop(columns=[target_var])  # Drop the target variable to get all other features
y = data_copy[target_var]  # Target variable

# Display the selected features and target for verification
print("Independent Variables (Features):", x.columns.tolist())
print("Target Variable:", target_var)

from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=60)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
from sklearn.preprocessing import MinMaxScaler
import joblib

# Fit the scaler on the training data
scal_obj = MinMaxScaler(feature_range=(0, 1))
x_train = scal_obj.fit_transform(x_train)
x_train=pd.DataFrame(x_train)# Scaling training data
x_test = scal_obj.transform(x_test)  # Scaling test data (use transform, not fit_transform)
x_test=pd.DataFrame(x_test)
# Save the fitted scaler
joblib.dump(scal_obj, 'scaler1.pkl')
EMResults1 = pd.DataFrame(columns=[
    'Model Name', 
    'True_Positive', 
    'False_Negative', 
    'False_Positive', 
    'True_Negative', 
    'Accuracy', 
    'Precision', 
    'Recall', 
    'F1 Score', 
    'Specificity', 
    'MCC', 
    'ROC_AUC_Score', 
    'Balanced Accuracy'
])
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

# Create objects for classification algorithms
ModelLR = LogisticRegression(multi_class='ovr', solver='liblinear')
ModelDC = DecisionTreeClassifier()
ModelRF = RandomForestClassifier()
ModelET = ExtraTreesClassifier()
ModelKNN = KNeighborsClassifier(n_neighbors=5)
ModelSVM1 = SVC(kernel="rbf", random_state=42, class_weight="balanced", probability=True)
ModelSVM2 = SVC(kernel="linear", random_state=0, probability=True)
ModelSVM3 = SVC(kernel="poly", random_state=0, probability=True)
ModelGNB = GaussianNB()

# List of models
models_list = [ModelLR, ModelDC, ModelRF, ModelET, ModelKNN, ModelSVM1, ModelSVM2, ModelSVM3, ModelGNB]

# Create an empty DataFrame to store results
EMResults = pd.DataFrame(columns=[
    "Model Name", "Accuracy", "Precision", "Recall", "F1 Score", "Specificity",
    "Balanced Accuracy", "MCC", "ROC AUC Score"
])

# Loop through models and evaluate
for model in models_list:
    print(f"\n{'-' * 60}\nEvaluating Model: {model.__class__.__name__}\n{'-' * 60}")
    
    # Fit the model
    model.fit(x_train, y_train)
    
    # Predictions
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
    
    # Confusion Matrix
    matrix = confusion_matrix(y_test, y_pred, labels=list(set(y_test)))
    print("Confusion Matrix:\n", matrix)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    specificity = sum(matrix[i, i] / matrix[:, i].sum() for i in range(matrix.shape[0]) if matrix[:, i].sum() != 0) / len(set(y_test))
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Adjust ROC AUC Score for multi-class classification
    if y_pred_prob is not None and len(set(y_test)) > 2:
        roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    elif y_pred_prob is not None:
        roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
    else:
        roc_auc = None
    
    # Print Metrics
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1}")
    print(f"Specificity: {specificity * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
    print(f"MCC: {mcc}")
    print(f"ROC AUC Score: {roc_auc}")
    
    # ROC Curve
    if y_pred_prob is not None:
        for i in range(y_pred_prob.shape[1]):
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, i], pos_label=i)
            plt.figure()
            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], "r--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {model.__class__.__name__}")
            plt.legend(loc="lower right")
            plt.show()
    
    # Add metrics to results DataFrame
    new_row = pd.DataFrame([{
        "Model Name": model.__class__.__name__,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity,
        "Balanced Accuracy": balanced_accuracy,
        "MCC": mcc,
        "ROC AUC Score": roc_auc,
    }])
    EMResults = pd.concat([EMResults, new_row], ignore_index=True)

# Display consolidated results
print("\nConsolidated Results:\n", EMResults)


joblib.dump(ModelRF, 'ModelRF1.pkl')

# Predict and merge with actual results
y_pred = ModelRF.predict(x_test)
result = pd.DataFrame({"Placed_actual": y_test, "Placed_pred": y_pred})
data = data.merge(result, right_index=True, left_index=True)


