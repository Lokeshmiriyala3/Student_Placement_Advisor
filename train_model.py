import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as s
import warnings as w
w.filterwarnings("ignore")
pd.set_option('display.max_rows', None)

# Load the Excel file into a DataFrame

data = pd.read_csv('Students_Dataset.csv')

# Display the first few rows of the DataFrame
print(data.head())
data_copy=data.copy()

# Display the first few rows of the DataFrame
print(data.head())
data_copy.drop(columns=['Full Name', 'Mobile Number','Department','Status','What package were you offered during your placement, either on-campus or off-campus?','Timestamp'], inplace=True)

col=[]
for i in data_copy.columns:
  if data[i].dtypes=='object':
    col.append(i)
from sklearn.preprocessing import LabelEncoder
obj = LabelEncoder()
for i in col:
    data_copy[i] = data_copy[i].astype(str)  # Convert to strings
    data_copy[i] = obj.fit_transform(data_copy[i])
target_var='Have you been placed in any company?'
independ_var=[]
for i in data_copy.columns:
  if i!='Have you been placed in any company?':
    independ_var.append(i)
x=data_copy[independ_var]
y=data_copy[target_var]
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
joblib.dump(scal_obj, 'scaler.pkl')
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
)

# Create objects for classification algorithms
ModelLR = LogisticRegression()
ModelDC = DecisionTreeClassifier()
ModelRF = RandomForestClassifier()
ModelET = ExtraTreesClassifier()
ModelKNN = KNeighborsClassifier(n_neighbors=5)
ModelSVM1 = SVC(kernel="rbf", random_state=42, class_weight="balanced", probability=True)
ModelSVM2 = SVC(kernel="linear", random_state=0)
ModelSVM3 = SVC(kernel="poly", random_state=0)
ModelGNB = GaussianNB()

# List of models
MM = [ModelLR, ModelDC, ModelRF, ModelET, ModelKNN, ModelSVM1, ModelSVM2, ModelSVM3, ModelGNB]

# Create an empty DataFrame to store results
EMResults = pd.DataFrame(
    columns=[
        "Model Name",
        "True_Positive",
        "False_Negative",
        "False_Positive",
        "True_Negative",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Specificity",
        "MCC",
        "ROC_AUC_Score",
        "Balanced Accuracy",
    ]
)

# Loop through models and evaluate
for models in MM:
    print(f"\n{'-' * 60}\nEvaluating Model: {models.__class__.__name__}\n{'-' * 60}")

    # Fit the model
    models.fit(x_train, y_train)

    # Predictions
    y_pred = models.predict(x_test)
    y_pred_prob = models.predict_proba(x_test)[:, 1] if hasattr(models, "predict_proba") else None

    # Confusion Matrix
    matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
    tp, fn, fp, tn = matrix.ravel()

    # Classification Report
    C_Report = classification_report(y_test, y_pred, labels=[1, 0])
    print("Confusion Matrix:\n", matrix)
    print("Classification Report:\n", C_Report)

    # Metrics Calculation
    sensitivity = round(tp / (tp + fn), 3) if (tp + fn) != 0 else 0
    specificity = round(tn / (tn + fp), 3) if (tn + fp) != 0 else 0
    accuracy = round((tp + tn) / (tp + fp + tn + fn), 3)
    balanced_accuracy = round((sensitivity + specificity) / 2, 3)
    precision = round(tp / (tp + fp), 3) if (tp + fp) != 0 else 0
    f1Score = round((2 * tp) / (2 * tp + fp + fn), 3) if (2 * tp + fp + fn) != 0 else 0
    mcc_denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    MCC = round(((tp * tn) - (fp * fn)) / sqrt(mcc_denom), 3) if mcc_denom != 0 else 0
    roc_auc = round(roc_auc_score(y_test, y_pred), 3) if hasattr(models, "predict_proba") else "N/A"

    # Print Metrics
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall (Sensitivity): {sensitivity * 100:.2f}%")
    print(f"F1 Score: {f1Score}")
    print(f"Specificity (TNR): {specificity * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
    print(f"MCC: {MCC}")
    print(f"ROC AUC Score: {roc_auc}")

    # ROC Curve
    if y_pred_prob is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{models.__class__.__name__} (AUC = {roc_auc})")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {models.__class__.__name__}")
        plt.legend(loc="lower right")
        plt.show()

    # Add metrics to results DataFrame
    new_row = pd.DataFrame([{
        "Model Name": models.__class__.__name__,
        "True_Positive": tp,
        "False_Negative": fn,
        "False_Positive": fp,
        "True_Negative": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": sensitivity,
        "F1 Score": f1Score,
        "Specificity": specificity,
        "MCC": MCC,
        "ROC_AUC_Score": roc_auc,
        "Balanced Accuracy": balanced_accuracy,
    }])
    EMResults = pd.concat([EMResults, new_row], ignore_index=True)



# Display consolidated results
print("\nConsolidated Results:\n", EMResults)
joblib.dump(ModelRF, 'ModelRF.pkl')

# Predict and merge with actual results
y_pred = ModelRF.predict(x_test)
result = pd.DataFrame({"Placed_actual": y_test, "Placed_pred": y_pred})
data = data.merge(result, right_index=True, left_index=True)

