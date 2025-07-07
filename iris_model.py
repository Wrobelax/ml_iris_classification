"""Main script with the iris model and the data"""
from pyexpat import features
# importing modules.
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# loading the data.
iris = load_iris()


# Converting the data to dataframe.
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
series = pd.Series(iris.target)


# basic data exploration.
# print(df.head())
# print(df.dtypes) # 6 columns. All in float/int.
# print(df.info) # 150 rows.
# print(df.describe(include = "all"))
# print(series.value_counts()) # 50 for each series.


# Replacing series numbers to true names and merging with dataframe.
series_named = series.replace({0 : "setosa", 1 : "versicolor", 2 : "virginica"})
df = pd.concat([df, series_named.rename("species")], axis = 1)
# print(series_named.value_counts())


# Basic visualization.
sns.pairplot(df, hue = "species")
# plt.savefig("../ml_iris_classification/data/petal_visualisation.png") # Saving the plot to file.



"""Preparing data for training and testing"""

# Dividing data into training and testing.
df_train = df.drop(columns = ["species"])
x_train, x_test, y_train, y_test = train_test_split(df_train, series_named, test_size = 0.2, random_state = 42)


# Data standardization.
scaler = StandardScaler()

scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Adding train and test data to dataframe.
x_train_scaled_df = pd.DataFrame(x_train_scaled, columns = df_train.columns)
x_test_scaled_df = pd.DataFrame(x_test_scaled, columns = df_train.columns)



"""Training classification model - Logistic regression"""

# Model initialization, training and evaluating accuracy.
log_reg = LogisticRegression(max_iter = 200, random_state = 42)
log_reg.fit(x_train_scaled_df, y_train)
accuracy = log_reg.score(x_test_scaled_df, y_test)
# print(f"Accuracy: {accuracy:.2f}") # 1.00


# Creating predictions, classification report and confusion matrix.
y_pred = log_reg.predict(x_test_scaled_df)
cm = confusion_matrix(y_test,y_pred)
class_report_lr = classification_report(y_test, y_pred)
# print(class_report_lr)
# print(cm)


# Creating cross-validation.
cv_score_lr= cross_val_score(log_reg, df_train, series_named, cv = 10)
# print(f"Cross-validation accuracy score: {cv_score_lr}")
# print(f"Mean accuracy: {cv_score_lr.mean()}") # Accuracy: 0.97


# Heatmap visualization of a confusion matrix.
plt.figure(figsize = (6,5))
sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues",
            xticklabels = log_reg.classes_,
            yticklabels = log_reg.classes_)
plt.title("Confusion matrix for logistic regression")
plt.xlabel("Predicted label")
plt.ylabel("True label")
# plt.savefig("../ml_iris_classification/data/conf_matrix_log_reg.png") # Saving the plot to file.



"""Training classification model - Random Forest"""

# Model initialization, training and predicting.
rf = RandomForestClassifier(random_state = 42)
rf.fit(x_train_scaled_df, y_train)

y_pred_rf = rf.predict(x_test_scaled_df)


# Checking accuracy of the model.
accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print(f"Accuracy: {accuracy_rf:.2f}") # 1.00


# Creating confusion matrix and classification report.
cm_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)
# print(class_report_rf)
# print(cm_rf)

# Creating cross-validation.
cv_score_rf = cross_val_score(rf, df_train, series_named, cv = 10)
# print(f"Cross-validation accuracy score: {cv_score_rf}")
# print(f"Mean accuracy: {cv_score_rf.mean()}") # Accuracy: 0.96


# Visualization of a confusion matrix.
plt.figure(figsize = (6,5))
sns.heatmap(cm_rf, annot = True, fmt = "d", cmap = "Blues",
            xticklabels = rf.classes_,
            yticklabels = rf.classes_)
plt.title("Confusion matrix for random forest")
plt.xlabel("Predicted label")
plt.ylabel("True label")
# plt.savefig("../ml_iris_classification/data/conf_matrix_rand_forest.png") # Saving the plot to file.



"""Visualization of a cross-validation for Logistic Regression and Random Forest"""

plt.figure(figsize = (8, 5))
x_pos = np.arange(1, 11)
plt.plot(x_pos, cv_score_lr, marker = "o", label = "Logistic Regression")
plt.plot(x_pos, cv_score_rf, marker = "s", label = "Random Forest")
plt.title("Cross-validation accuracy per fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.1)
plt.xticks(x_pos)
plt.grid(True)
plt.legend()
# plt.savefig("../ml_iris_classification/data/cross_val_lin.png") # Saving the plot to file.



"""Visualization of classification reports metrics for Logistic Regression and Random Forest"""

# Creating variable for unique class names.
class_names = ["setosa", "versicolor", "virginica"]


# Creating dictionary of a report.
report_lr = classification_report(y_test, y_pred, target_names = class_names, output_dict = True)
report_rf = classification_report(y_test, y_pred_rf, target_names = class_names, output_dict = True)

classes = class_names
x_position = np.arange(len(classes))
width = 0.12


# Creating function for extracting reports from reports.
def get_metrics(report):
    precision = [report[cls]["precision"] for cls in classes]
    recall = [report[cls]["recall"] for cls in classes]
    f1 = [report[cls]["f1-score"] for cls in classes]
    return precision, recall, f1

prec_lr, rec_lr, f1_lr = get_metrics(report_lr)
prec_rf, rec_rf, f1_rf = get_metrics(report_rf)


# Visualization.
fig, ax = plt.subplots(figsize = (14, 7))

offsets = {
    "prec_lr": -2 * width,
    "rec_lr": -1 * width,
    "f1_lr": 0,
    "prec_rf": width,
    "rec_rf": 2 * width,
    "f1_rf": 3 * width
}

ax.bar(x_position + offsets["prec_lr"], prec_lr, width, label = "LogReg Precision", color = "tab:blue", hatch = "//")
ax.bar(x_position + offsets["rec_lr"], rec_lr, width, label = "LogReg Recall", color = "tab:blue", hatch = "\\\\")
ax.bar(x_position + offsets["f1_lr"], f1_lr, width, label = "LogReg F1-Score", color = "tab:blue")

ax.bar(x_position + offsets["prec_rf"], prec_rf, width, label = "RandFor Precision", color = "tab:orange", hatch = "//")
ax.bar(x_position + offsets["rec_rf"], rec_rf, width, label = "RandFor Recall", color = "tab:orange", hatch = "\\\\")
ax.bar(x_position + offsets["f1_rf"], f1_rf, width, label = "RandFor F1-Score", color = "tab:orange")

ax.set_xticks(x_position)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.1)
ax.set_title("Comparison of classification metric for Logistic Regression and Random Forest")
ax.set_ylabel("Score")
ax.legend(loc = "upper left", bbox_to_anchor = (1,1))
plt.tight_layout()
# plt.savefig("../ml_iris_classification/data/class_report_visualization.png") # Saving the plot to file.



"""Visualization of a 3D plot for Logistic Regression and Random Forest."""

# Preparing data for visualization.
features_3d = ["sepal length (cm)", "petal length (cm)", "petal width (cm)"]
class_vals = {0 : "setosa", 1 : "versicolor", 2 : "virginica"}
x_vis_3d = x_test_scaled_df[features_3d].copy()


# Predicting the data and mapping names to classes.
y_pred_lr = log_reg.predict(x_test_scaled_df)
y_pred_rf = rf.predict(x_test_scaled_df)

x_vis_3d["LogReg"] = pd.Series(y_pred_lr)
x_vis_3d["RandFor"] = pd.Series(y_pred_rf)


# Colors assignment.
colors = {
    "setosa" : "red",
    "versicolor" : "green",
    "virginica" : "blue"
}


# Plotting the chart.
fig = plt.figure(figsize = (16,7))

# Logistic Regression.
ax1 = fig.add_subplot(121, projection = "3d")
for cls in class_vals.values():
    subset = x_vis_3d[x_vis_3d["LogReg"] == cls]
    ax1.scatter(
        subset[features_3d[0]],
        subset[features_3d[1]],
        subset[features_3d[2]],
        label = cls,
        c = colors[cls],
        s = 50,
        alpha = 0.7
    )

ax1.set_xlabel(features_3d[0])
ax1.set_ylabel(features_3d[1])
ax1.set_zlabel(features_3d[2])
ax1.set_title("Logistic Regression")


# Random Forest.
ax2 = fig.add_subplot(122, projection = "3d")
for cls in class_vals.values():
    subset = x_vis_3d[x_vis_3d["RandFor"] == cls]
    ax2.scatter(
        subset[features_3d[0]],
        subset[features_3d[1]],
        subset[features_3d[2]],
        label = cls,
        c = colors[cls],
        s = 50,
        alpha = 0.7
    )

ax2.set_xlabel(features_3d[0])
ax2.set_ylabel(features_3d[1])
ax2.set_zlabel(features_3d[2])
ax2.set_title("Random Forest")

ax2.legend(loc = "upper left", bbox_to_anchor = (1, 1))
plt.tight_layout()
# plt.savefig("../ml_iris_classification/data/3d_visualization.png") # Saving the plot to file.



"""Pipeline"""
#=========================
# This is an example of use of a pipeline in the whole process.
# It is not used in the project as ir would change the core of it.
# It is only to demonstrate good practices.
#==========================

# Logistic Regression pipeline.
log_reg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter = 200, random_state = 42))
])


# Random Forest pipeline.
rand_for_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("randfor", RandomForestClassifier(n_estimators = 100, random_state = 42))
])


# Example cross-validation.
logreg_scores = cross_val_score(log_reg_pipeline, df_train, series_named, cv = 10, scoring = "accuracy")
rand_for_scores = cross_val_score(rand_for_pipeline, df_train, series_named, cv = 10, scoring = "accuracy")


# Fitting and predicting data for Logistic Regression.
log_reg_pipeline.fit(x_train, y_train)
y_pred_lr_pipe = log_reg_pipeline.predict(x_test)


# Fitting and predicting data for Random Forest.
rand_for_pipeline.fit(x_train, y_train)
y_pred_rf_pipe = rand_for_pipeline.predict(x_test)