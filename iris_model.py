"""Main script with the iris model and the data"""

# importing modules.
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
log_reg = LogisticRegression(random_state = 42)
log_reg.fit(x_train_scaled_df, y_train)
accuracy = log_reg.score(x_test_scaled_df, y_test)
# print(f"Accuracy: {accuracy:.2f}") # 1.00


# Creating predictions and confusion matrix.
y_pred = log_reg.predict(x_test_scaled_df)
cm = confusion_matrix(y_test,y_pred)
# print(classification_report(y_test, y_pred))
# print(cm)


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
# print(classification_report(y_test, y_pred_rf))


# Creating confusion matrix.
cm_rf = confusion_matrix(y_test, y_pred_rf)


# Visualization of a confusion matrix.
plt.figure(figsize = (6,5))
sns.heatmap(cm_rf, annot = True, fmt = "d", cmap = "Blues",
            xticklabels = rf.classes_,
            yticklabels = rf.classes_)
plt.title("Confusion matrix for random forest")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.savefig("../ml_iris_classification/data/conf_matrix_rand_forest.png") # Saving the plot to file.
plt.show()