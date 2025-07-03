```Project status: In progress - initialized.```


## **Project description**
This project utilises iris data from sklearn module, manipulates it, trains ml model and uses for prediction models visualization.
The data used in the project is pretty simple and covers only 150 cases (rows). Visualization of predictions and trainig/testing data shows close to perfection fit which corresponds with simplicity of the data but also confirm that chosen models properly correspond with the data.


### Features


### Tech Stack
- Pandas
- Matplotlib
- Seaborn
- Sklearn


### Project structure


### Data description
* "petal_visualisation.png": Basic visualization of the iris data before any train or test.
* "conf_matrix_log_reg.png": Visualization of a confusion matrix for logistic regression prediction model. It shows where the model is and isn't accurate.
  * **Conclusion:** 'versicolor' class is confused with 'virginica' sometimes.
* "conf_matrix_rand_forest.png": Visualization of a confusion matrix for random forest prediction model. It shows where the model is and isn't accurate.
  * **Conclusion:** Logistic regression and random forest confusion matrix are the same. It shows that both work fine on the dataset.
* "cross_val_lin.png": Linear model for logistic regression and random forest cross-validation 10-fold results.
  * **Conclusion:** Both models return similar results with small differences. It may be worth to use simpler model which is logistic regression.
* "class_report_visualization.png": Classification report visualization for logistic regression and random forest.
  * **Conclusion:** All values are on 1 which means that both models perfectly fit in the data and prediction.