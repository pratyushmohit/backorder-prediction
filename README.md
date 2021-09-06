# Predicting Material Backorders in Inventory Management
This is a case study on a research paper published at the IEEE conference, 2017. 
https://www.researchgate.net/publication/319553365_Predicting_Material_Backorders_in_Inventory_Management_using_Machine_Learningv

The goal is to minimize backorders by identifying the material at risk of backorder before the event occurs. This gives the business management a suitable time to react and make appropriate changes. The dataset for this problem has 2 classes, positive and negative classes. The positive class meaning the product went into backorder and the negative class indicating the opposite. This makes the problem a binary class classification. The data is highly imbalanced with a ratio of 1:148 for the positive and negative class respectively for the train set. Majority of the classes are negative i.e, most of the products did not go into backorder.

eda_and_feature_engineering.ipynb: This is file includes extensive exploratory data analysis on the train set. All my observations for each feature are documented in detail. I have also performed basic preprocessing and some feature engineering techniques on the train set to improve model results. The probability matrix files are a result of some feature transforms and the files are listed below.
    * deck_risk_probability_matrix.csv
    * oe_constraint_probability_matrix.csv
    * potential_issue_probability_matrix.csv
    * ppap_risk_probability_matrix.csv
    * rev_stop_probability_matrix.csv
    * stop_auto_buy_probability_matrix.csv 

eda_and_feature_engineering.pdf: This is a pdf version of the eda_and_feature_engineering.ipynb notebook.

model_building.ipynb: After preprocessing and feature engineering, comes model building. Here, I have tested different machine learning models like Logistic Regression, Decision Trees, Random Forests and more. All the results are documents and plotted wherever possible.
model_building.pdf: This is a pdf version of the model_building.ipynb notebook.

final_updated.ipynb: After finializing the model which gave the best results on the test set, I have built the entire pipeline beginning from taking a raw datapoint 'x' to predicting the class label. It also have a function which takes the entire dataset as input and gives the performance metrics of the best model as output.
final_updated.pdf: This is a pdf version of the final_updated.ipynb notebook.

streamlitapp: I have deployed the model on AWS using streamlit. All the required files are in this directory. To view the deployed model, please follow the link below. 
http://34.238.245.11:8501/
