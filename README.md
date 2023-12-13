# Machine-Learning challenge


### Problem
Supervised learning.


### Project Summary
Use thesis data to predict the published years.

### Data
#### Data Description
| Column  | Description |
| ------------- | ------------- |
| ENTRYTYPE | inproceedings, proceedings, articles  |
| title  | the title of the thesis  |
| editor | editors  |
| year | published year  |
| publisher | publishers |
| author  | authors |
| abstract  | the summary of the thesis |
#### Data Files

The dataset folder consists of the following files:

* Train.json: Contains training data [65914 x 7] 
* Test.json: Contains test data [21972 x 6]
* predicted json: final output 


### Process
# Feature engineering
We started the challenge with exploratory data analysis (EDA) in which we discovered the data types and
the amount of missing values. Missing values in the author column were imputed with the corresponding
values of the editor column as we found out that in some cases the editors could be the authors. All other
missing values were imputed with empty text. We defined certain functions to preprocess and clean the
data. For the author column, the first and last name were connected, the list and punctuation were removed,
and finally the text was converted to lowercase. For the publisher, title, and abstract columns, they were
also all converted to lowercase. Additionally, from the title and abstract columns, the text between “[]” was
removed, English, French, and Chinese stop words were deleted, and the punctuation was removed. Lastly,
the editor column was removed because 97.76% of its values were missing. Multiple methods were applied
to transform the categorical/text to numerical features3. ENTRYTYPE and publisher were one-hot encoded
due to the lacking ordinal relationship between the categories. The title and abstract columns were
transformed using a HashingVectorizer6, the author column was transformed with a TfidfVectorizer7. We
used a for loop to check the different combinations of Hashing, Tfidf, and CountVecorizers1,2, and this
combination turned out best regarding the MAE. To ensure the model would capture relevant patterns and
relationships between the data, we applied lemmatization, stemming, LSA, and word2vec. However, this
created a lot of noise in the data and an increased MAE when running our final model, except for
lemmatization. Thus, we used lemmatization with the spaCy language model instead. After trial and error,
we tried different n_grams to see whether different combinations of words conveyed specific meanings and
n_features to control the dimensionality. N_grams of (1,3) and n_features of 2^23 were most successful in
terms of MAE.
# Learning algorithms
Ridge Regression (MAE 3.6), Lasso Regression (MAE 4.8), Random Forest Regressor (MAE 3.1), Decision
Tree Classifier (MAE 2.9) and Regressor (MAE 3.0), SGD Classifier (MAE 2.7), Gradient Boosting
Regression (MAE 3.6), and a Linear Support Vector Classifier (LSVC, MAE 2.4) were all used to fit the
models and to compare the MAEs. Even though the target variable is numeric, and the evaluation metric
MAE is more typical for a regression problem, we realized that the classifier models performed better in
terms of MAE compared to the regression models. Hence, we continued with the best-performing model,
the LSVC which has the advantage of being robust to the high-dimensional feature space that we created
with feature engineering.
# Hyperparameter tuning
The baseline MAE for our model was 2.416. We used GridSearch with cross-validation to tune the
hyperparameters of the LSVC on the training set. The scoring parameter was changed to
‘neg_mean_absolute_error’ to minimize the MAE. Different values for the range of the regularization
parameter were tried ([0.001, 0.01, 0.1, 1.0, 10.0, 20.0]), 1.0 turned out to be the best. After tuning, the
MAE remained the same (2.416).
# Performance of the model
Once we decided to use the LSVC, we improved the performance and featuring engineering systematically
after each submission. The first submission consisted of the imputation of missing values and text cleaning
by removing stop words, punctuation, text between brackets, and conversion to lowercase (MAE 2.8). In
the second submission, we added the lemmatization, n_grams, and connected the first and last name of
the authors (MAE 2.43). For the third submission, we also increased the capacity of n_features in the
HashingVectorizer (MAE 2.4067). The fourth submission changed the imputed missing values in the
publisher column with the mode instead of empty text. This was not successful and increased our MAE to
2.41. Thus, the best model was model 3 with an MAE 2.4067 and the final feature engineering as described
in the first part of this report (Feature engineering).

### My contribution
1. Processed the data
    - Created functions (connect_name, remove_list, remove punctuation, remove unnecessary words) and applied them to the features
    - Imputed the missing data of the author with editors and other features with an empty text.
    - Implemented text and categorical data encoding: Text: TfidfVectorizer with n_grams, HashingVectorizer with n_grams and n_features, Lemmatize the tokens; Categorical: OneHotEncoder
    - - Tried PCA and SelectFromModel to select the most important features
2. Experimented with different models (Regressor: XGB, Ridge, Lasso, SVM, Linear Regression, GradientBoostingRegressor, SGD; Classifier: SGD, Perceptron, PassiveAggressiveClassifier, LogisticRegression, RidgeClassifier, LinearSVC, MLPClassifier, RandomForestClassifier) and found the best model
3. Proposed two approaches for hypermeter-tuning and adopted one in the end: Three-way split and GridSearch with cross-validation; provided code with others to reuse it
4. Produced 4 submission files without any errors and optimized successfully for three times
      - First submission with MAE 2.80 (Found the best model LinearSVC)
      - Second submission with MAE 2.43 (Used lemmatize, ngram, and connected the first name with the last name in the author column for feature engineering)
      - Third submission with MAE 2.40 (Imputed the missing data of author with editor, increased the capacity of n_features in HashingVectorizer)
      - Fourth submission with MAE 2.41 (Impute null values in publish with mode -> it was imputed with an empty text at first)
5. Established and organized the final py file to submit
6. Provided extra comments and fine-tuned the report
7. Sought out solutions on the Sklearn website, Medium, and Stackoverflow for feature engineering and hyperparameter tuning in Introduction To Machine Learning with Python (OREILLY)

### Result

<img width="1440" alt="third prize" src="https://i.imgur.com/ZKxGM1u.png">

