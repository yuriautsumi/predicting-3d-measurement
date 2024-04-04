# Questions

## 1. Briefly describe your methodology for analyzing the data.

First, I inspected missing data. There weren’t very many (less than 10 rows out of 5000+ observations), which led me to believe that a simple imputation (e.g. with median or KNN) should be sufficient. I dropped any rows where all input or all output features were missing. 

Then, I plotted the distribution of each variable in a histogram and a box plot. The histogram helped me see that the variables had skewed distributions. The box plot and summary statistics helped me see what kind of outliers exist and check that there are no strange data points that should be dropped. I repeated this procedure to compare distributions across gender, which showed the differences in distribution depending on whether the data comes from a male or female. 

I computed correlations within input variables, outcome variables, and across the two variables. Some input variables were correlated. Outcome variables were correlated, which made me think that regressors that exploit this correlation may perform better. It was a good sign to see that inputs were correlated with outcome. 

I then regressed each outcome on each input feature to confirm whether there exists a linear relationship. There was a strong linear relationship between each outcome and weight, but it was weaker for the other variables. I repeated this procedure to compare regressions for each gender to check for Simpson’s paradox (differences in trend on overall population v. Within subpopulations). 

Then, I repeated both procedures above on polynomial features. I decided to look into polynomial features because e.g. an increase in weight may have different effect on waist circumference, depending on whether height increased or decreased. I indeed observed linear relationship between order 2 features and outcomes. I observed Simpson’s paradox when regressing on these features. This made me think that fitting separate regressions for each gender may be useful. 

Although the data is not very high dimensional, I also visualized the data using PCA to see if there would be any obvious clusterings. After computing the first two principal components, I created a scatterplot of the points projected onto the PCs. The scatter points’ colors were uniquely determined by the outcomes so that similar outcomes have similar color. Results showed that PCA preserves the information from the gender (Male v. Female) feature, while possibly introducing useful features that allow e.g. the green group to be easily distinguishable from the black group. However, some groups (e.g. purple) were not as clearly separated, which made me think that linear projections of the data may not be sufficient to get good estimates.

I combined these insights to build a data transform pipeline. It uses KNN to impute missing features and transforms skewed distributions to a unit Gaussian. Depending on the setting, it also adds polynomial (order 3) and PCA features. 

Finally, I transformed the data with the pipeline and plotted the new distributions for each variable in a histogram and a box plot. I noticed that e.g. differences in male v. Female subpopulations and outliers were preserved, but distributions were no longer as skewed. 

## 2. Which ML algorithm did you pick for making predictions? Why? Which ML algorithm would be a poor choice for this dataset? Why?

I chose a Gradient Boosted Tree trained using Chained multi output with polynomial features on the entire population to make predictions. It had one of the highest mean R2 values on the validation set, and its 90% confidence intervals of R2 metric (on held out set) is comparable to the top performing algorithms. It had little to no significant relationship between residuals and the input features and output features, as well as the distribution of residuals between gender groups. The way they are trained decorrelates the variables and ensures that feature importances aggregated across the ensemble models captures feature importances from correlated features. That and its ability to smooth out inconsistencies as an ensemble model made this a good choice. 

MLP and Support Vector Machines also performed well, however, they had slightly more correlation between residuals and predicted values or input features. Otherwise, these would have been good choices due to MLP’s ability to create shared representations for the multiple outputs & model interactions and SVR’s maximal margin formulation and equivalence to an infinite order polynomial function that can efficiently capture interactions. 

For the experiments where I do not add any polynomial features, any linear model (Linear, Lasso, Ridge, ElasticNet, SVR (order 1 polynomial kernel)), would be a poor choice because they cannot model effects of interactions, e.g. an increase in weight may have different effect on waist circumference, depending on whether height increased or decreased. 

Since we are modeling multiple outcomes that are correlated, ML algorithms that do not take advantage of the correlation may also be a poor choice. In Scikit-learn, Support Vector Regression and Gradient Boosted Trees do not support multiple outcome regression. Building a separate regression for each outcome may be less preferred over e.g. chaining models so that \hat{y}^{d-1} is used to predict \hat{y}^{d}.  

## 3. Which evaluation metric did you pick? Why? What evaluation metric would you use if this was a classification problem to predict whether a person was male or female? How would your answer change if the classes were very imbalanced? 

I chose the R-squared evaluation metric (a measure of proportion of variation explained by the model) because it is less sensitive to outliers or few very bad predictions than MSE or MAE.

If this were a classification problem to predict gender, I would choose AUROC, which would tell me the probability that the classifier correctly ranks a randomly sampled negative point and a randomly sampled positive point. The classification threshold can be adjusted to balance/trade off false positive, false negative, etc. 

If the classes were very imbalanced, I would choose AUPRC since it is capable of detecting whether or not the classifier performs well on the minority positive class by focusing on precision (PPV) and recall (TPR) metrics. However, it is important to note that AUPRC is interpreted relative to baseline (the proportion of positive events), which differs across datasets, so it cannot be used to compare results across different datasets. 

## 4. The body measurement prediction problem is a regression problem. In a classification problem which would worry you more: false positives or false negatives? Why?

Assuming that the positive outcome encodes the outcome we are interested in surfacing, e.g. +1 = presence of disease, then we would worry about false negatives more. False negatives are true positive samples that are predicted negative. High false negative metric means that we are unable to surface that many positive (e.g. disease) samples. 

However, if we have 2 classes that can be encoded arbitrarily, e.g. +1 = male, -1 = female, then both false negatives and false positives matter equally for male (+1) and female (-1) groups, respectively. 

## 5. How would you deal with missing data for numerical and categorical variables?

If we do not want to impute missing values using e.g. sample statistics, nearby points, we can replace them with a constant value such as 0 or -100. 

Otherwise, we can impute and add an additional ‘missingness indicator’ column (0 if missing, 1 if present in original data): 
A simple approach would be to impute missing numerical variable by the median (more robust to outliers than mean) and the missing categorical variable by the mode. 
K-Nearest Neighbors is a non parametric model that imputes missing values from the nearest neighbors (e.g. average of their values) in the training set.
For time series data, we might fill forward missing values by the last non missing value. 


## 6. Explain regularization to a layperson. What is it and why would you want to use it?

Regularization controls how complex our model is, so that the model does not overfit to the data. 

In technical terms, regularizers add bias to reduce variance of the estimator so that we can decrease generalization error (on the held out test set). 

## 7. What’s the difference between L1 and L2 regularization methods?

L1 regularization adds a constraint that the L1 norm of the weight vector must be within some threshold, while L2 regularization adds a constraint that the L2 norm of the weight vector must be within some threshold. The problem turns into a constrained optimization to minimize the objective function subject to these regularization constraints. 

The L1 regularization constraint can set weights to 0, enforcing sparse weight vectors to be learned, while L2 regularization spreads weight across many variables. L1 regularization is often used when working with high dimensional feature sets with collinearity, since the sparse weights implicitly perform feature selection. L2 regularizer helps address issues that arise in Ordinary Least Squares due to linearly dependent features. The ridge (L2) estimator alleviates this issue by adding a constant along the diagonal (ridge) of X^{T}X matrix, making it non singular.

When L1 and L2 regularization are both used, this results in the “elastic net” regularizer. This regularizer assigns similar weights to highly correlated features, thus “grouping” correlated features together to have the same weight. This may be used when L1 regularizer cannot be used because the solution is non-unique or if we would like to surface all associated variables instead of 1 variable in each correlated group. 

## 8. What is better: 50 small decision trees or one large decision tree? Why?

50 small decision trees is better. Larger trees (more complex models) tend to overfit to the training data and generalize less to the held out dataset. Using 1 tree leads to high variance in outcomes. In addition, there are theoretical arguments that show that aggregating outcomes across multiple trees trained on bootstrap samples reduce estimation error. 

However, aggregating results across multiple trees loses any simple form of interpretability. 

