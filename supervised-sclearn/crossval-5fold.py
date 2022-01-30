# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
 
# Create a linear regression object: reg
reg = LinearRegression()
 
# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)
 
# Print the 5-fold cross-validation scores
print(cv_scores)
# [ 0.81720569  0.82917058  0.90214134  0.80633989  0.94495637]
 
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
# Average 5-Fold CV Score: 0.8599627722793232