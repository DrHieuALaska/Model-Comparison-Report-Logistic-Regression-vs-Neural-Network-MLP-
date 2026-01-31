from sklearn.linear_model import LogisticRegression
from Data.Scaling_Split_dataset import X_train_scaled, y_train, X_val_scaled, y_val

log_reg = LogisticRegression(
    penalty="l2", # Regularization L2
    C=0.1,        # Regularization strength {0,1 is the best out of [0.01,0.1,1,10,100]}
    solver="lbfgs", # default solver (optimizer)
    max_iter=1000,
    random_state=42
)

log_reg.fit(X_train_scaled, y_train)

y_val_pred = log_reg.predict(X_val_scaled)
y_val_prob = log_reg.predict_proba(X_val_scaled)[:, 1] # Probability for positive class (Malignant)

from Model.Evaluation import evaluation
evaluation(y_val, y_val_pred)

import joblib
# Save the logistic regression model object to a file
joblib.dump(log_reg, 'Model/logistic_model.pkl')
print("Logistic Regression model saved!")