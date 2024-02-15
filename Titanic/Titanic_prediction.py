#import pandas
import pandas as pd

#import sklearn
import sklearn

#import numpy
import numpy as np

# Configure seed for reproducibility
np.random.seed(42)

# create a DataFrame with 20 lines
data = {
    'Survived': np.random.choice([0, 1], size=20),
    'Pclass': np.random.choice([1, 2, 3], size=20),
    'Sex': np.random.choice([0, 1], size=20),
    'Age': np.round(np.random.normal(loc=29, scale=10, size=20)),
    'SibSp': np.random.choice([0, 1, 2], size=20),
    'Parch': np.random.choice([0, 1, 2], size=20),
    'Fare': np.random.normal(loc=30, scale=15, size=20),
    'C': np.random.choice([False, True], size=20),
    'Q': np.random.choice([False, True], size=20),
    'S': np.random.choice([False, True], size=20),
}

final_base = pd.DataFrame(data)

#import load model
from joblib import load
#import the model
model = load('model/Titanic_model.joblib')

# predictions with new data
#X_= final_base.drop(['Survived', 'Unnamed: 0'],axis=1) Use this line if you want use other data
X= final_base.drop(['Survived'],axis=1)
y_= final_base['Survived']
pred_model = model.predict(X)
X['Predictions']  = pred_model

X.to_csv('data/processed/model_predictions.csv')

