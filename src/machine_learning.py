import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class MachineLearning:

    def __init__(self):
        self.df = pd.read_csv("../data/dataset_test.csv")
        self.X = self.df.drop(columns=["Re-Entry Achieved?"])
        self.y = self.df["Re-Entry Achieved?"]
        self.y = self.y.map({"No": 0, "Yes": 1})

        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
            self.X, self.y, 
            test_size=0.2,
            random_state=0)
        
   
    def decision_tree_classifier(self):
        orbit_model = DecisionTreeClassifier(random_state=1)
        orbit_model.fit(self.train_X, self.train_y)

        predicted_decay_success = orbit_model.predict(self.val_X)
        accuracy = accuracy_score(self.val_y, predicted_decay_success)
        matrix = confusion_matrix(self.val_y, predicted_decay_success)
        importance = orbit_model.feature_importances_
        # for feature, score in zip(self.X.columns, importance):
            # print(f"{feature}: {score:.3f}")

        return accuracy
    
    def random_forest_classifier(self):
        orbit_model = RandomForestClassifier(random_state=1)
        orbit_model.fit(self.train_X, self.train_y)

        predicted_decay_success = orbit_model.predict(self.val_X)
        accuracy = accuracy_score(self.val_y, predicted_decay_success)
        matrix = confusion_matrix(self.val_y, predicted_decay_success)
        importance = orbit_model.feature_importances_
        # for feature, score in zip(self.X.columns, importance):
            # print(f"{feature}: {score:.3f}")

        return accuracy

    def logistic_regression(self):
        orbit_model = LogisticRegression(max_iter=5000)
        orbit_model.fit(self.train_X, self.train_y)

        predicted_decay_success = orbit_model.predict(self.val_X)
        accuracy = accuracy_score(self.val_y, predicted_decay_success)
        matrix = confusion_matrix(self.val_y, predicted_decay_success)

        return accuracy
    
    def xgboost(self):
        orbit_model = XGBClassifier(n_estimators=400, learning_rate=0.05)
        orbit_model.fit(self.train_X, self.train_y)

        predicted_decay_success = orbit_model.predict(self.val_X)
        accuracy = accuracy_score(self.val_y, predicted_decay_success)

        return accuracy

    def results(self):
        results = {"Decision Tree Classifier": self.decision_tree_classifier(),
                   "Random Forest Classifier": self.random_forest_classifier(),
                   "Logistic Regression": self.logistic_regression(),
                   "XGBoost": self.xgboost()}
        
        print(results)
    
    def prediction(self, )

ml = MachineLearning()
ml.results()