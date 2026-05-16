import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class MachineLearning:

    def __init__(self):
        self.df = pd.read_csv("../data/dataset_test.csv")
        self.X = self.df.drop(columns=["Re-Entry Achieved?"])
        self.y = self.df["Re-Entry Achieved?"]
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
        for feature, score in zip(self.X.columns, importance):
            print(f"{feature}: {score:.3f}")

        return accuracy, matrix
    
    def random_forest_classifier(self):
        orbit_model = RandomForestClassifier(random_state=1)
        orbit_model.fit(self.train_X, self.train_y)

        predicted_decay_success = orbit_model.predict(self.val_X)
        accuracy = accuracy_score(self.val_y, predicted_decay_success)
        matrix = confusion_matrix(self.val_y, predicted_decay_success)
        importance = orbit_model.feature_importances_
        for feature, score in zip(self.X.columns, importance):
            print(f"{feature}: {score:.3f}")

        return accuracy, matrix

    def logistic_regression(self):
        orbit_model = LogisticRegression(max_iter=5000)
        orbit_model.fit(self.train_X, self.train_y)

        predicted_decay_success = orbit_model.predict(self.val_X)
        accuracy = accuracy_score(self.val_y, predicted_decay_success)
        matrix = confusion_matrix(self.val_y, predicted_decay_success)

        return accuracy, matrix
    
    
    
    def test(self):
        # print(self.decision_tree_classifier())
        # print(self.random_forest_classifier())
        # print(self.logistic_regression())
        print()

ml = MachineLearning()
ml.test()