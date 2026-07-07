#Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#Load the Breast Cancer dataset
cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target

#Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

#Train the Random Forest classifier
rf_classifier.fit(x_train, y_train)

#Predict the labels for test set
y_pred = rf_classifier.predict(x_test)

#Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print ("Accuracy:", accuracy)

#Print classification report
print ("\nClassificaion Report:")
print (classification_report(y_test, y_pred))
