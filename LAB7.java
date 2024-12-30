// Code 1
import pandas as pd
import numpy as np
data = {
'Student': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
'Math': [95, np.nan, 85, 78, 88],
'Science': [90, 75, np.nan, 80, 95],
'English': [85, 90, 88, np. nan, 92]
}
df = pd.DataFrame(data)
df_filled = df.fillna(method='ffill')
print("Original DataFrame:")
print(df)
print("\nDataFrame after Forward Fill:")
print(df_filled)


// Code 2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    'Applicant_ID': [1, 2, 3, 4, 5],
    'Credit_Score': [700, 650, 600, 720, 680],
    "Annual_Income": [60000, 50000, 45000, 70000, 55000],
    "Employment_Status": ['Employed', 'Employed', 'Unemployed', 'Employed', 'Unemployed'],
    'Debt_Income_Ratio': [0.3, 0.4, 0.5, 0.2, 0.45],
    'Previous_Default': ['No', 'No', 'Yes', 'No', 'Yes'],
    "Loan_Approved": ['Yes', 'No', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

label_encoder = LabelEncoder()
df['Employment_Status'] = label_encoder.fit_transform(df['Employment_Status'])
df['Previous_Default'] = label_encoder.fit_transform(df['Previous_Default'])
df["Loan_Approved"] = label_encoder.fit_transform(df["Loan_Approved"])

X = df.drop(['Applicant_ID', 'Loan_Approved'], axis=1)  # Features
y = df['Loan_Approved']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



// Code 3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

data = {
    'Loan_Amount': [20000, 15000, 25000, 10000, 30000, 22000, 18000, 27000, 23000, 15000],
    'Annual_Income': [70000, 60000, 80000, 50000, 40000, 85000, 75000, 65000, 70000, 50000],
    'Credit_Score': [700, 600, 800, 650, 550, 720, 710, 730, 680, 590],
    'Employment_Status': ['Employed', 'Employed', 'Self-Employed', 'Employed', 'Unemployed', 'Employed', 'Self-Employed', 'Employed', 'Employed', 'Unemployed'],
    'Debt_to_Income_Ratio': [0.3, 0.5, 0.2, 0.4, 0.7, 0.3, 0.25, 0.2, 0.35, 0.6],
    'Previous_Default': ['No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes'],
    'Risk_Class': ['Low Risk', 'High Risk', 'Low Risk', 'High Risk', 'High Risk', 'Low Risk', 'Low Risk', 'Low Risk', 'High Risk', 'High Risk']
}

df = pd.DataFrame(data)
label_encoder = LabelEncoder()

df['Employment_Status'] = label_encoder.fit_transform(df['Employment_Status'])
df['Previous_Default'] = label_encoder.fit_transform(df['Previous_Default'])
df['Risk_Class'] = label_encoder.fit_transform(df['Risk_Class'])

X = df.drop('Risk_Class', axis=1)  # Features
y = df['Risk_Class']  # Target (Risk classification)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'loan_risk_classifier_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

def plot_feature_importance(model, features):
    feature_importance = model.feature_importances_
    sorted_idx = feature_importance.argsort()
    plt.figure(figsize=(10, 6))
    plt.barh(features[sorted_idx], feature_importance[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.show()

plot_feature_importance(model, X.columns)

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

labels = label_encoder.classes_
plot_confusion_matrix(y_test, y_pred, labels)



// Code 4


# ... (previous code for model training, evaluation, saving, etc.)

def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC Curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

plot_roc_curve(model, X_test, y_test)

new_applicant = pd.DataFrame({
    'Credit_Score': [710],
    'Annual_Income': [75000],
    'Loan_Amount': [22000],
    'Employment_Status': ['Employed'],  # New Label
    'Debt_to_Income_Ratio': [0.3],
    'Previous_Default': ['No']
})

def safe_transform(label_encoder, column_data, column_name):
    """
    Transform the column using the label encoder while handling unseen labels.
    If an unseen label is encountered, return a default or known value.
    """

    if column_name in ('Employment_Status', 'Previous_Default'):
        existing_labels = label_encoder.classes_
        transformed_labels = []

        for label in column_data:
            if label in existing_labels:
                transformed_labels.append(label_encoder.transform([label])[0])
            else:
                # Handle unseen label by assigning it a default category
                transformed_labels.append(label_encoder.transform([existing_labels[0]])[0])  # Default to the first class (could be 'Employed')

        return transformed_labels
    else:
        return label_encoder.transform(column_data)

# Apply the safe transform to the new applicant data
new_applicant['Employment_Status'] = safe_transform(label_encoder, new_applicant['Employment_Status'], 'Employment_Status')
new_applicant['Previous_Default'] = safe_transform(label_encoder, new_applicant['Previous_Default'], 'Previous_Default')

# Predict the risk class for the new applicant
predicted_risk = model.predict(new_applicant)

# Get the predicted risk class label
predicted_risk_label = label_encoder.inverse_transform(predicted_risk)

print(f"The predicted risk class for the new loan applicant is: {predicted_risk_label[0]}")





// Code 5

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk

# Download NLTK resources (if needed)
nltk.download("stopwords")

# Step 1: Load the data (replace with your dataset path)
# Here, we're assuming the dataset is a CSV with 'email' and 'Label' columns
data = pd.read_csv("spam_dataset.csv")

# Display the first few rows of the dataset
print(data.head())

# Step 2: Preprocess the text data
# The text data is in the 'email' column, and Labels (spam/not spam) are in the 'Label' column

# Step 3: Text preprocessing (Tokenization and stopwords removal) using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')

# Convert the emails to feature vectors using TF-IDF
X = vectorizer.fit_transform(data['email'])

# The target Labels (spam or not spam) are in the 'Label' column
y = data['Label']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Naive Bayes model (MultinomialNB)
model = MultinomialNB()

# Train the model on the training data
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Optionally: Save the model and vectorizer for future use
import joblib
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')



//code6

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

data = {
    'Symptom 1': ['Fever', 'Headache', 'Fever', 'Nausea', 'Headache', 'Cough', 'Fever', 'Nausea', 'Fatigue', 'Cough'],
    'Symptom 2': ['Cough', 'Dizziness', 'Cough', 'Fatigue', 'Dizziness', 'Fatigue', 'Cough', 'Fatigue', 'Headache', 'Dizziness'],
    'Symptom 3': ['Headache', 'Fatigue', 'Vomiting', 'Fever', 'Headache', 'Nausea', 'Headache', '', '', ''],
    'Symptom 4': ['Fatigue', '', '', '', 'Fatigue', '', '', '', '', ''],
    'Diagnosis': ['Disease A', 'Disease B', 'Disease A', 'Disease B', 'Disease B', 'Disease A', 'Disease A', 'Disease B', 'Disease A', 'Disease B']
}

df = pd.DataFrame(data)

label_encoder = LabelEncoder()

all_symptoms = pd.concat([df['Symptom 1'], df['Symptom 2'], df['Symptom 3'].fillna(''), df['Symptom 4'].fillna('')])
label_encoder.fit(all_symptoms)

df['Symptom 1'] = label_encoder.transform(df['Symptom 1'])
df['Symptom 2'] = label_encoder.transform(df['Symptom 2'])
df['Symptom 3'] = label_encoder.transform(df['Symptom 3'].fillna(''))
df['Symptom 4'] = label_encoder.transform(df['Symptom 4'].fillna(''))
df['Diagnosis'] = label_encoder.fit_transform(df['Diagnosis'])

X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ... (rest of the code for saving the model, handling new patient data, etc.)





//code 7




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

data = {
    'Size (sq ft)': [1500, 1800, 2400, 3000, 3500, 1200, 2000, 2500, 4000, 1800],
    'Bedrooms': [3, 4, 3, 5, 4, 2, 3, 4, 5, 3],
    'Age of House (years)': [10, 15, 20, 5, 12, 25, 10, 7, 3, 18],
    'Location': ['Downtown', 'Suburb', 'Suburb', 'Downtown', 'Suburb', 'Suburb', 'Downtown', 'Downtown', 'Suburb', 'Downtown'],
    'Price': [400000, 500000, 650000, 700000, 800000, 300000, 550000, 600000, 850000, 490000]
}

df = pd.DataFrame(data)

label_encoder = LabelEncoder()
df['Location'] = label_encoder.fit_transform(df['Location'])

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'R-squared: {r2_score(y_test, y_pred)}')

joblib.dump(model, 'house_price_predictor_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

new_house = pd.DataFrame({
    'Size (sq ft)': [2200],
    'Bedrooms': [4],
    'Age of House (years)': [10],
    'Location': ['Suburb']
})

new_house['Location'] = label_encoder.transform(new_house['Location'])

predicted_price = model.predict(new_house)

print(f'The predicted price of the new house is: ${predicted_price[0]:,.2f}')