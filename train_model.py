import pandas as pd #Used to load datasets
from sklearn.model_selection import train_test_split, GridSearchCV #Used to split datasets into training & testing sets and perform hyperparameter tuning
from sklearn.preprocessing import StandardScaler #Used to normalize features
from sklearn.metrics import accuracy_score, classification_report #Used to evaluate model performance
from sklearn.ensemble import RandomForestClassifier #Used to implement Random Forest Algorithm
from imblearn.over_sampling import SMOTE #Used to balance the dataset
import joblib #Used to save the trained models to the disk

data = pd.read_csv([INSERT LOCATION OF THE CSV FILE], header=None) #Used to load the CSV file
print("Original dataset shape:", data.shape) #Prints the rows and columns in the CSV file
data = data[(data[0] > 0.1) & (data[0] < 0.6)] #Filters out extremely small and large values of EAR
data = data[(data[1] > 0.1) & (data[1] < 2.5)] #Filters out extremely small and large values of MAR
print("Filtered dataset shape:", data.shape) #Prints out the remaining rows and columns in the CSV file after filtering
data["eye_mouth_ratio"] = data[0] / (data[1] + 1e-6) #Creates a new feature by calculating the ratio of EAR by MAR
data["jaw_face_ratio"] = data[9] / (data[10] + 1e-6) #Creates a new feature by calculating the ratio of the jaw's opening by the size of the face
data["brow_ratio"] = data[6] / (data[7] + 1e-6) #Creates a new feature by calculating the ratio of the distance between the eyebrows by the distance between the eyebrow and the eye
data["eye_diff"] = abs(data[2] - data[3]) #Creates a new feature by calculating the difference between the size of both the eyes
data["mouth_jaw_ratio"] = data[4] / (data[8] + 1e-6) #Creates a new feature by calculating the ratio of the width of the mouth by the jaw's opening
y = data[12].astype(int) #Extracts the columns and ensures that they are labelled
x = data.drop(columns=[12]) #Extracts the rows
x.columns = x.columns.astype(str) #Converts the column names to strings
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y) #Splits the dataset into taining and testing sets
scaler = StandardScaler() #Used to create a normalization object
x_train = scaler.fit_transform(x_train) #Used to compute the mean and standard deviation
x_test = scaler.transform(x_test) #Uses the same parameters on the test data
smote = SMOTE(random_state=42) #Used to create a SMOTE object
x_train, y_train = smote.fit_resample(x_train, y_train) #Used to generate synthetic samples for minority classes
print("Training samples after SMOTE:", x_train.shape) #Prints the number of training samples available after balancing
model = RandomForestClassifier(class_weight="balanced", random_state=42) #Used to create a Random Forest Classifier
param_grid = {"n_estimators": [300, 500, 800], "max_depth": [10, 16, 22], "min_samples_split": [2, 4, 6]} #Defines the number of trees, their maximum depth and their minimum samples required to split to test for
grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1) #Used to search for the best hyperparameters with the Random Forest parameter grid and splitting the dataset into 5 parts to train the model 5 times by utilising all the CPU cores
print("Training model with grid search...") 
grid.fit(x_train, y_train) #Used to train the model for every parameter combination
best_model = grid.best_estimator_ #Used to store the best performing model
print("Best parameters:", grid.best_params_) #Prints the best parameters
pred = best_model.predict(x_test) #Used to predict the labels for the test data
print("\nFinal Accuracy:", accuracy_score(y_test, pred)) #Used to compute the accuracy of the model
print("\nClassification Report:\n")
print(classification_report(y_test, pred)) #Prints the detailed metrics for the accuracy
joblib.dump(best_model, "stress_model.pkl") #Used to save the trained model
joblib.dump(scaler, "feature_scaler.pkl") #Used to save the feature normalization parameters
print("\nModel saved.")
