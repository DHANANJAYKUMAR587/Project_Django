import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your dataset
data = pd.read_csv(r"C:\Users\Ethnotech\Desktop\project_py\Crop_recommendation.csv")

# Split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model to project_py folder
pickle.dump(model, open(r"C:\Users\Ethnotech\Desktop\project_py\model.pkl", "wb"))

print("✅ Model trained and saved as model.pkl")
