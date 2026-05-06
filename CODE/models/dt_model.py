from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
from config import MODEL_DIR, OUTPUT_DIR

def train_dt_model(X, y):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train Decision Tree model
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Decision Tree Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Cross-validation score
    cv_scores = cross_val_score(dt, X, y, cv=5)
    print("CV Mean Accuracy:", cv_scores.mean())
    
    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "dt").mkdir(exist_ok=True)
    joblib.dump(dt, MODEL_DIR / "dt" / "decision_tree.pkl")
    print("DT model saved to MODEL/dt/decision_tree.pkl")
    
    return dt, accuracy