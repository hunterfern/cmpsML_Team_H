from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
from config import MODEL_DIR

def train_svm_model(X, y):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train SVM model
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("SVM Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    
    # Cross-validation score
    cv_scores = cross_val_score(svm, X, y, cv=5)
    print("CV Mean Accuracy:", cv_scores.mean())
    
    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "svm").mkdir(exist_ok=True)
    joblib.dump(svm, MODEL_DIR / "svm" / "svm_model.pkl")
    print("SVM model saved to MODEL/svm/svm_model.pkl")
    
    return svm, accuracy