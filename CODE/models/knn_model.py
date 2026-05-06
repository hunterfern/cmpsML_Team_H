from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
from config import MODEL_DIR

def train_knn_model(X, y):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train K-NN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("KNN Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    
    # Cross-validation score
    cv_scores = cross_val_score(knn, X, y, cv=5)
    print("CV Mean Accuracy:", cv_scores.mean())
    
    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "knn").mkdir(exist_ok=True)
    joblib.dump(knn, MODEL_DIR / "knn" / "knn_model.pkl")
    print("KNN model saved to MODEL/knn/knn_model.pkl")
    
    return knn, accuracy