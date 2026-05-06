import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
from config import MODEL_DIR

def train_ann_model(X, y):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert target to categorical for neural network
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    # Create and train ANN model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_cat, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print("ANN Accuracy:", accuracy)
    
    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "ann").mkdir(exist_ok=True)
    model.save(MODEL_DIR / "ann" / "ann_model.keras")
    print("ANN model saved to MODEL/ann/ann_model.keras")
    
    return model, accuracy