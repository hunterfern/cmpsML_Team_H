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
    history = model.fit(X_train, y_train_cat, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    
    # Save epoch-error curve
    import matplotlib.pyplot as plt
    from config import FIGURE_DIR
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('ANN Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy curve
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('ANN Accuracy Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "ann_epoch_curves.png")
    plt.close(fig)
    print("ANN epoch curves saved to DOC/figures/ann_epoch_curves.png")

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print("ANN Accuracy:", accuracy)
    
    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "ann").mkdir(exist_ok=True)
    model.save(MODEL_DIR / "ann" / "ann_model.keras")
    print("ANN model saved to MODEL/ann/ann_model.keras")
    
    return model, accuracy