# Mushroom Toxicity Classification - Team H

A machine learning project for classifying mushrooms as poisonous or edible using multiple ML algorithms.

**Team Members:** Hunter Fernandez, Hunter Merritt

---

## 📋 Project Overview

### Problem Statement
Many mushrooms can be either not poisonous, mildly poisonous, or lethally poisonous. When attempting to cook or consume wild mushrooms, it is critically important to know their toxicity level. While manual attribute analysis hasn't revealed obvious correlations, machine learning models can help discover hidden patterns in mushroom characteristics that predict toxicity.

### Goals
- Identify objects' class labels (edible vs. poisonous) using ML algorithms
- Explore feature distributions and correlations
- Compare model performance across multiple algorithms: **ANN, SVM, Decision Tree, and K-NN**
- Create a robust, well-documented codebase for team collaboration

### Key Tasks
- ✓ Visualize data
- ✓ Preprocess data
- ✓ Design representation
- ✓ Extract features
- ✓ Explore feature distributions
- ✓ Design and create models
- ✓ Conduct model assessment

---

## 📊 Dataset Information

**Dataset Name:** Mushroom Classification Dataset (UCI Machine Learning Repository)

**Citation:** Mushroom [Dataset]. (1981). UCI Machine Learning Repository. https://doi.org/10.24432/C5959T

**Direct Link:** https://archive.ics.uci.edu/dataset/73/mushroom

**File Location:** `INPUT/TRAIN/mushroom.csv`

### Features (23 attributes)
| Feature | Type | Encoding |
|---------|------|----------|
| **poisonous** (Target) | Categorical | e=edible, p=poisonous |
| cap-shape | Categorical | b=bell, c=conical, x=convex, f=flat, k=knobbed, s=sunken |
| cap-surface | Categorical | f=fibrous, g=grooves, y=scaly, s=smooth |
| cap-color | Categorical | n=brown, b=buff, c=cinnamon, g=gray, r=green, p=pink, u=purple, e=red, w=white, y=yellow |
| bruises | Categorical | t=bruises, f=no bruises |
| odor | Categorical | a=almond, l=anise, c=creosote, y=fishy, f=foul, m=musty, n=none, p=pungent, s=spicy |
| gill-attachment | Categorical | a=attached, d=descending, f=free, n=notched |
| gill-spacing | Categorical | c=close, w=crowded, d=distant |
| gill-size | Categorical | b=broad, n=narrow |
| gill-color | Categorical | k=black, n=brown, b=buff, h=chocolate, g=gray, r=green, o=orange, p=pink, u=purple, e=red, w=white, y=yellow |
| stalk-shape | Categorical | e=enlarging, t=tapering |
| stalk-root | Categorical | b=bulbous, c=club, u=cup, e=equal, z=rhizomorphs, r=rooted, ?=missing |
| stalk-surface-above-ring | Categorical | f=fibrous, y=scaly, k=silky, s=smooth |
| stalk-surface-below-ring | Categorical | f=fibrous, y=scaly, k=silky, s=smooth |
| stalk-color-above-ring | Categorical | n=brown, b=buff, c=cinnamon, g=gray, o=orange, p=pink, e=red, w=white, y=yellow |
| stalk-color-below-ring | Categorical | n=brown, b=buff, c=cinnamon, g=gray, o=orange, p=pink, e=red, w=white, y=yellow |
| veil-type | Binary | p=partial, u=universal |
| veil-color | Categorical | n=brown, o=orange, w=white, y=yellow |
| ring-number | Categorical | n=none, o=one, t=two |
| ring-type | Categorical | c=cobwebby, e=evanescent, f=flaring, l=large, n=none, p=pendant, s=sheathing, z=zone |
| spore-print-color | Categorical | k=black, n=brown, b=buff, h=chocolate, r=green, o=orange, u=purple, w=white, y=yellow |
| population | Categorical | a=abundant, c=clustered, n=numerous, s=scattered, v=several, y=solitary |
| habitat | Categorical | g=grasses, l=leaves, m=meadows, p=paths, u=urban, w=waste, d=woods |

**Data Characteristics:**
- Total instances: 8,124 mushroom observations
- All features are categorical (no missing values except stalk-root)
- Binary classification problem (edible/poisonous)
- Well-balanced dataset

---

## 🏗️ Project Structure

```
cmpsML_Team_H/
├── CODE/                          # Python source code
│   ├── main.py                   # Main entry point
│   ├── config.py                 # Configuration and paths
│   ├── io_utils.py               # Data loading utilities
│   ├── preprocess.py             # Data preprocessing
│   ├── feature_exploration.py    # Feature analysis & visualization
│   ├── models/
│   │   ├── ann_model.py         # Artificial Neural Network
│   │   ├── svm_model.py         # Support Vector Machine
│   │   ├── dt_model.py          # Decision Tree
│   │   └── knn_model.py         # K-Nearest Neighbors
│   └── evaluation.py             # Model evaluation & metrics
├── INPUT/                         # Input data directory (READ ONLY in git)
│   ├── TRAIN/
│   │   └── mushroom.csv         # Training dataset
│   └── TEST/
│       └── mushroom_test.csv    # Test dataset (optional)
├── OUTPUT/                        # Generated outputs
│   ├── charts/                   # Visualization plots
│   ├── metrics/                  # Performance metrics & results
│   ├── predictions/              # Model predictions on test data
│   └── samples/                  # Sample outputs
├── MODEL/                         # Trained model parameters
│   ├── ann/                      # ANN model files
│   ├── svm/                      # SVM model files
│   ├── dt/                       # Decision Tree model files
│   └── knn/                      # K-NN model files
├── DOC/                           # Documentation
│   ├── figures/                  # Documentation figures
│   ├── ppt/                      # Presentation files
│   └── report/                   # Final project report
├── OTHER/                         # Miscellaneous files
│   └── scratch/                  # Experimental/throwaway code
├── .venv/                         # Python virtual environment
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── CONTRIBUTING.md              # Collaboration guidelines
├── PROGRESS.md                  # Team progress tracking
└── .gitignore                   # Git ignore rules

```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for version control)

### Initial Setup (Run Once)

**1. Clone the repository (if starting fresh):**
```bash
git clone <repository-url>
cd cmpsML_Team_H
```

**2. Create and activate virtual environment:**
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows - PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Windows - Command Prompt)
.venv\Scripts\activate.bat

# Activate (macOS/Linux)
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Verify setup:**
```bash
python CODE/main.py
```

---

## 📦 Dependencies

Core libraries:
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms (DT, KNN, SVM)
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualization
- **tensorflow/keras** - Artificial Neural Networks
- **joblib** - Model serialization

See `requirements.txt` for exact versions.

---

## 📅 Project Timeline & Assignment

| Week | Task | Primary Owner | Status |
|------|------|---------------|--------|
| 1 | **Visualize data** - EDA, distributions, correlations | | ⏳ Not Started |
| 2 | **Preprocess data** - Handle missing values, encoding | | ⏳ Not Started |
| 3 | **Design representation** - Feature scaling, normalization | | ⏳ Not Started |
| 4 | **Extract features** - Feature selection, dimensionality reduction | | ⏳ Not Started |
| 5 | **Explore feature distributions** - Statistical analysis | | ⏳ Not Started |
| 6 | **Design & create models** - Implement ANN, SVM, DT, KNN | | ⏳ Not Started |
| 7 | **Model assessment** - Cross-validation, performance metrics | | ⏳ Not Started |

**Update this table in [PROGRESS.md](PROGRESS.md) as work progresses.**

---

## 💻 Running the Project

### Execute Main Pipeline
```bash
python CODE/main.py
```

### Train Individual Models
```bash
# Decision Tree
python CODE/models/dt_model.py

# K-Nearest Neighbors
python CODE/models/knn_model.py

# Support Vector Machine
python CODE/models/svm_model.py

# Artificial Neural Network
python CODE/models/ann_model.py
```

### Expected Outputs
- **MODEL/** - Trained model parameters saved as `.pkl` or `.h5` files
- **OUTPUT/metrics/** - Accuracy, precision, recall, F1-score, confusion matrices
- **OUTPUT/charts/** - Visualizations (ROC curves, feature importance, etc.)
- **OUTPUT/predictions/** - Test predictions with confidence scores

---

## 📐 Coding Standards & Requirements

**You MUST follow these standards or lose points:**

### 1. **Code Organization**
- Divide code into procedural sections: Pre-processing → Model → Assessment
- Use separate modules for different functionality (`preprocess.py`, `models/`, `evaluation.py`)
- Import statements at top of file

### 2. **Naming Conventions**
- Use self-explanatory variable names: `mushroom_features` not `x`
- Use snake_case for functions/variables: `calculate_accuracy()`
- Use PascalCase for classes: `DecisionTreeModel`
- Avoid single-letter variables except for loop indices (`i`, `j`)

### 3. **Comments & Documentation**
- **Each code block** (function, loop, conditional) must have a descriptive comment
- Add module docstrings explaining purpose
- Example:
```python
def calculate_model_accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        float: Accuracy score (0-1)
    """
    # Compare predictions to actual labels and compute percentage
    correct = sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    return accuracy  #
```

### 4. **Code Efficiency**
- **Do NOT** calculate constant values inside loops
  ```python
  # ❌ BAD - calculates every iteration
  for sample in dataset:
      multiplier = 2 * 3
      result = sample * multiplier
  
  # ✅ GOOD - calculate once
  multiplier = 2 * 3
  for sample in dataset:
      result = sample * multiplier
  ```

- Initialize lists with `None` if size is known, not `append()`:
  ```python
  # ❌ BAD - inefficient
  predictions = []
  for sample in dataset:
      predictions.append(model.predict(sample))
  
  # ✅ GOOD - pre-allocate
  predictions = [None] * len(dataset)
  for i, sample in enumerate(dataset):
      predictions[i] = model.predict(sample)
  ```

- Use appropriate data structures (dict for lookups, list for ordered data)

### 5. **Data Types & Security**
- Use immutable types where possible (tuples over lists for constants)
- Validate input data before processing
- Handle missing values explicitly

### 6. **Testing**
- Test with artificial data where you know the expected output
- Example: Create a small test dataset and verify preprocessing steps

### 7. **Comments on Every Block**
- **Add `#` at the end of each code block** (as per standard #8)
  ```python
  # Load and preprocess data
  df = pd.read_csv('data.csv')
  X = df.drop('target', axis=1)
  y = df['target']
  #

  # Encode categorical features
  encoder = OneHotEncoder()
  X_encoded = encoder.fit_transform(X)
  #
  ```

### 8. **Paths**
- Use **relative paths** from `config.py`
- Do NOT hardcode absolute paths like `C:/Users/Hunter/...`
- Example: Use `from config import MODEL_DIR` then `MODEL_DIR / "model.pkl"`

### 9. **Output Organization**
- Always save to categorized folders in `OUTPUT/`
- Charts → `OUTPUT/charts/chart_name.png`
- Metrics → `OUTPUT/metrics/model_metrics.csv`
- Predictions → `OUTPUT/predictions/predictions.csv`

### 10. **Visualization & Aesthetics**
- Use distinct colors, line styles, and markers for multi-series plots
- Include legends with clear labels
- Add titles, x-axis labels, y-axis labels
- Save at high DPI for quality: `plt.savefig(..., dpi=300)`

---

## 👥 Team Collaboration Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed collaboration workflows.

### Quick Version:
1. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Work on your assigned task** (see PROGRESS.md)

3. **Commit frequently** with clear messages:
   ```bash
   git commit -m "Add feature_exploration.py for EDA"
   ```

4. **Push and create a Pull Request** when done:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Review & merge** after checking for conflicts

---

## 🤖 Models to Implement

All four models should:
- Train on preprocessed training data
- Make predictions on test data
- Save model parameters to `MODEL/` directory
- Generate metrics (accuracy, precision, recall, F1-score, confusion matrix)
- Be importable and usable for inference later

### 1. **Decision Tree** (`CODE/models/dt_model.py`)
- Good baseline, interpretable
- Check feature importance

### 2. **K-Nearest Neighbors** (`CODE/models/knn_model.py`)
- Tune k parameter
- Compare different distance metrics

### 3. **Support Vector Machine** (`CODE/models/svm_model.py`)
- Try different kernels (linear, RBF)
- Optimize hyperparameters

### 4. **Artificial Neural Network** (`CODE/models/ann_model.py`)
- Design network architecture (layers, neurons)
- Tune learning rate and epochs
- Use regularization to prevent overfitting

---

## 📊 Model Persistence

Save trained models so you can reload them later without retraining:

```python
# Save model
import joblib
joblib.dump(model, 'MODEL/dt/decision_tree.pkl')

# Load model
model_loaded = joblib.load('MODEL/dt/decision_tree.pkl')
predictions = model_loaded.predict(test_data)
```

For neural networks, use:
```python
# Save
model.save('MODEL/ann/nn_model.h5')

# Load
from tensorflow.keras.models import load_model
model = load_model('MODEL/ann/nn_model.h5')
```

---

## 📈 Team Member Participation

**Required for final report:**
- Document what each team member did
- Include in DOC/report/
- Detail contributions week-by-week (use PROGRESS.md)

---

## ✅ Submission Checklist

- [ ] All code in `CODE/` follows coding standards
- [ ] All four models implemented (ANN, SVM, DT, KNN)
- [ ] Trained models saved in `MODEL/`
- [ ] Results exported to `OUTPUT/`
- [ ] README.md complete and up-to-date
- [ ] Code comments follow standards (comments on every block)
- [ ] PROGRESS.md shows team participation
- [ ] Python virtual environment documented
- [ ] GitHub repository contains all code
- [ ] Canvas submission includes PPT presentation

---

## 🔍 Troubleshooting

**Virtual environment not activating?**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Import errors?**
```bash
pip install --upgrade -r requirements.txt
```

**Model not loading?**
- Check that `MODEL/` directory exists
- Verify model file path and format (.pkl vs .h5)
- Check Python version compatibility

---

## 📝 Resources

- **Scikit-learn Documentation:** https://scikit-learn.org/
- **Keras/TensorFlow:** https://www.tensorflow.org/
- **Pandas:** https://pandas.pydata.org/
- **Dataset Info:** https://archive.ics.uci.edu/dataset/73/mushroom

---

**Last Updated:** March 21, 2026  
**Repository:** cmpsML_Team_H  
**Team:** Hunter Fernandez, Hunter Merritt
