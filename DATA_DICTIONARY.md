# 📊 Dataset Reference & Data Dictionary

**Dataset:** UCI Mushroom Classification  
**Location:** `INPUT/TRAIN/mushroom.csv`  
**Format:** CSV (comma-separated)  
**Rows:** 8,124 mushroom observations  
**Columns:** 23 features + 1 target variable

---

## 🎯 Target Variable

### `poisonous` (Column 0)
- **Type:** Categorical (Binary)
- **Role:** Classification Target
- **Encoding:**
  - `e` = Edible (safe to eat)
  - `p` = Poisonous (toxic)
- **Class Distribution:** Approximately balanced
- **Missing Values:** None

---

## 📌 Feature Variables (Columns 1-23)

All features are **categorical** and describe the physical attributes of mushrooms.

### **Cap Features** (appearance of mushroom cap)

#### 1. `cap-shape` (Column 1)
- **Type:** Categorical
- **Description:** The shape of the mushroom cap
- **Values:**
  - `b` = Bell
  - `c` = Conical
  - `x` = Convex
  - `f` = Flat
  - `k` = Knobbed
  - `s` = Sunken

#### 2. `cap-surface` (Column 2)
- **Type:** Categorical
- **Description:** Texture of the mushroom cap
- **Values:**
  - `f` = Fibrous
  - `g` = Grooves
  - `y` = Scaly
  - `s` = Smooth

#### 3. `cap-color` (Column 3)
- **Type:** Categorical
- **Description:** Dominant color of the cap
- **Values:**
  - `n` = Brown
  - `b` = Buff (pale yellow)
  - `c` = Cinnamon
  - `g` = Gray
  - `r` = Green
  - `p` = Pink
  - `u` = Purple
  - `e` = Red
  - `w` = White
  - `y` = Yellow

---

### **Gill Features** (gills on underside of cap)

#### 4. `gill-attachment` (Column 4)
- **Type:** Categorical
- **Description:** How gills attach to the stem
- **Values:**
  - `a` = Attached
  - `d` = Descending
  - `f` = Free
  - `n` = Notched

#### 5. `gill-spacing` (Column 5)
- **Type:** Categorical
- **Description:** Distance between gills
- **Values:**
  - `c` = Close
  - `w` = Crowded
  - `d` = Distant

#### 6. `gill-size` (Column 6)
- **Type:** Categorical
- **Description:** Width of individual gills
- **Values:**
  - `b` = Broad (wide)
  - `n` = Narrow

#### 7. `gill-color` (Column 7)
- **Type:** Categorical
- **Description:** Color of the gills
- **Values:**
  - `k` = Black
  - `n` = Brown
  - `b` = Buff
  - `h` = Chocolate
  - `g` = Gray
  - `r` = Green
  - `o` = Orange
  - `p` = Pink
  - `u` = Purple
  - `e` = Red
  - `w` = White
  - `y` = Yellow

---

### **Stalk Features** (stem/stalk of mushroom)

#### 8. `stalk-shape` (Column 8)
- **Type:** Categorical
- **Description:** Shape of the stalk
- **Values:**
  - `e` = Enlarging (wider at top)
  - `t` = Tapering (narrower at bottom)

#### 9. `stalk-root` (Column 9)
- **Type:** Categorical
- **Description:** Shape of the root
- **Values:**
  - `b` = Bulbous (onion-like)
  - `c` = Club
  - `u` = Cup
  - `e` = Equal
  - `z` = Rhizomorphs
  - `r` = Rooted
  - `?` = Missing (⚠️ Handle in preprocessing)
- **Missing Values:** Some entries have `?` - treat as separate category or most common value

#### 10. `stalk-surface-above-ring` (Column 10)
- **Type:** Categorical
- **Description:** Texture of stalk above the ring (veil remnant)
- **Values:**
  - `f` = Fibrous
  - `y` = Scaly
  - `k` = Silky
  - `s` = Smooth

#### 11. `stalk-surface-below-ring` (Column 11)
- **Type:** Categorical
- **Description:** Texture of stalk below the ring
- **Values:**
  - `f` = Fibrous
  - `y` = Scaly
  - `k` = Silky
  - `s` = Smooth

#### 12. `stalk-color-above-ring` (Column 12)
- **Type:** Categorical
- **Description:** Color of stalk above the ring
- **Values:**
  - `n` = Brown
  - `b` = Buff
  - `c` = Cinnamon
  - `g` = Gray
  - `o` = Orange
  - `p` = Pink
  - `e` = Red
  - `w` = White
  - `y` = Yellow

#### 13. `stalk-color-below-ring` (Column 13)
- **Type:** Categorical
- **Description:** Color of stalk below the ring
- **Values:** (Same as above-ring)

---

### **Veil Features** (membrane covering gills in young mushrooms)

#### 14. `veil-type` (Column 14)
- **Type:** Binary Categorical
- **Description:** Type of partial veil
- **Values:**
  - `p` = Partial
  - `u` = Universal
- **Note:** Almost all values are `p` (very little variation)

#### 15. `veil-color` (Column 15)
- **Type:** Categorical
- **Description:** Color of the veil
- **Values:**
  - `n` = Brown
  - `o` = Orange
  - `w` = White
  - `y` = Yellow

---

### **Ring Features** (veil remnant on stalk)

#### 16. `ring-number` (Column 16)
- **Type:** Categorical
- **Description:** Number of rings (partial veils that broke away)
- **Values:**
  - `n` = None
  - `o` = One
  - `t` = Two

#### 17. `ring-type` (Column 17)
- **Type:** Categorical
- **Description:** Shape/type of the ring
- **Values:**
  - `c` = Cobwebby
  - `e` = Evanescent (disappearing)
  - `f` = Flaring
  - `l` = Large
  - `n` = None
  - `p` = Pendant
  - `s` = Sheathing
  - `z` = Zone (double/marked)

---

### **Spore & Reproduction**

#### 18. `spore-print-color` (Column 18)
- **Type:** Categorical
- **Description:** Color of the spore print (reproductive powder left behind)
- **Values:**
  - `k` = Black
  - `n` = Brown
  - `b` = Buff
  - `h` = Chocolate
  - `r` = Green
  - `o` = Orange
  - `u` = Purple
  - `w` = White
  - `y` = Yellow
- **Note:** Important feature - spore color is often correlates with toxicity

---

### **Population & Habitat Features** (where/how it grows)

#### 19. `habitat` (Column 19)
- **Type:** Categorical
- **Description:** Primary habitat where mushroom is found
- **Values:**
  - `g` = Grasses
  - `l` = Leaves (decaying leaves)
  - `m` = Meadows
  - `p` = Paths
  - `u` = Urban (cities, parks)
  - `w` = Waste (dumps, disturbed areas)
  - `d` = Woods (forests)

#### 20. `population` (Column 20)
- **Type:** Categorical
- **Description:** Frequency or abundance in that habitat
- **Values:**
  - `a` = Abundant (very common)
  - `c` = Clustered (grows in groups)
  - `n` = Numerous (many present)
  - `s` = Scattered (spread out)
  - `v` = Several (a few)
  - `y` = Solitary (grows alone)

---

### **Sensory Features**

#### 21. `bruises` (Column 21)
- **Type:** Categorical or Binary
- **Description:** Visible bruising on the mushroom
- **Values:**
  - `t` = Yes, bruises present
  - `f` = No bruises

#### 22. `odor` (Column 22)
- **Type:** Categorical
- **Description:** Smell of the mushroom
- **Values:**
  - `a` = Almond (pleasant, sweet)
  - `l` = Anise (licorice-like)
  - `c` = Creosote (chemical, tarry)
  - `y` = Fishy
  - `f` = Foul (garbage, rotting)
  - `m` = Musty (moldy, damp)
  - `n` = None (no smell)
  - `p` = Pungent (sharp, strong)
  - `s` = Spicy
- **Note:** Odor is often a strong predictor of poisoning

---

## 📈 Data Characteristics

### Class Balance
- **Edible vs. Poisonous:** Roughly equal distribution (good for classification)
- No severe class imbalance issues

### Missing Values
- **Only feature with missing values:** `stalk-root` has `?` values
- **Preprocessing action:** Replace `?` with `missing` category or most frequent value

### Categorical Encoding
- All features are categorical (letters/symbols, not numbers)
- **Preprocessing needed:** One-hot encoding or label encoding for ML algorithms

### Feature Correlations
- Some features likely correlated (e.g., color features)
- Feature selection may help reduce dimensionality

---

## 🔬 Preprocessing Checklist

When you load the data (`CODE/preprocess.py`):

- [ ] Load CSV from `INPUT/TRAIN/mushroom.csv`
- [ ] Replace `?` in `stalk-root` column
- [ ] Verify all 8,124 rows and 23 feature columns loaded
- [ ] Check for any missing values
- [ ] Encode categorical features (One-Hot Encoding recommended)
- [ ] Split features (X) and target (y)
- [ ] Save preprocessing pipeline for later use

---

## 📊 Example Data View

```
poisonous,cap-shape,cap-surface,cap-color,...
p,x,s,n,...
e,x,s,y,...
e,b,s,w,...
p,x,y,w,...
```

**Interpretation (Row 1):**
- Poisonous mushroom (`p`)
- Convex cap shape (`x`)
- Smooth cap surface (`s`)
- Brown cap color (`n`)
- ... (more features)

---

## 💡 Tips for Analysis

1. **Odor feature:** Often strongly predictive - check correlations first
2. **Spore print color:** Another strong predictor
3. **Habitat & population:** May have weak predictive power
4. **Veil type:** Very little variation - consider dropping
5. **Color features:** Multiple color columns may be correlated - watch for multicollinearity

---

## 🔗 Dataset Citation

Mushroom [Dataset]. (1981). UCI Machine Learning Repository. https://doi.org/10.24432/C5959T

**Original Source:** Aurelien Geron's "Hands-On Machine Learning" course

---

**Last Updated:** March 21, 2026

