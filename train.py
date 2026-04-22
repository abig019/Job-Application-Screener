import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import numpy as np # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore #Used for classification problems - 0 or 1
from sklearn.ensemble import RandomForestClassifier  # type: ignore #multiple decision trees 
from sklearn.model_selection import train_test_split   # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score,  #Overall correctness
    confusion_matrix,  #Shows detailed result
    classification_report,  
    precision_score,
    recall_score,
    f1_score
)
import pickle
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore


# ══════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════

df = pd.read_csv('Job_Placement_Data.csv')
print("✅ Data loaded")
print(f"Shape: {df.shape}")
print()

# WHY? We always load first and confirm shape.
# Your explore.py already told us: 215 rows, 13 columns.
# We confirm again here because train.py is independent.

# ══════════════════════════════════════════════════════════
# STEP 2 — ENCODE TEXT COLUMNS
# ══════════════════════════════════════════════════════════

# WHY encode? ML models only understand numbers.
# They cannot process the word "Placed" or "Male".
# LabelEncoder converts text → numbers automatically.
# Example: Placed → 1, Not Placed → 0
#          Male → 1, Female → 0

le_dict = {}
# WHY a dictionary? You have 7 text columns.
# You need to save each column's encoder separately
# so app.py can use them later to encode user input.
# If you only saved one encoder you'd lose the others.

text_columns = [
    'gender', 'ssc_board', 'hsc_board', 'hsc_subject',
    'undergrad_degree', 'work_experience', 'specialisation', 'status'
]
# WHY list them manually? Safer than auto-detecting.
# You know exactly which columns are text from explore.py output.

for col in text_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
    print(f"Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

print()

# Save all encoders in one file
with open('encoders.pkl', 'wb') as f:
    pickle.dump(le_dict, f)
print("✅ encoders.pkl saved")
print()

# WHY pkl (pickle)? It saves a Python object exactly as-is
# to a file. When app.py loads it, it gets the same encoder
# object back — same mapping, same classes. Like freezing
# and unfreezing the object.

# ══════════════════════════════════════════════════════════
# STEP 3 — SPLIT FEATURES AND TARGET
# ══════════════════════════════════════════════════════════

X = df.drop('status', axis=1)   # everything EXCEPT status
y = df['status']                 # ONLY status

# WHY X and y? This is universal ML naming convention.
# X = inputs  (what you know)  → 12 columns
# y = output  (what to predict) → 1 column: status
# Think of it as: X predicts y.

print("Features used to predict:")
print(list(X.columns))
print(f"Target: status (0 = Not Placed, 1 = Placed)")
print()


# ══════════════════════════════════════════════════════════
# STEP 4 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing = 43 rows
    random_state=42   # fixes the random split so results are repeatable
)

print(f"Training rows : {len(X_train)}")  # 172 rows
print(f"Testing rows  : {len(X_test)}")   # 43 rows
print()

# WHY split? You cannot test on the same data you trained on.
# That's like giving a student the exam questions in advance —
# they score 100% but learned nothing.
# The test set is data the model has NEVER seen.
# If it scores well there, the model genuinely learned.

# WHY 80/20? Industry standard for small datasets.
# Larger datasets (100k+ rows) often use 90/10.

# ══════════════════════════════════════════════════════════
# STEP 5 — TRAIN LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc  = accuracy_score(y_test, lr_pred)

print(f"Logistic Regression accuracy: {lr_acc:.4f}")

# WHY Logistic Regression for classification?
# Despite the name it's a CLASSIFIER, not a regressor.
# It calculates the probability of each class.
# Output: 0.82 → 82% chance = Placed → predicts Placed
# It draws a decision boundary — a line that separates
# Placed from Not Placed in the feature space.
# Simple, fast, and very explainable in interviews.

# WHY max_iter=1000? The algorithm improves in steps (iterations).
# Default is 100 — sometimes not enough for it to converge.
# 1000 gives it enough steps to find the best answer.


# WHY Logistic Regression for classification?
# Despite the name it's a CLASSIFIER, not a regressor.
# It calculates the probability of each class.
# Output: 0.82 → 82% chance = Placed → predicts Placed
# It draws a decision boundary — a line that separates
# Placed from Not Placed in the feature space.
# Simple, fast, and very explainable in interviews.

# WHY max_iter=1000? The algorithm improves in steps (iterations).
# Default is 100 — sometimes not enough for it to converge.
# 1000 gives it enough steps to find the best answer.

# ══════════════════════════════════════════════════════════
# STEP 6 — TRAIN RANDOM FOREST
# ══════════════════════════════════════════════════════════

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)

print(f"Random Forest accuracy      : {rf_acc:.4f}")
print()

# WHY Random Forest?
# It builds 100 decision trees (n_estimators=100).
# Each tree votes: Placed or Not Placed?
# Final answer = majority vote across all 100 trees.
# WHY is this better than one tree?
# One tree overfits — it memorises training data.
# 100 trees average out each other's mistakes.
# This is called ENSEMBLE learning — many weak learners
# combine into one strong learner.

# WHY train BOTH models?
# You don't know in advance which will perform better.
# Training both and comparing is the real industry workflow.
#Ensemble Learning means:Using multiple models together instead of just one
# You pick the winner based on test data, not assumption.

# WHY train BOTH models?
# You don't know in advance which will perform better.
# Training both and comparing is the real industry workflow.
# You pick the winner based on test data, not assumption.

# ══════════════════════════════════════════════════════════
# STEP 7 — PICK THE BEST MODEL
# ══════════════════════════════════════════════════════════

if rf_acc >= lr_acc:
    best_model = rf
    best_pred  = rf_pred
    best_name  = "Random Forest"
else:
    best_model = lr
    best_pred  = lr_pred
    best_name  = "Logistic Regression"

print(f"Best model: {best_name}")
print()

# ══════════════════════════════════════════════════════════
# STEP 8 — EVALUATE WITH FULL METRICS
# ══════════════════════════════════════════════════════════

print("── Classification Report ───────────────────────────")
print(classification_report(y_test, best_pred,
      target_names=['Not Placed', 'Placed']))
print()

# WHY not just accuracy?
# Imagine 190 Placed and 25 Not Placed in your dataset.
# A model that ALWAYS predicts Placed gets 88% accuracy
# — but it's completely useless! It never catches rejections.
# That's why you need these 3 extra metrics:

# PRECISION — of everyone the model said "Placed",
#             how many were actually Placed?
#             High precision = few false positives (false alarms)

# RECALL    — of everyone who was actually Placed,
#             how many did the model correctly find?
#             High recall = few false negatives (missed cases)

# F1 SCORE  — the balance between precision and recall.
#             Use this as your single go-to metric when
#             classes are imbalanced (unequal Placed vs Not Placed).

print(f"Accuracy  : {accuracy_score(y_test, best_pred):.4f}")
print(f"Precision : {precision_score(y_test, best_pred):.4f}")
print(f"Recall    : {recall_score(y_test, best_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, best_pred):.4f}")
print()

# ══════════════════════════════════════════════════════════
# STEP 9 — CONFUSION MATRIX
# ══════════════════════════════════════════════════════════

cm = confusion_matrix(y_test, best_pred)
print("── Confusion Matrix ────────────────────────────────")
print(cm)
print()

# HOW TO READ IT:
# Rows = what actually happened
# Columns = what the model predicted
#
#                  Predicted:      Predicted:
#                  Not Placed      Placed
# Actual: Not Placed  [ TN ]       [ FP ]
# Actual: Placed      [ FN ]       [ TP ]
#
# TN = True Negative  → correctly said Not Placed ✅
# TP = True Positive  → correctly said Placed     ✅
# FP = False Positive → said Placed, was Not Placed ❌ (false alarm)
# FN = False Negative → said Not Placed, was Placed ❌ (missed)
#
# In HR context: FN is worse — you rejected a good candidate.
# In fraud detection: FP is worse — you let fraud through.
# The context decides which error matters more.

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Placed', 'Placed'],
            yticklabels=['Not Placed', 'Placed'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix — {best_name}')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("✅ confusion_matrix.png saved")
print()

# ══════════════════════════════════════════════════════════
# STEP 10 — FEATURE IMPORTANCE (Random Forest only)
# ══════════════════════════════════════════════════════════

if best_name == "Random Forest":
    importances = best_model.feature_importances_
    # WHY feature_importances_? Random Forest tracks how much
    # each feature reduced prediction error across all 100 trees.
    # Higher score = that column mattered more to the prediction.
    # This is one of the most useful things to show in interviews —
    # "my model says degree percentage and MBA score matter most."

    feat_df = pd.DataFrame({
        'feature'   : X.columns,
        'importance': importances
    }).sort_values('importance', ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(feat_df['feature'], feat_df['importance'], color='steelblue')
    plt.xlabel('Importance score')
    plt.title('Which factors affect placement most?')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    print("✅ feature_importance.png saved")
    print()

    # ══════════════════════════════════════════════════════════
# STEP 11 — SAVE EVERYTHING
# ══════════════════════════════════════════════════════════

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

with open('model_name.pkl', 'wb') as f:
    pickle.dump(best_name, f)

print(f"✅ model.pkl saved  ({best_name})")
print(f"✅ feature_names.pkl saved")
print(f"✅ model_name.pkl saved")
print()
print("═══════════════════════════════════")
print("  Training complete. Run app.py next.")
print("═══════════════════════════════════")


with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

with open('model_name.pkl', 'wb') as f:
    pickle.dump(best_name, f)

print(f"✅ model.pkl saved  ({best_name})")
print(f"✅ feature_names.pkl saved")
print(f"✅ model_name.pkl saved")