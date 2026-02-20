

import pandas as pd
import numpy as np
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print(" BOOSTED MODEL TRAINING + VISUALIZATION")
print("="*80)


# STEP 1 — LOAD DATA


df = pd.read_csv("real_person_expenses_1200.csv")

print("\n ORIGINAL DATA:", len(df))


# STEP 2 — CLEANING (BOOSTED)


df['category'] = df['category'].str.lower().str.strip()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

df['desc_clean'] = df['description'].apply(clean_text)

# Remove ultra rare classes
cat_counts = df['category'].value_counts()
valid_cats = cat_counts[cat_counts >= 5].index
df = df[df['category'].isin(valid_cats)]

print(" After Cleaning:", len(df))
print(" Categories:", df['category'].value_counts().to_dict())


# STEP 3 — ENCODING


le = LabelEncoder()
y = le.fit_transform(df['category'])


# STEP 4 — TFIDF (BOOSTED)


vectorizer = TfidfVectorizer(
    max_features=120,
    ngram_range=(1,2),
    stop_words='english',
    min_df=2
)

X_text = vectorizer.fit_transform(df['desc_clean'])

# ==============================
# STEP 5 — AMOUNT FEATURE
# ==============================

amount_mean = df['amount'].mean()
amount_std = df['amount'].std()

X_amount = ((df['amount'] - amount_mean) / amount_std).values.reshape(-1,1)

X = np.hstack([X_text.toarray(), X_amount])

print("Features:", X.shape)

# ==============================
# STEP 6 — TRAIN TEST SPLIT The percentage of each category remains the same in train and test data.
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# STEP 7 — CLASS BALANCING 
# ==============================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# ==============================
# STEP 8 — MODEL (BOOSTED RF)
# ==============================P

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=12,
    min_samples_split=6,
    min_samples_leaf=3,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==============================
# STEP 9 — TEST PERFORMANCE
# ==============================

y_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='weighted')

print("\n TEST ACCURACY:", f"{test_accuracy*100:.2f}%")
print(" TEST F1:", f"{test_f1*100:.2f}%")

# ==============================
# STEP 10 — CROSS VALIDATION
# ==============================

cv_scores = cross_val_score(model, X, y, cv=5)
cv_accuracy = cv_scores.mean()
cv_std = cv_scores.std()

print("\n CROSS VAL:", f"{cv_accuracy*100:.2f}%", "±", f"{cv_std*100:.2f}%")

# ==============================
# STEP 11 — CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, y_pred)

# ==============================
# STEP 12 — VISUALIZATION 
# ==============================

fig, axes = plt.subplots(2,2, figsize=(16,12))

# Accuracy Bar
metrics = ['Test', 'CV Mean', 'CV Min', 'CV Max']
values = [test_accuracy, cv_accuracy, cv_scores.min(), cv_scores.max()]
axes[0,0].bar(metrics, values)
axes[0,0].set_title("Model Accuracy")

# CV Box
axes[0,1].boxplot(cv_scores)
axes[0,1].set_title("Cross Validation")

# Feature Importance
importance = model.feature_importances_
top = np.argsort(importance)[-15:]
axes[1,0].barh(range(15), importance[top])
axes[1,0].set_title("Top Features")

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=axes[1,1])
axes[1,1].set_title("Confusion Matrix")

plt.suptitle(f"Boosted Model Accuracy: {test_accuracy*100:.1f}%")
plt.tight_layout()
plt.savefig("accuracy_proof_teacher.png", dpi=300)
plt.show()

# ==============================
# STEP 13 — SAVE MODEL
# ==============================

joblib.dump(model, "category_model_final.pkl")
joblib.dump(vectorizer, "vectorizer_final.pkl")
joblib.dump(le, "encoder_final.pkl")

print("\n MODEL SAVED")

# ==============================
# STEP 14 — EXTERNAL TEST
# ==============================

print("\n EXTERNAL TEST")

test_data = pd.DataFrame({
'description': [
'uber petrol koramangala','swiggy lunch order','uber cab airport',
'amazon shoes shopping','netflix monthly sub','bescom electricity bill',
'flipkart shirt order','ola bike ride','mcdonalds burger meal',
'hotstar ipl match'],
'amount':[500,250,300,1200,650,2500,800,150,200,300],
'category':['transport','food','transport','shopping','entertainment','bills','shopping','transport','food','entertainment']
})

test_clean = test_data['description'].apply(clean_text)

X_text_test = vectorizer.transform(test_clean)
X_amount_test = ((test_data['amount'] - amount_mean)/amount_std).values.reshape(-1,1)

X_test_final = np.hstack([X_text_test.toarray(), X_amount_test])

y_true = le.transform(test_data['category'])
y_pred_ext = model.predict(X_test_final)

ext_acc = accuracy_score(y_true, y_pred_ext)

print("🎯 EXTERNAL ACCURACY:", f"{ext_acc*100:.2f}%")

print("\n🚀 SUMMARY")
print("Train:", f"{test_accuracy*100:.2f}%")
print("CV:", f"{cv_accuracy*100:.2f}%")
print("External:", f"{ext_acc*100:.2f}%")
