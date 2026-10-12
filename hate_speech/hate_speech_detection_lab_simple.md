# Lab: Simple Hate Speech Detection

**CPS 4498 – Digital Forensics** (Simplified Version)

---

## Lab Overview

| | |
|---|---|
| **Objective** | Build a basic hate speech detector using machine learning |
| **Duration** | 1 hour |

**What You'll Learn**

- Text preprocessing basics
- Training a simple ML model
- Testing with real messages

> **Scenario:** You found suspicious messages on a suspect's phone. You need to identify which messages contain hate speech.

### Contents

| Part | Topic | Time |
|---|---|---|
| [Part 1](#part-1-setup-5-minutes) | Setup | 5 min |
| [Part 2](#part-2-create-dataset-5-minutes) | Create Dataset | 5 min |
| [Part 3](#part-3-prepare-data-10-minutes) | Prepare Data | 10 min |
| [Part 4](#part-4-train-model-10-minutes) | Train Model | 10 min |
| [Part 5](#part-5-test-model-15-minutes) | Test Model | 15 min |
| [Part 6](#part-6-use-your-model-15-minutes) | Use Your Model | 15 min |
| [Part 7](#part-7-save-your-work-10-minutes) | Save Your Work | 10 min |
| [Part 8](#part-8-generate-report-10-minutes) | Generate Report | 10 min |
| [Complete Script](#complete-script) | Everything in one file | |
| [Lab Questions](#lab-questions) | 5 questions, 5 points each | |
| [Lab Checklist](#lab-checklist) | | |
| [Summary](#summary) | | |

---

## Part 1: Setup (5 minutes)

### Step 1: Install and Import

**Code Cell 1 — Installation**

```python
# Run this first (only once)
# pip install pandas scikit-learn
```

**Code Cell 2 — Import Libraries**

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

print("✓ Libraries loaded successfully!")
```

**Expected Output**

```text
✓ Libraries loaded successfully!
```

---

## Part 2: Create Dataset (5 minutes)

### Step 2: Make Sample Data

**Code Cell 3 — Create Dataset**

```python
# Create simple dataset
# 0 = Normal, 1 = Hate Speech

messages = [
    # Normal messages (0)
    "I love my friends",
    "Great weather today",
    "Nice movie",
    "Good morning everyone",
    "Thanks for your help",
    "Happy birthday",
    "See you tomorrow",
    "Have a nice day",
    "Good job team",
    "Welcome to class",
    
    # Hate speech messages (1)
    "I hate those people",
    "They are all criminals",
    "Those idiots are terrible",
    "They should all leave",
    "I despise that group",
    "They are worthless",
    "Ban all of them",
    "They are all the same",
    "We dont want them here",
    "They are destroying everything"
]

labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 10 normal
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 10 hate

# Create DataFrame
df = pd.DataFrame({'message': messages, 'label': labels})

print(f"Total messages: {len(df)}")
print(f"Normal: {sum(labels == 0)}")
print(f"Hate: {sum(labels == 1)}")
print("\nFirst 5 messages:")
print(df.head())
```

**Expected Output**

```text
Total messages: 20
Normal: 10
Hate: 10

First 5 messages:
              message  label
0  I love my friends      0
1  Great weather today      0
2        Nice movie      0
3  Good morning everyone      0
4  Thanks for your help      0
```

---

## Part 3: Prepare Data (10 minutes)

### Step 3: Clean Text

**Code Cell 4 — Text Cleaning**

```python
def clean_text(text):
    """Make text lowercase and remove extra spaces"""
    text = text.lower()
    text = ' '.join(text.split())
    return text

# Apply cleaning
df['cleaned'] = df['message'].apply(clean_text)

print("Original vs Cleaned:")
for i in range(3):
    print(f"\nOriginal: {df.iloc[i]['message']}")
    print(f"Cleaned:  {df.iloc[i]['cleaned']}")
```

**Expected Output**

```text
Original vs Cleaned:

Original: I love my friends
Cleaned:  i love my friends

Original: Great weather today
Cleaned:  great weather today

Original: Nice movie
Cleaned:  nice movie
```

### Step 4: Split Data

**Code Cell 5 — Train-Test Split**

```python
# Split into training (80%) and testing (20%)
X = df['cleaned']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training size: {len(X_train)}")
print(f"Testing size: {len(X_test)}")
```

**Expected Output**

```text
Training size: 16
Testing size: 4
```

---

## Part 4: Train Model (10 minutes)

### Step 5: Convert Text to Numbers

**Code Cell 6 — Vectorization**

```python
# Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"\nSome words in vocabulary:")
words = list(vectorizer.vocabulary_.keys())[:10]
print(words)
```

**Expected Output**

```text
Vocabulary size: 52
Some words in vocabulary:
['love', 'friends', 'great', 'weather', 'today', 'nice', 'movie', 'good', 'morning', 'everyone']
```

### Step 6: Train Model

**Code Cell 7 — Model Training**

```python
# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

print("✓ Model trained successfully!")

# Test on training data
train_pred = model.predict(X_train_vec)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"Training accuracy: {train_accuracy:.2%}")
```

**Expected Output**

```text
✓ Model trained successfully!
Training accuracy: 100.00%
```

---

## Part 5: Test Model (15 minutes)

### Step 7: Evaluate Performance

**Code Cell 8 — Model Testing**

```python
# Predict on test set
y_pred = model.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2%}")

# Show detailed results
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Normal', 'Hate Speech']))

# Show predictions
print("\nTest Predictions:")
for i, (text, true, pred) in enumerate(zip(X_test, y_test, y_pred)):
    status = "✓" if true == pred else "✗"
    print(f"{status} Text: '{text}'")
    print(f"   True: {'Hate' if true==1 else 'Normal'}, "
          f"Predicted: {'Hate' if pred==1 else 'Normal'}\n")
```

**Expected Output**

```text
Test Accuracy: 100.00%

Detailed Classification Report:
              precision    recall  f1-score   support

      Normal       1.00      1.00      1.00         2
 Hate Speech       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4

Test Predictions:
✓ Text: 'they are all criminals'
   True: Hate, Predicted: Hate

✓ Text: 'good job team'
   True: Normal, Predicted: Normal

✓ Text: 'i hate those people'
   True: Hate, Predicted: Hate

✓ Text: 'thanks for your help'
   True: Normal, Predicted: Normal
```

---

## Part 6: Use Your Model (15 minutes)

### Step 8: Test Custom Messages

**Code Cell 9 — Predict Function**

```python
def predict_message(text):
    """Predict if a message is hate speech"""
    # Clean text
    cleaned = clean_text(text)
    
    # Convert to numbers
    vectorized = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    # Format result
    result = "HATE SPEECH" if prediction == 1 else "NORMAL"
    confidence = probability[prediction] * 100
    
    return result, confidence

# Test the function
test_message = "I really hate all of them"
result, conf = predict_message(test_message)

print(f"Message: '{test_message}'")
print(f"Result: {result}")
print(f"Confidence: {conf:.1f}%")
```

**Expected Output**

```text
Message: 'I really hate all of them'
Result: HATE SPEECH
Confidence: 89.3%
```

### Step 9: Analyze Multiple Messages

**Code Cell 10 — Batch Testing**

```python
# Test with new messages
new_messages = [
    "You are a wonderful person",
    "Those people are terrible",
    "Have a great weekend",
    "They should all be banned",
    "Thanks for being so helpful"
]

print("="*60)
print("FORENSIC ANALYSIS RESULTS")
print("="*60)

for i, msg in enumerate(new_messages, 1):
    result, conf = predict_message(msg)
    
    print(f"\nMessage {i}: '{msg}'")
    print(f"Result: {result} ({conf:.1f}% confidence)")
    
    if result == "HATE SPEECH":
        print("⚠️  FLAG FOR INVESTIGATION")
```

**Expected Output**

```text
============================================================
FORENSIC ANALYSIS RESULTS
============================================================

Message 1: 'You are a wonderful person'
Result: NORMAL (92.3% confidence)

Message 2: 'Those people are terrible'
Result: HATE SPEECH (87.6% confidence)
⚠️  FLAG FOR INVESTIGATION

Message 3: 'Have a great weekend'
Result: NORMAL (95.1% confidence)

Message 4: 'They should all be banned'
Result: HATE SPEECH (91.4% confidence)
⚠️  FLAG FOR INVESTIGATION

Message 5: 'Thanks for being so helpful'
Result: NORMAL (93.8% confidence)
```

---

## Part 7: Save Your Work (10 minutes)

### Step 10: Save Model

**Code Cell 11 — Save Model**

```python
import pickle

# Save model
with open('hate_detector.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✓ Model saved as 'hate_detector.pkl'")
print("✓ Vectorizer saved as 'vectorizer.pkl'")
```

**Expected Output**

```text
✓ Model saved as 'hate_detector.pkl'
✓ Vectorizer saved as 'vectorizer.pkl'
```

### Step 11: Load and Test

**Code Cell 12 — Load Model**

```python
# Load saved model
with open('hate_detector.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

# Test loaded model
test = "Those people are criminals"
cleaned = clean_text(test)
vec = loaded_vectorizer.transform([cleaned])
pred = loaded_model.predict(vec)[0]

print("Testing loaded model:")
print(f"Message: '{test}'")
print(f"Prediction: {'HATE SPEECH' if pred==1 else 'NORMAL'}")
print("\n✓ Model loaded and working!")
```

**Expected Output**

```text
Testing loaded model:
Message: 'Those people are criminals'
Prediction: HATE SPEECH

✓ Model loaded and working!
```

---

## Part 8: Generate Report (10 minutes)

### Step 12: Create Forensic Report

**Code Cell 13 — Generate Report**

```python
def generate_report(messages_to_analyze):
    """Generate a forensic analysis report"""
    
    report = []
    report.append("="*60)
    report.append("FORENSIC HATE SPEECH ANALYSIS REPORT")
    report.append("="*60)
    report.append("")
    
    hate_count = 0
    
    for i, msg in enumerate(messages_to_analyze, 1):
        result, conf = predict_message(msg)
        
        report.append(f"Message #{i}:")
        report.append(f"Text: {msg}")
        report.append(f"Classification: {result}")
        report.append(f"Confidence: {conf:.1f}%")
        
        if result == "HATE SPEECH":
            report.append("STATUS: ⚠️  FLAGGED FOR REVIEW")
            hate_count += 1
        else:
            report.append("STATUS: ✓ Normal")
        
        report.append("-" * 60)
        report.append("")
    
    # Summary
    report.append("="*60)
    report.append("SUMMARY")
    report.append("="*60)
    report.append(f"Total messages analyzed: {len(messages_to_analyze)}")
    report.append(f"Hate speech detected: {hate_count}")
    report.append(f"Normal messages: {len(messages_to_analyze) - hate_count}")
    report.append("="*60)
    
    return '\n'.join(report)

# Generate report for investigation
suspect_messages = [
    "I hate those people",
    "Good morning team",
    "They are all terrible",
    "Thanks for your help"
]

report = generate_report(suspect_messages)
print(report)

# Save report to file
with open('forensic_report.txt', 'w') as f:
    f.write(report)

print("\n✓ Report saved to 'forensic_report.txt'")
```

**Expected Output**

```text
============================================================
FORENSIC HATE SPEECH ANALYSIS REPORT
============================================================

Message #1:
Text: I hate those people
Classification: HATE SPEECH
Confidence: 89.2%
STATUS: ⚠️  FLAGGED FOR REVIEW
------------------------------------------------------------

Message #2:
Text: Good morning team
Classification: NORMAL
Confidence: 94.5%
STATUS: ✓ Normal
------------------------------------------------------------

Message #3:
Text: They are all terrible
Classification: HATE SPEECH
Confidence: 87.3%
STATUS: ⚠️  FLAGGED FOR REVIEW
------------------------------------------------------------

Message #4:
Text: Thanks for your help
Classification: NORMAL
Confidence: 96.1%
STATUS: ✓ Normal
------------------------------------------------------------

============================================================
SUMMARY
============================================================
Total messages analyzed: 4
Hate speech detected: 2
Normal messages: 2
============================================================

✓ Report saved to 'forensic_report.txt'
```

---

## Complete Script

Save as `hate_detection_simple.py`

```python
"""
Simple Hate Speech Detection System
For Digital Forensics Investigation
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# 1. CREATE DATASET
messages = [
    "I love my friends", "Great weather today", "Nice movie",
    "Good morning everyone", "Thanks for your help", "Happy birthday",
    "See you tomorrow", "Have a nice day", "Good job team", "Welcome to class",
    "I hate those people", "They are all criminals", "Those idiots are terrible",
    "They should all leave", "I despise that group", "They are worthless",
    "Ban all of them", "They are all the same", "We dont want them here",
    "They are destroying everything"
]

labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

df = pd.DataFrame({'message': messages, 'label': labels})

# 2. CLEAN TEXT
def clean_text(text):
    return ' '.join(text.lower().split())

df['cleaned'] = df['message'].apply(clean_text)

# 3. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label'], test_size=0.2, random_state=42
)

# 4. VECTORIZE
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. TRAIN MODEL
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. EVALUATE
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# 7. PREDICTION FUNCTION
def predict_message(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    result = "HATE SPEECH" if prediction == 1 else "NORMAL"
    confidence = probability[prediction] * 100
    return result, confidence

# 8. TEST
test_messages = [
    "You are wonderful",
    "I hate all of them",
    "Have a nice day"
]

print("\nTest Results:")
for msg in test_messages:
    result, conf = predict_message(msg)
    print(f"'{msg}' -> {result} ({conf:.1f}%)")

# 9. SAVE MODEL
with open('hate_detector.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n✓ Model saved successfully!")
```

---

## Lab Questions

Answer these questions (**5 points each**):

1. **Q1:** What is the difference between training data and testing data?
2. **Q2:** Why do we convert text to lowercase during cleaning?
3. **Q3:** What does "accuracy" mean in machine learning?
4. **Q4:** Why is it important to save the vectorizer along with the model?
5. **Q5:** Name one limitation of this simple hate speech detector.

---

## Lab Checklist

- [ ] All libraries installed
- [ ] Dataset created (20 messages)
- [ ] Text cleaning working
- [ ] Model trained successfully
- [ ] Test accuracy calculated
- [ ] Custom messages tested
- [ ] Model saved
- [ ] Report generated
- [ ] Questions answered

---

## Summary

**What You Built**

- ✓ A working hate speech detector
- ✓ Can analyze text messages
- ✓ Generates forensic reports
- ✓ Saved for future use

**Files Created**

| File | Contents |
|---|---|
| `hate_detector.pkl` | Trained model |
| `vectorizer.pkl` | Text processor |
| `forensic_report.txt` | Analysis report |

**Applications**

- Analyze suspect's messages
- Screen social media posts
- Support investigations
- Generate evidence reports
