# CodTech Internship Task-4
# Machine Learning Model Implementation - Spam Detection (Simplified Version)
# Created by Himesh Kumar Gupta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Sample dataset (No external file needed)
messages = [
    "Congratulations! You've won a free ticket to Bahamas. Call now!",
    "Hi bro, are we going to the gym today?",
    "Limited offer! Click to win ₹1,00,000 now!",
    "Don't forget our meeting at 5 PM.",
    "You are selected for a free recharge. Reply YES to claim.",
    "Can we reschedule the dinner?",
    "Urgent! Your loan is approved. Call immediately.",
    "Good morning, have a nice day!",
    "Claim your free Netflix subscription now!",
    "I'll call you after the class."
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.3, random_state=42)

# Step 3: Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 5: Predict & evaluate
predictions = model.predict(X_test_vec)

print("✅ Accuracy:", accuracy_score(y_test, predictions))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\n📄 Classification Report:\n", classification_report(y_test, predictions))

# Step 6: Test on custom message
test_msg = ["Congratulations! You won ₹50,000!", "Let's catch up tomorrow."]
test_vec = vectorizer.transform(test_msg)
result = model.predict(test_vec)

print("\n🧪 Sample Predictions:")
for msg, res in zip(test_msg, result):
    print(f"'{msg}' ➡ {'Spam' if res else 'Ham'}")