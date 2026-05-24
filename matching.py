from posixpath import split

import pandas as pd
import pip

# Loading the datasets
users = pd.read_csv("D:\\unlox\\MajorProject\\Datasets\\users.csv")
feedback = pd.read_csv("D:\\unlox\\MajorProject\\Datasets\\feedback.csv")

print("Users Dataset loaded:", users.shape) 
print("Feedback Dataset Loaded:", feedback.shape) 
# The .shape() function is called to understad the rows and columns of the dataset.

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Dowloading the required NLTK data (only runs once)
nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet = True)

# Inititalzie tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Step 1: Lowercase everything
    text = text.lower()
    # Step 2: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Step 3: Remove stopwords and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Apply preprocessing to the feedback comments
users['cleaned_summary'] = users['professional_summary'].apply(preprocess_text)
users['cleaned_about'] = users['about_me'].apply(preprocess_text)

# Combine both into one rich text field per user
users['combined_text'] = users['cleaned_summary'] + " " + users['cleaned_about']

print("\nPreprocessing done!")
print(users[['user_id', 'combined_text']].head(3))

# Perform TF-IDF vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert combined text into numerical vectors
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(users['combined_text'])
print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)

# Cosine Similarity Calcuation
from sklearn.metrics.pairwise import cosine_similarity

# Similarity between all users
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

print("\nCosine Similarity Matrix Shape:", cosine_sim_matrix.shape)

# Similarity between user1 and user2
user1_idx = 0 # U001
user2_idx = 1 # U002
score = cosine_sim_matrix[user1_idx][user2_idx]
print(f"Text Similarity between User1 and User2: {round(score,4)}")

# MBTI Compatibility Matrix
mbti_compatibility = {
    ("INTJ", "ENFP"): 1.0, ("ENFP", "INTJ"): 1.0,
    ("INTP", "ENFJ"): 1.0, ("ENFJ", "INTP"): 1.0,
    ("ENTJ", "INFP"): 1.0, ("INFP", "ENTJ"): 1.0,
    ("ENTP", "INFJ"): 1.0, ("INFJ", "ENTP"): 1.0,
    ("ISTJ", "ESFP"): 1.0, ("ESFP", "ISTJ"): 1.0,
    ("ISFJ", "ESTP"): 1.0, ("ESTP", "ISFJ"): 1.0,
    ("ESTJ", "ISFP"): 1.0, ("ISFP", "ESTJ"): 1.0,
    ("ESFJ", "ISTP"): 1.0, ("ISTP", "ESFJ"): 1.0,
}

def get_mbti_score(mbti_a, mbti_b) :
    # If same type, then good match
    if mbti_a == mbti_b:
        return 0.8
    # Checking the compatibity dictionary
    score = mbti_compatibility.get((mbti_a, mbti_b), 0.3)
    return score
# Testing the MBTI compatibility function
print("\nMBTI Score (INTJ vs ENFP):", get_mbti_score("INTJ", "ENFP"))
print("MBTI Score (INTJ vs INTJ):", get_mbti_score("INTJ", "INTJ"))
print("MBTI Score (INTJ vs ISTJ):", get_mbti_score("INTJ", "ISTJ"))

# Location matching function
def get_location_score(location_a, location_b):
    if location_a == location_b:
        return 1.0 # Since same location is a perfect match
    else:
        return 0.2 # Different location will recieve lower score
# Testing the location score function
print("\nLocation Score (Mumbai vs Mumbai):", get_location_score("Mumbai", "Mumbai"))
print("Location Score (Mumbai vs Delhi):", get_location_score("Mumbai", "Delhi"))

# Setting default weights for each component
w1 = 0.5 # For text similarity
w2 = 0.3 # For MBTI compatibility
w3 = 0.2 # For location matching

def get_compatibility_score(user_id_a, user_id_b):
    # Getting the index of both the users
    idx_a = users[users['user_id'] == user_id_a].index[0]
    idx_b = users[users['user_id'] == user_id_b].index[0]
    # Getting each individual score
    text_score = cosine_sim_matrix[idx_a][idx_b]
    mbti_score = get_mbti_score(users.loc[idx_a,'mbti'], users.loc[idx_b,'mbti'])
    location_score = get_location_score(users.loc[idx_a, 'location'], users.loc[idx_b, 'location'])
    # Weighted formula:
    total = (w1 * text_score) + (w2 * mbti_score) + (w3 * location_score)
    # Converting to percentage
    total_percent = round(total * 100, 2)
    return total_percent
# Testing the final compatibility score function
print("\nCompatibility Score (U001 vs U002):", get_compatibility_score("U001", "U002"), "%")
print("Compatibility Score (U001 vs U003):", get_compatibility_score("U001", "U003"), "%") 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Training data from feedback
training_data = []
for _, row in feedback.iterrows():
    uid_a = row['user_id']
    uid_b = row['matched_user_id']
    action = row['action']
    if uid_a not in users['user_id'].values or uid_b not in users['user_id'].values:
        continue

    # To get the index position of the user
    idx_a = users[users['user_id'] == uid_a].index[0]
    idx_b = users[users['user_id'] == uid_b].index[0]

    # Calculate the 3 feature scores
    text_score     = cosine_sim_matrix[idx_a][idx_b]
    mbti_score     = get_mbti_score(users.loc[idx_a, 'mbti'], users.loc[idx_b, 'mbti'])
    location_score = get_location_score(users.loc[idx_a, 'location'], users.loc[idx_b, 'location'])

    training_data.append([text_score, mbti_score, location_score, action])

# Convert to DataFrame
train_df = pd.DataFrame(training_data, columns=['text_score', 'mbti_score', 'location_score', 'action'])

print("\nTraining Data Shape:", train_df.shape)
print(train_df.head())

from sklearn.linear_model import LogisticRegression
# Split features and target
X = train_df[['text_score', 'mbti_score', 'location_score']]
y = train_df['action']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Complete!")
print(f"Model Accuracy: {round(accuracy * 100, 2)}%")
print(f"Learned Coefficients: {model.coef_[0]}")

def get_top5_matches(user_id):
    # Check if user exists
    if user_id not in users['user_id'].values:
        print("User not found!")
        return

    # Get index of the input user
    idx = users[users['user_id'] == user_id].index[0]

    scores = []

    for _, other_user in users.iterrows():
        # Skip comparing user with themselves
        if other_user['user_id'] == user_id:
            continue

        other_idx = other_user.name

        # Calculate all 3 scores
        text_score     = cosine_sim_matrix[idx][other_idx]
        mbti_score     = get_mbti_score(users.loc[idx, 'mbti'], other_user['mbti'])
        location_score = get_location_score(users.loc[idx, 'location'], other_user['location'])

        # Use learned coefficients instead of manual weights
        coef = model.coef_[0]
        total = (coef[0] * text_score) + (coef[1] * mbti_score) + (coef[2] * location_score)

        scores.append({
            'user_id'       : other_user['user_id'],
            'name'          : other_user['name'],
            'profession'    : other_user['profession'],
            'location'      : other_user['location'],
            'mbti'          : other_user['mbti'],
            'match_score'   : round(total, 4)
        })

    # Sort by score and return top 5
    results = pd.DataFrame(scores).sort_values('match_score', ascending=False).head(5)
    results = results.reset_index(drop=True)
    results.index += 1  # Start ranking from 1

    print(f"\nTop 5 Matches for {user_id}:")
    print(results.to_string())

# Test it
get_top5_matches("U001")

#PERFORMANCE ANALYSIS REPORT

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# ── Before Training (Manual Weights) ──────────────────────────
# Predict using our manual weights w1=0.5, w2=0.3, w3=0.2
manual_predictions = []
for _, row in zip(y_test.index, y_test):
    text  = X_test.loc[_]['text_score']
    mbti  = X_test.loc[_]['mbti_score']
    loc   = X_test.loc[_]['location_score']
    score = (0.5 * text) + (0.3 * mbti) + (0.2 * loc)
    manual_predictions.append(1 if score >= 0.5 else 0)

manual_accuracy = accuracy_score(y_test, manual_predictions)

# ── After Training (Learned Weights) ──────────────────────────
learned_accuracy = accuracy_score(y_test, y_pred)

# ── Print Report ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("PERFORMANCE ANALYSIS REPORT")
print("=" * 50)
print(f"Accuracy BEFORE Feedback Training : {round(manual_accuracy * 100, 2)}%")
print(f"Accuracy AFTER  Feedback Training : {round(learned_accuracy * 100, 2)}%")
print(f"Improvement                       : {round((learned_accuracy - manual_accuracy) * 100, 2)}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# ── Plot Graph ─────────────────────────────────────────────────
labels   = ['Before Training\n(Manual Weights)', 'After Training\n(Learned Weights)']
accuracy = [round(manual_accuracy * 100, 2), round(learned_accuracy * 100, 2)]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, accuracy, color=['#ff6b6b', '#51cf66'], width=0.4)
plt.ylim(0, 100)
plt.title('Feedback Loop Performance Analysis', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')

# Add value labels on bars
for bar, val in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val}%', ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('performance_analysis.png')
plt.show()
print("\nGraph saved as performance_analysis.png")