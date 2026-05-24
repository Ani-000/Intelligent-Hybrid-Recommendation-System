import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ── Load Data ──────────────────────────────────────────────────
users = pd.read_csv("Datasets/users.csv")
feedback = pd.read_csv("Datasets/feedback.csv")

# ── Preprocessing ──────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

users['cleaned_summary'] = users['professional_summary'].apply(preprocess_text)
users['cleaned_about']   = users['about_me'].apply(preprocess_text)
users['combined_text']   = users['cleaned_summary'] + " " + users['cleaned_about']

# ── TF-IDF & Cosine Similarity ─────────────────────────────────
tfidf            = TfidfVectorizer()
tfidf_matrix     = tfidf.fit_transform(users['combined_text'])
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

# ── MBTI & Location Scoring ────────────────────────────────────
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

def get_mbti_score(a, b):
    if a == b:
        return 0.8
    return mbti_compatibility.get((a, b), 0.3)

def get_location_score(a, b):
    return 1.0 if a == b else 0.2

# ── Feedback Training ──────────────────────────────────────────
training_data = []
for _, row in feedback.iterrows():
    uid_a = row['user_id']
    uid_b = row['matched_user_id']
    if uid_a not in users['user_id'].values or uid_b not in users['user_id'].values:
        continue
    idx_a = users[users['user_id'] == uid_a].index[0]
    idx_b = users[users['user_id'] == uid_b].index[0]
    training_data.append([
        cosine_sim_matrix[idx_a][idx_b],
        get_mbti_score(users.loc[idx_a, 'mbti'], users.loc[idx_b, 'mbti']),
        get_location_score(users.loc[idx_a, 'location'], users.loc[idx_b, 'location']),
        row['action']
    ])

train_df = pd.DataFrame(training_data, columns=['text_score', 'mbti_score', 'location_score', 'action'])
X = train_df[['text_score', 'mbti_score', 'location_score']]
y = train_df['action']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# ── Top 5 Matches Function ─────────────────────────────────────
def get_top5_matches(user_id):
    idx = users[users['user_id'] == user_id].index[0]
    scores = []
    for _, other_user in users.iterrows():
        if other_user['user_id'] == user_id:
            continue
        other_idx = other_user.name
        text_score     = cosine_sim_matrix[idx][other_idx]
        mbti_score     = get_mbti_score(users.loc[idx, 'mbti'], other_user['mbti'])
        location_score = get_location_score(users.loc[idx, 'location'], other_user['location'])
        coef  = model.coef_[0]
        total = (coef[0] * text_score) + (coef[1] * mbti_score) + (coef[2] * location_score)
        scores.append({
            'User ID'    : other_user['user_id'],
            'Name'       : other_user['name'],
            'Profession' : other_user['profession'],
            'Location'   : other_user['location'],
            'MBTI'       : other_user['mbti'],
            'Match Score': round(total, 4)
        })
    results = pd.DataFrame(scores).sort_values('Match Score', ascending=False).head(5)
    results = results.reset_index(drop=True)
    results.index += 1
    return results

# ── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(page_title="Profile Matcher", page_icon="", layout="centered")

st.title(" Intelligent Profile Matching System")
st.markdown("*Powered by NLP, MBTI Logic & Adaptive Feedback Learning*")
st.divider()

# User selection
user_ids = users['user_id'].tolist()
selected_id = st.selectbox("Select a User ID to find matches for:", user_ids)

# Show selected user profile
selected_user = users[users['user_id'] == selected_id].iloc[0]
with st.expander("📋 Selected User Profile"):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Name:** {selected_user['name']}")
        st.write(f"**Age:** {selected_user['age']}")
        st.write(f"**Location:** {selected_user['location']}")
        st.write(f"**MBTI:** {selected_user['mbti']}")
    with col2:
        st.write(f"**Profession:** {selected_user['profession']}")
        st.write(f"**Experience:** {selected_user['experience_years']} years")
        st.write(f"**Interests:** {selected_user['interests']}")

st.divider()

# Find matches button
if st.button("🔍 Find Top 5 Matches"):
    with st.spinner("Finding best matches..."):
        results = get_top5_matches(selected_id)
    st.success("Top 5 Matches Found!")
    st.dataframe(results, use_container_width=True)