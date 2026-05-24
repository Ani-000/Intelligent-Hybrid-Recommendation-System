=================================================================
INTELLIGENT HYBRID RECOMMENDATION SYSTEM
Profile-Based Matching Algorithm
=================================================================

DEMO LINK:
--------------------------------------------------------------------------------------------------
https://intelligent-hybrid-recommendation-system.streamlit.app/
--------------------------------------------------------------------------------------------------

PROJECT OVERVIEW:
----------------------------------------------------------------------------------------------------
This project builds an intelligent profile matching system that recommends the top 5 most compatible users based on a combination of NLP similarity, MBTI personality compatibility, and location matching. The system also learns from user feedback to improve its recommendations over time.
----------------------------------------------------------------------------------------------------
DATASET DESCRIPTION:
----------------------------------------------------------------------------------------------------
The dataset was generated to simulate the real-world user profiles for an intelligent hybrid recommendation system.
users.csv - contains all the profiles with profession, MBTI score, location, and free-text bios.
feedback.csv - contains all the user interactions used to train the adaptive feedback model.
----------------------------------------------------------------------------------------------------
TECHNICAL MODULES:
--------------------------------------------------------------------------------------------------
Module 1 : NLP & Semantic Analysis
- Text cleaned using NLTK.
- Converted to vectors using TF-IDF.
- Similarity measured using Cosine Similarity.
Module 2 : Profile Scoring Engine
- MBTI compatibility matrix.
- Location matching score.
- Final formula:
	TotalScore = (w1 * TextSim) + (w2 * MBTIMatch) + (w3 * Location)
Module 3 - Adaptive Feedback Loop
- Logistic Regression trained on all the feedback interactions.
- Accuracy improved from 57.14% to 61.34% after training.
- Model automatically learns which features matter most.
--------------------------------------------------------------------------------------------------
LIBRARIES USED:
--------------------------------------------------------------------------------------------------
- Pandas -> Data manipulation.
- nltk -> Text preprocessing
- scikit-learn -> TF-IDF, Cosine Similarity, Logistic Regression
- streamlit -> UI Demo
- matplotlib -> Performance analysis graph
==================================================================================================
