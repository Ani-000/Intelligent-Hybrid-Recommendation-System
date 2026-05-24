import pandas as pd
import random
from datetime import datetime, timedelta

# Load users
users = pd.read_csv("D:\\unlox\\MajorProject\\Dataset\\users.csv")
user_ids = users["user_id"].tolist()

data = []

for user in user_ids:
    interactions = random.randint(5, 8)
    possible_matches = [u for u in user_ids if u != user]
    matches = random.sample(possible_matches, interactions)
    
    for match in matches:
        user_prof = users.loc[users["user_id"] == user, "profession"].values[0]
        match_prof = users.loc[users["user_id"] == match, "profession"].values[0]
        
        # Logical matching
        if user_prof == match_prof:
            action = random.choices([1, 0], weights=[0.65, 0.35])[0]
        else:
            action = random.choices([1, 0], weights=[0.3, 0.7])[0]
        
        timestamp = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 60))
        
        data.append({
            "user_id": user,
            "matched_user_id": match,
            "action": action,
            "timestamp": timestamp.strftime("%Y-%m-%d")
        })

df = pd.DataFrame(data)
df.to_csv("feedback.csv", index=False)

print("feedback.csv generated successfully!")