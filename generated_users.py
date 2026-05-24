import pandas as pd
import random

# Data to be inserted
first_names = ["Rahul","Priya","Arjun","Neha","Karan","Sneha","Amit","Riya","Vikram","Anjali",
               "John","Emily","Liam","Olivia","Noah","Sophia","Ethan","Ava","James","Mia"]

last_names = ["Sharma","Mehta","Singh","Verma","Patel","Nair","Joshi","Kapoor","Rao","Das",
              "Smith","Brown","Wilson","Taylor","Anderson","Thomas","Jackson","White","Harris","Clark"]

locations = [
    "Delhi","Mumbai","Bangalore","Chennai","Hyderabad","Pune","Kolkata",
    "New York","London","Berlin","Paris","Toronto","Sydney","Dubai","Singapore",
    "Tokyo","Seoul","Amsterdam","Dublin","Barcelona","Rome"
]

professions_mbti = [
    ("Machine Learning Engineer","INTJ"),
    ("Data Scientist","INTP"),
    ("Software Developer","ISTJ"),
    ("Backend Engineer","ISTP"),
    ("Frontend Developer","ISFP"),
    ("Product Manager","ENTJ"),
    ("Project Manager","ESTJ"),
    ("UI/UX Designer","ENFP"),
    ("Marketing Specialist","ESFP"),
    ("Business Analyst","INFJ")
]

interests_pool = [
    "AI, Coding, Reading",
    "Travel, Photography, Blogging",
    "Fitness, Sports, Wellness",
    "Gaming, Tech, Movies",
    "Art, Design, Creativity",
    "Music, Writing, Culture"
]

# Templates of individuals
def generate_summary(role, exp):
    return f"{role} with {exp} years of experience working on real-world projects. Skilled in relevant tools and focused on solving practical problems. Passionate about continuous learning and growth."

def generate_about(role):
    return f"I am a {role.lower()} who enjoys improving skills through consistent practice. I prefer structured work environments and value discipline, curiosity, and continuous self-development."

# Generation of the dataset
data = []

for i in range(89):
    fname = random.choice(first_names)
    lname = random.choice(last_names)
    role, mbti = random.choice(professions_mbti)
    exp = random.randint(1,5)
    
    user = {
        "user_id": f"U{str(i+1).zfill(3)}",
        "name": f"{fname} {lname}",
        "age": random.randint(21,35),
        "location": random.choice(locations),
        "profession": role,
        "experience_years": exp,
        "professional_summary": generate_summary(role, exp),
        "about_me": generate_about(role),
        "mbti": mbti,
        "interests": random.choice(interests_pool)
    }
    
    data.append(user)

df = pd.DataFrame(data)
df.to_csv("users.csv", index=False)

print("users.csv generated successfully!")