# UlNN-Project-Prateek-Raj-
Skill-Based Job Recommendation System 💼🚀
A lightweight, interpretable, and modular machine learning system that recommends jobs based on your current skillset, highlights your skill gaps, and suggests a personalized learning path to bridge them.

🔍 Overview
This project helps users:

Get job recommendations based on their current technical skills.
Understand how close they are to their desired job role.
Identify missing skills (Skill Gap Analysis).
Receive curated course recommendations to upskill.
Track career progress dynamically.
It is implemented as a web application using Streamlit and scikit-learn, using structured .csv datasets.


🧠 How It Works

📌 Step-by-Step Workflow:

User inputs their skills using checkboxes.
System vectorizes skills using MultiLabelBinarizer.
KMeans clustering groups similar job roles.
KNN recommends jobs that best match user’s profile.
Missing skills are extracted via vector difference.
Mapped learning resources are suggested for each gap.
Everything is shown instantly via the Streamlit frontend.
