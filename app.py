import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Load datasets
job_skills_df = pd.read_csv("job_skills_dataset_corrected.csv")
learning_resources_df = pd.read_csv("learning_resources_dataset.csv")

# Preprocess skills
job_skills_df['Skills Required'] = job_skills_df['Skills Required'].str.replace(' ', '')
job_skills_df['Skill_List'] = job_skills_df['Skills Required'].str.split(',')

# Multi-hot encoding
mlb = MultiLabelBinarizer()
skills_encoded = pd.DataFrame(mlb.fit_transform(job_skills_df['Skill_List']), columns=mlb.classes_)

# Combine features
jobs_encoded_df = pd.concat([job_skills_df[['Job Title', 'Job Category', 'Company', 'Salary']], skills_encoded], axis=1)

# Label encode categories
le = LabelEncoder()
jobs_encoded_df['Job Category Encoded'] = le.fit_transform(jobs_encoded_df['Job Category'])

# Scale skill features
skill_features = skills_encoded.columns
scaler = StandardScaler()
skills_scaled = scaler.fit_transform(jobs_encoded_df[skill_features])

# KMeans clustering
kmeans = KMeans(n_clusters=11, random_state=42)
jobs_encoded_df['Cluster'] = kmeans.fit_predict(skills_scaled)

# Fit KNN model
knn_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn_model.fit(skills_scaled)

# Streamlit app
st.title("🔍 Career Skill Match and Recommendations")
st.markdown("Enter your skills and career goals to see your best job matches, missing skills, and learning resources.")

# User input
all_skills = list(skill_features)
user_skills_input = st.multiselect("Select Your Current Skills:", options=all_skills)
career_goal_input = st.selectbox("Select Your Career Goal:", options=sorted(jobs_encoded_df['Job Title'].unique()))

if st.button("🔎 Analyze My Fit"):
    # Encode user skills
    user_skills_encoded = np.zeros(len(skill_features))
    for idx, skill in enumerate(skill_features):
        if skill in user_skills_input:
            user_skills_encoded[idx] = 1

    user_skills_scaled = scaler.transform([user_skills_encoded])

    # Match percentage
    goal_jobs = jobs_encoded_df[jobs_encoded_df['Job Title'].str.contains(career_goal_input, case=False)]
    if len(goal_jobs) > 0:
        goal_skills = goal_jobs[skill_features].mean().values
    else:
        goal_skills = jobs_encoded_df[skill_features].mean().values

    match_percent = (np.sum((user_skills_encoded * goal_skills)) / np.sum(goal_skills)) * 100
    st.subheader(f"🎯 You match {match_percent:.2f}% of the skills required for a {career_goal_input} role.")

    # Missing skills
    missing_skills = [skill_features[idx] for idx, val in enumerate(goal_skills) if val > 0 and user_skills_encoded[idx] == 0]
    st.subheader("⚙️ Skills You Still Need:")
    if missing_skills:
        for skill in missing_skills:
            st.markdown(f"- {skill}")
    else:
        st.markdown("✅ You have all the required skills!")

    # Learning resources
    st.subheader("📚 Recommended Learning Resources:")
    for skill in missing_skills:
        resources = learning_resources_df[learning_resources_df['Skill'] == skill]['Learning Resource'].values
        if len(resources) > 0:
            st.markdown(f"- **{skill}**: {resources[0]}")

    # Closest jobs
    distances, indices = knn_model.kneighbors(user_skills_scaled)
    st.subheader("💼 Closest Matching Jobs:")
    for idx in indices[0]:
        job = jobs_encoded_df.iloc[idx]
        st.markdown(f"- {job['Job Title']} at {job['Company']} (💰 {job['Salary']})")

    # Predict user's cluster and suggest common fields
    user_cluster = kmeans.predict(user_skills_scaled)[0]
    similar_jobs = jobs_encoded_df[jobs_encoded_df['Cluster'] == user_cluster]
    top_fields = similar_jobs['Job Category'].value_counts().head(3).index.tolist()

    st.subheader("📊 Based on Your Skills, You May Fit Into Fields Like:")
    for field in top_fields:
        st.markdown(f"- ✅ {field}")

    top_titles = similar_jobs['Job Title'].value_counts().head(5).index.tolist()
    st.subheader("🧭 Common Job Titles in Your Skill Cluster:")
    for title in top_titles:
        st.markdown(f"- {title}")

    st.info("These suggestions are derived from clustering jobs by their skill requirements.")
