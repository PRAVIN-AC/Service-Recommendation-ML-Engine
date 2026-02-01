import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

# --- PHASE 1: DATA LOADING & PREPROCESSING ---
def load_data():
    try:
        # Load the official dataset you uploaded
        df = pd.read_csv("service_recommendation_data (1).csv")
        
        # Clean the data: Remove extra spaces and handle missing values [cite: 142, 143]
        for col in ['Target_Business_Type', 'Price_Category', 'Language_Support', 'Location_Area']:
            df[col] = df[col].astype(str).str.strip()
        
        df = df.fillna("Not Available")
        return df
    except FileNotFoundError:
        st.error("Error: 'service_recommendation_data (1).csv' not found. Please place it in the same folder as this script.")
        return pd.DataFrame()

# --- PHASE 2: RECOMMENDATION ENGINE ---
def get_recommendations(user_input, df):
    # Features required for matching as per project brief [cite: 116, 121]
    features = ['Target_Business_Type', 'Price_Category', 'Language_Support', 'Location_Area']
    
    # 1. Feature Encoding (Converting text to numbers) [cite: 121]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(df[features])
    
    # 2. Convert User Input into a matching vector
    user_df = pd.DataFrame([user_input])
    user_vector = encoder.transform(user_df[features])
    
    # 3. Calculate Similarity Scores using Cosine Similarity [cite: 123, 126]
    scores = cosine_similarity(user_vector, encoded_features).flatten()
    
    # 4. Add scores to the dataframe
    df['Match_Score'] = scores
    
    # Generate Match Quality tag (High/Medium) [cite: 127]
    df['Match_Quality'] = df['Match_Score'].apply(lambda x: 'High' if x > 0.7 else 'Medium')
    
    # Return Top 3 results [cite: 124]
    return df.sort_values(by='Match_Score', ascending=False).head(3)

# --- PHASE 3: EXPLANATION GENERATOR ---
def generate_explanation(user_input, row):
    # Creates human-readable reasons for the recommendation [cite: 128, 129]
    reasons = []
    if user_input['Target_Business_Type'] == row['Target_Business_Type']:
        reasons.append(f"it specifically supports {row['Target_Business_Type']} businesses")
    if user_input['Price_Category'] == row['Price_Category']:
        reasons.append(f"it fits your {row['Price_Category']} budget")
    if user_input['Location_Area'] == row['Location_Area']:
        reasons.append(f"it is available in {row['Location_Area']}")
    
    if reasons:
        return "Recommended because " + " and ".join(reasons) + "."
    return "This service matches your general preferences and language requirements."

# --- PHASE 4: STREAMLIT UI ---
def main():
    st.set_page_config(page_title="UNLOX Recommender", layout="wide")
    st.title("ML-Powered Service Recommendation System")
    st.markdown("### Matching Marketplace Services with Business Needs")
    
    df = load_data()
    if df.empty:
        return

    # Sidebar for User Input [cite: 144, 145]
    st.sidebar.header("Filter Preferences")
    biz_type = st.sidebar.selectbox("Your Business Type", sorted(df['Target_Business_Type'].unique()))
    price = st.sidebar.selectbox("Budget Category", sorted(df['Price_Category'].unique()))
    lang = st.sidebar.selectbox("Language Preference", sorted(df['Language_Support'].unique()))
    loc = st.sidebar.selectbox("Location/Area", sorted(df['Location_Area'].unique()))
    
    user_input = {
        'Target_Business_Type': biz_type,
        'Price_Category': price,
        'Language_Support': lang,
        'Location_Area': loc
    }
    
    if st.sidebar.button("Get Best Matches"):
        results = get_recommendations(user_input, df)
        
        st.write("---")
        st.subheader("Top 3 Recommended Services for You")
        
        # Display results in columns
        cols = st.columns(3)
        for i, (idx, row) in enumerate(results.iterrows()):
            with cols[i]:
                st.info(f"**{row['Service_Name']}**")
                st.write(f"**Match Quality:** {row['Match_Quality']}")
                st.progress(float(row['Match_Score']))
                st.write(f"**Score:** {row['Match_Score']:.2f}")
                st.write(f"**Description:** {row['Description']}")
                st.success(generate_explanation(user_input, row))

if __name__ == "__main__":
    main()