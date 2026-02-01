===========================================================
PROJECT: ML-POWERED SERVICE RECOMMENDATION SYSTEM
DEVELOPER: PRAVIN KUMAR AC
===========================================================

1. OVERVIEW
This project is an intelligent recommendation engine that matches 
users with marketplace services based on business type, budget, 
language, and location using Machine Learning (Cosine Similarity).

2. FILE STRUCTURE
Ensure the following files are in the same folder:
- app.py                             (The main application code)
- service_recommendation_data (1).csv (The official dataset)
- README.txt                         (This instruction file)

3. PREREQUISITES
You must have Python installed. Before running the app, install 
the required libraries using the following command:

pip install pandas scikit-learn streamlit

4. HOW TO RUN THE APPLICATION
1. Open your Terminal or PowerShell.
2. Navigate to this project folder using the 'cd' command.
3. Run the application by typing:

   streamlit run app.py

5. CORE ML MODULES INCLUDED
- Feature Encoding: Converts categorical data into numerical vectors.
- Similarity Engine: Uses Cosine Similarity to rank services.
- Match Scoring: Generates a relevance score (0.0 to 1.0).
- Explainable AI: Provides human-readable reasons for each match.

===========================================================