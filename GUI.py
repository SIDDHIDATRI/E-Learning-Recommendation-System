import streamlit as st
from Main import recommend, feedback

st.set_page_config(page_title="E-Learning Recommender", layout="centered")

st.title("E-Learning Content Recommendation System")
st.markdown("### Find courses based on your interest")

# User input
query = st.text_input("Enter your interest (e.g., Machine Learning, Python, AI)")

# Store state
if "results" not in st.session_state:
    st.session_state.results = None

if st.button(" Recommend"):
    if query.strip() == "":
        st.warning("Please enter a query")
    else:
        st.session_state.results = recommend(query)

# Show results
if st.session_state.results is not None:
    st.subheader("Recommended Courses")

    selected = []

    for i, row in st.session_state.results.iterrows():
        if st.checkbox(row['Course Title'], key=i):
            selected.append(i)

    # Save results
    st.session_state.results[['Course Title']].to_csv("recommendations.csv", index=False)

    # Feedback
    if st.button("Improve Recommendations"):
        if len(selected) == 0:
            st.warning("Select at least one course")
        else:
            new_results = feedback(query, selected)

            st.subheader("Improved Recommendations")
            for _, row in new_results.iterrows():
                st.markdown(f"[{row['Course Title']}]({row['Course Url']})")