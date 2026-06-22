# E-Learning Content Recommendation System

## Overview

This project is an NLP-based E-Learning Content Recommendation System that provides personalized course recommendations based on user interests. The system combines Information Retrieval and Machine Learning techniques to improve recommendation accuracy and relevance.

## Features

* Text Preprocessing using NLTK
* TF-IDF Vectorization
* Query Expansion using WordNet
* K-Nearest Neighbors (KNN) Retrieval
* Cosine Similarity Based Matching
* Rocchio Relevance Feedback
* Hybrid Ranking using Similarity, Ratings, and Reviews
* Interactive Streamlit User Interface

## Tech Stack

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* SciPy
* Streamlit

## Project Workflow

1. User enters a course/topic of interest.
2. Query preprocessing is performed.
3. Query expansion adds semantically related terms using WordNet.
4. TF-IDF converts textual data into vector representations.
5. KNN retrieves the most relevant courses.
6. User provides relevance feedback.
7. Rocchio Algorithm refines recommendations.
8. Final ranking combines:

   * Cosine Similarity
   * Course Ratings
   * Course Popularity (Reviews)

## Dataset

The project uses a Coursera course dataset containing:

* Course Title
* Course Description
* Skills
* Ratings
* Number of Reviews
* Course URL

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd E-Learning-Recommendation-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run GUI.py
```

The application will be available at:

```text
http://localhost:8501
```

## Project Structure

```text
E-Learning-Recommendation-System/
│
├── GUI.py
├── Main.py
├── CourseraDataset-Clean.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Key Algorithms Used

* TF-IDF Vectorization
* Cosine Similarity
* Query Expansion (WordNet)
* K-Nearest Neighbors (KNN)
* Rocchio Relevance Feedback

## Future Enhancements

* Transformer-based semantic search
* Collaborative filtering
* User profiling
* Cloud deployment
* Multi-platform course integration

## Authors

Siddhi Datri
MCA (Data Science & Informatics)
National Institute of Technology Patna
