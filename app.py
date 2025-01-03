import streamlit as st
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Load data from the file (replace 'your_file.json' with your actual file name)
def load_course_data():
    with open('free_courses_vidhya.json', 'r') as file:
        return json.load(file)

# Load the course data
courses_data = load_course_data()

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to get embeddings of course names
def get_embeddings(courses):
    course_names = [course['course_name'] for course in courses]
    course_embeddings = model.encode(course_names)
    return course_embeddings

# Function to search courses based on query
def search_courses(query, top_k=3):
    # Generate embedding for the query
    query_embedding = model.encode([query])[0]
    
    # Get course embeddings
    course_embeddings = get_embeddings(courses_data)
    
    # Calculate similarity scores (cosine similarity)
    similarity_scores = np.dot(course_embeddings, query_embedding) / (np.linalg.norm(course_embeddings, axis=1) * np.linalg.norm(query_embedding))
    
    # Get the top_k most similar courses
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        course = courses_data[idx]
        results.append({
            'Course Name': course['course_name'],
            'Lessons': course['lessons'],
            'Price': course['price'],
            'Link': course['link']
        })
    return results

# Streamlit UI
st.title('Smart Search for Free Courses on Analytics Vidhya')
st.write('Search for relevant courses using keywords.')

# Text input for query
query = st.text_input('Enter a keyword or query')

# If the user submits a query
if query:
    search_results = search_courses(query)
    
    if search_results:
        st.write(f"Found {len(search_results)} courses:")
        for result in search_results:
            st.write(f"**Course Name:** {result['Course Name']}")
            st.write(f"**Lessons:** {result['Lessons']}")
            st.write(f"**Price:** {result['Price']}")
            st.write(f"[Course Link]({result['Link']})")
            st.write('---')
    else:
        st.write("No results found.")

