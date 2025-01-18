import streamlit as st
from conference_editor_agent import get_conference
from aggregate import process_paper
from utils.score import get_score, get_final_conference, get_review

# Function to evaluate a paper
def evaluate_paper(pdf_path):
    evaluation_1 = process_paper(pdf_path)
    evaluation_2 = get_conference(pdf_path)
    return evaluation_1, evaluation_2

# Streamlit app setup
st.set_page_config(
    page_title="Research Paper Evaluation Tool",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 WELCOME TO CRISP.AI")
st.write(
    """
    This tool assists in the evaluation of research papers for publishability 
    and recommends the most suitable conferences based on their content.
    """
)

# Sidebar for additional information
with st.sidebar:
    st.header("About the Tool")
    st.write(
        """
        - **Task 1**: Classify papers as "Publishable" or "Non-Publishable."
        - **Task 2**: Recommend the best conference for publishable papers.
        - **Framework**: Leverages advanced AI and Pathway technologies for real-time processing.
        """
    )
    st.markdown(
        "[📚 Learn more about Pathway](https://pathway.com/developers/user-guide/introduction/welcome/)"
    )

# File upload section
st.header("📤 Upload Your PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    # Save the uploaded file temporarily
    pdf_path = "uploaded_file.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info("✅ File uploaded successfully! Processing...")
    
    # Processing the file
    try:
        evaluation_1, evaluation_2 = evaluate_paper(pdf_path)
        report = evaluation_1
        st.success("🎉 Evaluation completed!")
        score = get_score(evaluation_1)
        evaluation_1 = True if score > 6.5 else False
        
        # Display the results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Publishability Assessment")
            Result = "Publishable" if evaluation_1 else "Non-Publishable"
            st.write(f"**Result:** \n{Result}")
            if Result=="Non-Publishable":
                st.write(f"**Review:** {report}")
        
        with col2:
            st.subheader("Conference Recommendation")
            if (Result=="Non-Publishable"):
                st.write(f"**Recommended Conference:** Not Applicable")
            else:    
                confernce = get_final_conference(evaluation_2)
                review = get_review(evaluation_2)
                st.write(f"**Recommended Conference:** {confernce}")
                st.write(f"**Review:** {review}")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    Developed for the **Kharagpur Data Science Hackathon 2025**.  
    Powered by [Pathway](https://pathway.com/) and Streamlit.
    """
)
