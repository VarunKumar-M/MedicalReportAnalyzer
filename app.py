import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF text extraction
import pytesseract
from PIL import Image
import pydicom  # For DICOM (X-ray) files

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Check if API key is set
if not api_key:
    st.error("⚠ API Key not found! Please check your .env file.")
    st.stop()

# Configure Gemini AI
genai.configure(api_key=api_key)

# Streamlit UI - Title
st.title("🩺 AI-Powered Medical Report & X-ray Analyzer")
st.write("Upload multiple **Medical Reports (PDF/TXT/Image) or X-rays (JPG/PNG/DICOM)** to get AI-based analysis.")

# File Upload Section (Multiple Files)
uploaded_files = st.file_uploader("📄 Upload your **Medical Reports / X-rays / Images**",
                                  type=["pdf", "txt", "png", "jpg", "jpeg", "dcm"],
                                  accept_multiple_files=True)

# Initialize text storage for uploaded files
extracted_text = ""

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)
    return text if text.strip() else "⚠ No readable text found in the PDF."

def extract_text_from_image(image_file):
    """Extract text from an uploaded image using OCR."""
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text.strip() if text.strip() else "⚠ No readable text found in the image."

def load_dicom_image(dicom_file):
    """Convert DICOM X-ray image to a displayable format."""
    dicom_data = pydicom.dcmread(dicom_file)
    image = dicom_data.pixel_array  # Extract pixel array
    return Image.fromarray(image)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")

        if uploaded_file.type == "application/pdf":
            # Extract text from PDF
            report_text = extract_text_from_pdf(uploaded_file)
            extracted_text += f"\n\n{report_text}"  # Accumulate extracted text
            st.subheader(f"📜 Extracted Report Text from {uploaded_file.name}:")
            st.text_area("", report_text, height=200)

            # AI Analysis for Medical Reports
            st.subheader("🧠 AI Analysis & Suggestions:")
            prompt = f"""
            You are a medical AI expert. Analyze the following medical report and provide:
            1. Key findings and possible conditions.
            2. Health and lifestyle suggestions.
            3. Whether a doctor consultation is required.

            Medical Report:
            {report_text}
            """

            try:
                model = genai.GenerativeModel(model_name="gemini-1.5-flash")
                response = model.generate_content(prompt)
                st.write(response.text)  # Show AI-generated response
            except Exception as e:
                st.error(f"Error in AI Analysis: {e}")

        elif uploaded_file.type.startswith("image/") or uploaded_file.type == "application/dicom":
            # Handle X-ray / Medical Image Analysis
            st.subheader(f"🩻 {uploaded_file.name} - X-ray / Medical Image Preview:")
            if uploaded_file.type == "application/dicom":
                # Load and display DICOM X-ray image
                image = load_dicom_image(uploaded_file)
            else:
                # Load and display general image (JPG, PNG)
                image = Image.open(uploaded_file)

            st.image(image, caption=f"Uploaded {uploaded_file.name}", use_container_width=True)

            # AI Analysis for X-rays or general medical images
            st.subheader("🧠 AI Analysis for X-ray / Medical Image:")
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")

            try:
                # Use the image to analyze for medical conditions or abnormalities
                response = model.generate_content(["Analyze this medical image for any abnormalities.", image])
                st.write(response.text)  # Show AI-generated response
            except Exception as e:
                st.error(f"Error in AI Analysis: {e}")

        else:
            st.error(f"⚠ Unsupported file type: {uploaded_file.name}. Please upload a PDF, TXT, PNG, JPG, or DICOM file.")

# -------------------- AI Chatbot for Medical Queries --------------------

st.sidebar.title("💬 Ask the AI Medical Assistant")

# Provide context from uploaded files to the chatbot
context_text = f"Context from uploaded files:\n{extracted_text}\n\nNow, feel free to ask any questions related to the uploaded medical report or images."

user_question = st.sidebar.text_input("Type your medical question here:")

if user_question:
    try:
        # Append the context and user question to the prompt for a more relevant AI response
        full_prompt = f"{context_text}\nUser Question: {user_question}\nAnswer the user's question based on the context."

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(full_prompt)
        st.sidebar.subheader("🧠 AI Response:")
        st.sidebar.write(response.text)

    except Exception as e:
        # Default message when no context is found
        if "which hand" in user_question.lower():
            st.sidebar.write("⚠ Please upload a relevant medical report or image containing details of the hand you are asking about.")
        else:
            st.sidebar.error(f"Error in AI Response: {e}")
