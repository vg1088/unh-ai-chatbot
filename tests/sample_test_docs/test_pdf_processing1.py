import pytest
from chatbot_fat import extract_text_from_pdf, chunk_text

def test_extract_text_from_real_pdf():
    # Path to a real PDF file in the qdrant folder
    real_pdf_path = r"C:\Users\mukth\internship project\Fall2024-Team-Avengers\qdrant\893_edited.pdf"

    # Expected keywords or phrases that should appear in the extracted text
    expected_phrases = [
        "COMP893",
        "internship",
        "credit",
        "hours",
        "Office",
        "hours",
        "COURSE SCHEDULE"
    ]

    # Call the function
    extracted_text = extract_text_from_pdf(real_pdf_path)

    # Normalize the extracted text to remove newlines and extra whitespace
    extracted_text = ' '.join(extracted_text.split())

    # Print extracted text for debugging (remove/comment out after confirming)
    print(extracted_text)

    # Check that each expected phrase is in the extracted text
    for phrase in expected_phrases:
        assert phrase in extracted_text, f"Expected phrase '{phrase}' not found in extracted text."



def test_chunk_text_with_real_pdf():
    # Path to a real PDF file in the qdrant folder
    real_pdf_path = r"C:\Users\mukth\internship project\Fall2024-Team-Avengers\qdrant\893_edited.pdf"
    
    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(real_pdf_path)
    
    # Define chunk size
    chunk_size = 2000  # Adjust based on what your application uses

    # Get chunks
    chunks = chunk_text(extracted_text, chunk_size)
    
    # Check that each chunk except possibly the last is of the specified size
    for chunk in chunks[:-1]:
        assert len(chunk) == chunk_size, "A chunk size does not match the specified chunk_size."
    
    # Ensure the last chunk is not larger than chunk_size
    assert len(chunks[-1]) <= chunk_size, "Last chunk exceeds the specified chunk_size."

    # Check that the reassembled text matches the original extracted text
    reassembled_text = ''.join(chunks)
    assert reassembled_text == extracted_text, "Reassembled text does not match original extracted text."
