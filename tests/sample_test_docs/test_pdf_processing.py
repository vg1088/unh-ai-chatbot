import pytest
from chatbot_fat import extract_text_from_pdf, chunk_text

def test_extract_text_from_pdf():
    # Path to a sample PDF file with known text
    sample_pdf_path = r"C:\Users\mukth\internship project\Fall2024-Team-Avengers\tests\sample_pdf.pdf"

    # Update expected output text to match the actual content of sample_pdf.pdf
    expected_text = "This is a sample PDF. It contains some text for testing purposes. Each line is predictable and easy to verify."

    # Call the function
    extracted_text = extract_text_from_pdf(sample_pdf_path)

    # Normalize the extracted text to remove newlines and extra whitespace
    extracted_text = ' '.join(extracted_text.split())

    # Assert the extracted text matches expected text
    assert extracted_text == expected_text, "Text extracted does not match expected text."




def test_chunk_text():
    # Sample long text to chunk
    sample_text = "This is a long text that we want to split into chunks." * 100
    
    # Define chunk size
    chunk_size = 50
    
    # Get chunks
    chunks = chunk_text(sample_text, chunk_size)
    
    # Check all chunks except the last one are of the specified chunk_size
    for chunk in chunks[:-1]:
        assert len(chunk) == chunk_size, "Chunk size does not match expected chunk size."

    # Verify the last chunk is not larger than chunk_size
    assert len(chunks[-1]) <= chunk_size, "Last chunk is larger than the chunk size."

    # Ensure that reassembling chunks gives the original text
    reassembled_text = ''.join(chunks)
    assert reassembled_text == sample_text, "Reassembled text does not match original text."

