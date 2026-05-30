import os
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings

def clean_text(text: str) -> str:
    """
    Cleans extracted text by normalizing whitespace, newlines, and removing unreadable characters.
    """
    # Replace multiple whitespace characters/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    return text.strip()

def process_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text from a PDF page by page, cleans it, chunks it, and returns
    a list of chunk dictionaries, each containing:
    - 'id': unique string identifier
    - 'text': string chunk text
    - 'metadata': dict with 'source' (filename), 'page' (1-indexed page number), and 'chunk_idx'
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
    reader = PdfReader(pdf_path)
    filename = os.path.basename(pdf_path)
    
    chunks = []
    
    # Initialize recursive text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len
    )
    
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text()
            if not page_text:
                continue
            
            cleaned_text = clean_text(page_text)
            if not cleaned_text or len(cleaned_text) < 10:
                continue
                
            page_chunks = splitter.split_text(cleaned_text)
            
            for chunk_idx, chunk_text in enumerate(page_chunks):
                # We skip extremely short chunks that are likely artifacts of headers/footers
                if len(chunk_text.strip()) < 20:
                    continue
                    
                chunk_id = f"{filename}_p{page_num}_c{chunk_idx}"
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": filename,
                        "page": page_num,
                        "chunk_idx": chunk_idx
                    }
                })
        except Exception as e:
            # Add logging/handling for damaged pages
            print(f"Error parsing page {page_num} of {filename}: {e}")
            
    return chunks
