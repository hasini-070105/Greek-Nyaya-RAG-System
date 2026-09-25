# Module: PDF Text Extraction & Cleaning
import re
from pathlib import Path
from pypdf import PdfReader

def extract_and_clean():
    input_fol = Path("intern/corpus")
    output_fol = Path("extracted_text")

    for pdf_path in input_fol.rglob("*.pdf"):
        relative_path = pdf_path.relative_to(input_fol)
        txt_path = output_fol / relative_path.with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            reader = PdfReader(str(pdf_path))
            text_parts = []
            
            # Extract text from all pages first
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            # Combine and clean after page iteration finishes
            extracted_text = "\n".join(text_parts).strip()
            extracted_text = re.sub(r'-\s*\n\s*', '', extracted_text)
            extracted_text = re.sub(r'\n{3,}', '\n\n', extracted_text)
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            print(f"Saved: {txt_path}")
            
        except Exception as e:
            print(f"Error processing {pdf_path} -> {e}")

if __name__ == "__main__":
    extract_and_clean()
