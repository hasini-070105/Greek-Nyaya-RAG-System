# Module: Paragraph Chunking & Metadata Generation

import json
from pathlib import Path

MAX_CHARS = 1200
MIN_CHARS = 800
OVERLAPS = 120

def split_long_text(text, max_char=MAX_CHARS, overlap=OVERLAPS):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def process_chunks():
    input_fol = Path("extracted_text")
    output_fol = Path("paragraph_chunks")
    output_fol.mkdir(parents=True, exist_ok=True)

    for txt_path in input_fol.rglob("*.txt"):
        rel_path = txt_path.relative_to(input_fol)
        output_path = output_fol / rel_path.with_suffix("")
        output_path.mkdir(parents=True, exist_ok=True)
        
        text = txt_path.read_text(encoding="utf-8").strip()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        chunks = []
        for para in paragraphs:
            if len(para) <= MAX_CHARS:
                chunks.append(para)
            else:
                chunks.extend(split_long_text(para))
        
        merged_chunks = []
        for chunk in chunks:
            if merged_chunks and len(chunk) < MIN_CHARS:
                candidate = merged_chunks[-1] + "\n\n" + chunk
                if len(candidate) <= MAX_CHARS:
                    merged_chunks[-1] = candidate
                else:
                    merged_chunks.append(chunk)
            else:
                merged_chunks.append(chunk)
                
        for i, chunk in enumerate(merged_chunks, 1):
            chunkfile = output_path / f"chunk_{i}.txt"
            metafile = output_path / f"chunk_{i}.json"
            chunkfile.write_text(chunk, encoding="utf-8")
            
            metadata = {
                "source_file": str(rel_path),
                "chunk_id": i,
                "character_count": len(chunk),
                "paragraph_split": "\n\n" in chunk
            }
            metafile.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Saved: {chunkfile} ({len(chunk)} chars)")

if __name__ == "__main__":
    process_chunks()
