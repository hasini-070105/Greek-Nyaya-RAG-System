# Module: Prompt Formatting & LLM Answer Generation

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from retriever import QdrantRetriever

def generate_rag_response():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize Retriever
    retriever = QdrantRetriever()

    # 2. Initialize LLM
    LLM_NAME = "Qwen/Qwen2.5-3B-Instruct"
    print("\nLoading LLM...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    # 3. User Query
    query = input("\nEnter your question: ")

    # 4. Fetch Top Chunks
    results = retriever.retrieve(query, top_k=6)
    print(f"\nRetrieved {len(results)} chunks from Qdrant.")

    # 5. Build Context
    context_parts = []
    for i, point in enumerate(results, start=1):
        source = point.payload.get("source_file", "Unknown source")
        chunk_id = point.payload.get("chunk_id", "Unknown")
        text = point.payload.get("text", "")
        context_parts.append(
            f"--- Retrieved Chunk {i} ---\n"
            f"Source: {source}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Similarity Score: {point.score:.4f}\n\n"
            f"{text}\n"
        )
    context = "\n".join(context_parts)

    # 6. Format Prompt
    prompt = f"""
You are a research assistant working with a specialized corpus of books and articles about logic and philosophy.

Answer the user's question using the retrieved context provided below.

Instructions:
1. Use the retrieved context as the primary source.
2. Do not invent information that is not supported by the context.
3. If the context does not contain enough information, say that the answer is not available in the retrieved context.
4. Give a clear and concise explanation.
5. Mention the source when relevant.

Retrieved Context:
{context}

User Question:
{query}

Answer:
"""

    messages = [
        {"role": "system", "content": "You are a research assistant. Answer questions using the retrieved context. Do not hallucinate information."},
        {"role": "user", "content": prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(llm.device)

    # 7. Generate Answer
    print("\nGenerating answer...")
    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens=500, temperature=0.2, do_sample=True)

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\n" + "="*80 + "\nFINAL RAG ANSWER\n" + "="*80)
    print(answer)
    print("="*80)

    print("\nSOURCES USED:")
    print("-" * 80)
    for i, point in enumerate(results, start=1):
        print(f"{i}. {point.payload.get('source_file', 'Unknown')} (Chunk {point.payload.get('chunk_id', 'Unknown')}) [Score: {point.score:.4f}]")

    retriever.close()

if __name__ == "__main__":
    generate_rag_response()
