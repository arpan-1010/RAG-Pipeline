import fitz
dataset = "/kaggle/input/datasets/amondal03/pdf-bot/attention-is-all-you-need.pdf"
doc = fitz.open(dataset)
print("Total Pages : ", len(doc))

# extract texts from one page
page = doc[0]
text = page.get_text()
print(text[:1000])

print(text[:5000])

# extract texts from all pages
all_text = ""
for page in doc :
    all_text += page.get_text()

print("Total no of texts : ",len(all_text))


# fixed size chunking
def fixed_chunking(text, chunk_size):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i: i + chunk_size]
        chunks.append(chunk)

    return chunks

chunks = fixed_chunking(all_text, 500)

print("Total Chunks:", len(chunks))

print("\nFirst Chunk : \n")
print(chunks[0])


# sentence chunking
def sentence_chunking(text, max_sentences=2):
    sentences = text.split(".")
    chunks = []

    for i in range(0, len(sentences), max_sentences):
        chunk = ".".join(sentences[i: i + max_sentences])
        chunks.append(chunk)

    return chunks

sentence_chunks = sentence_chunking(all_text, max_sentences = 2)
print("Total Sentence Chunks : ", len(sentence_chunks))

print("\nFirst Sentence Chunk : \n")
print(sentence_chunks[0])

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-V2")

# random text embedding
text = "Transformers uses self attention"
embedding = model.encode(text)

print(type(embedding))
print(embedding.shape)

print(embedding[:10])

chunk_embeddings = model.encode(sentence_chunks)
print(chunk_embeddings.shape)

print(chunk_embeddings[0 : 5].shape)

chunk = sentence_chunks[0]
print(chunk)

# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

query = "What is attention"

query_embedding  = model.encode(query)
chunk_embedding = model.encode(chunk)

score = cosine_similarity([query_embedding], [chunk_embedding])

print(score)

# compare query with all chunks
scores = []

for chunk_embedding in chunk_embeddings:
    score = cosine_similarity([query_embedding], [chunk_embedding])[0][0]

    scores.append(score)

print(len(scores))

# best chunk
best_idx = scores.index(max(scores))

print("Best Score : ", max(scores))

print("\nBest Chunk : \n")
print(sentence_chunks[best_idx])

# build vector index with FAISS
import faiss
print(faiss.__version__)

# build FAISS index
import numpy as np
embeddings = np.array(chunk_embeddings, dtype = np.float32)
print(embeddings.shape)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)

index.add(embeddings)
print(index.ntotal)

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding, dtype = np.float32)

print(query_embedding.shape)

# retrieve top k results
distances, indices = index.search(query_embedding, k = 5)

print(indices)
print(distances)

# retrieve the actual chunks
for idx in indices[0] :
    print("\n")
    print(sentence_chunks[idx])

# adding-metadata
doc = fitz.open("/kaggle/input/datasets/amondal03/pdf-bot/attention-is-all-you-need.pdf")
pages = []
for page_num, page in enumerate(doc) :
    pages.append({
        "page" : page_num + 1,
        "text" : page.get_text()
    })

all_chunks = []
metadata = []

for page in pages :
    chunks = sentence_chunking(page["text"])

    for chunk in chunks :
        all_chunks.append(chunk)
        metadata.append({
            "page" : page["page"],
            "Source" : "attention-is -all-you-need.pdf"
        })

print(all_chunks[0])
print(metadata[0])

for idx in indices[0] :
    print(all_chunks[idx])
    print(metadata[idx])

# metadata filtering
page_filter = 3
for idx in indices[0] :
    if(metadata[idx]["page"]) != page_filter :
        continue
    print(all_chunks[idx])
    print(metadata[idx])

retrieved_chunks = []
for idx in indices[0] :
    if metadata[idx]["page"] != page_filter :
        continue
    retrieved_chunks.append(all_chunks[idx])

print(len(retrieved_chunks))

# build context
context = "\n\n".join(retrieved_chunks)

prompt = f"""
Answer the question using only the provided context.
Context : {context}
Question : {query}
"""

from transformers import pipeline
llm = pipeline(
    "text-generation",
    model = "microsoft/Phi-3-mini-4k-instruct"
)

response = llm(prompt, max_new_tokens = 200)

print("\n",response[0]["generated_text"])