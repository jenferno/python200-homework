import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

# --- Step 1: Setup ---
load_dotenv()  # finds the root .env by searching upward from this file

api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "OPENAI_API_KEY not found. Check your .env file."
client = OpenAI(api_key=api_key)
print("API key loaded successfully.")

docs_dir = Path(__file__).parent / "resources" / "groundwork_docs"
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
print(f"Found documents directory: {docs_dir}")

# --- Step 2: Load the documents ---
documents = SimpleDirectoryReader(input_dir=str(docs_dir)).load_data()

print(f"Loaded {len(documents)} documents:")
for doc in documents:
    print(f"  {doc.metadata['file_name']}")

# --- Step 3: Build the index and query engine ---
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = LlamaOpenAI(model="gpt-4o-mini")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

print("Index built successfully. Ready to answer questions.")

# --- Step 4: Query the assistant ---
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    response = query_engine.query(question)
    top_node = response.source_nodes[0]

    print("=" * 70)
    print(f"Question: {question}")
    print(f"Answer: {response}")
    print("Top source:")
    print(f"  Document: {top_node.node.metadata['file_name']}")
    print(f"  Similarity score: {top_node.score:.4f}")
    print(f"  Chunk preview: {top_node.node.get_content()[:200]}")

# Reflection on Step 4:
# The assistant consistently provided confident responses, and the answers I reviewed 
# were accurate and supported by the retrieved documents. The founding story question 
# referenced Maya Torres and Sam Okafor, their founding of the business in 2018, and 
# Asheville, with our_story.txt receiving a high similarity score of 0.77. Similarly, 
# the catering and wholesale response correctly identified the minimum requirements 
# of 20 guests for events and 5 pounds for wholesale orders, using wholesale_catering.txt 
# with a score of 0.63. Both questions retrieved the appropriate source documents.
#
# The loyalty program question was more surprising. Although the response accurately 
# explained the rewards system (one point per dollar, 100 points for a free drink, 
# and registration at the register or website), the top retrieved source was faq.txt, 
# with a relatively low similarity score of 0.30. I suspect this is because the FAQ 
# combines several unrelated topics, including business hours, locations, and loyalty 
# information, which may reduce its overall similarity to a specific query.
#
# This demonstrates that a low similarity score does not necessarily indicate an 
# incorrect answer, just as a high score does not guarantee accuracy. Overall, the 
# assistant performed reliably when the necessary information was available, but 
# reviewing the retrieved source text is essential for evaluating the quality and 
# accuracy of its responses.

# --- Step 5: Find a failure ---
failure_question = "Do you have a drive-through, and can I order ahead on your mobile app?"

response = query_engine.query(failure_question)

print("=" * 70)
print(f"Question: {failure_question}")
print(f"Full response: {response}")
print("\nRetrieved source nodes:")
for i, node in enumerate(response.source_nodes, start=1):
    print(f"  [{i}] Document: {node.node.metadata['file_name']}")
    print(f"      Similarity score: {node.score:.4f}")
    print(f"      Chunk preview: {node.node.get_content()[:200]}")

# Failure analysis:
# What I asked / why it was hard: I asked whether Groundwork offers drive-through 
# service and mobile ordering. None of the five available documents mentioned either 
# service, so the appropriate response was to acknowledge that the information was 
# unavailable.
#
# What went wrong: The retrieval system returned three chunks from faq.txt, 
# wholesale_catering.txt, and seasonal_specials.txt, with relatively low similarity 
# scores of 0.33, 0.30, and 0.27. None of these sources addressed the question 
# directly. This occurred because the retriever always returns the specified 
# number of results through top_k, even when the retrieved content is not relevant. 
# However, the model correctly acknowledged that the documents did not provide the 
# requested information rather than making an unsupported assumption.
#
# Tone: The model used a neutral and cautious tone instead of presenting an 
# uncertain answer as fact. Although this was the appropriate response, the 
# system does not guarantee that the model will always recognize when information 
# is missing. Another query or response could potentially lead to an unsupported 
# assumption based on irrelevant sources. This reinforces the importance of 
# verifying AI-generated answers against their retrieved sources and treating 
# low similarity scores as potential warning signs.
#
# What I would change: I would implement a similarity threshold to exclude 
# low-scoring results and return a standard response when no relevant information 
# is available. I would also strengthen the system prompt to instruct the model 
# to answer only using the retrieved context and acknowledge when information is 
# missing. Finally, I would expand the document collection to include additional 
# topics, such as ordering methods and available customer services.

# --- Step 6: Reflection ---
#
# 1. Framework value:
# LlamaIndex simplifies the development of RAG applications by handling much 
# of the underlying retrieval and indexing process. The core pipeline required 
# approximately five lines of code, including loading documents with 
# SimpleDirectoryReader, configuring the embedding model and LLM through Settings, 
# creating a vector index with VectorStoreIndex.from_documents, and initializing 
# the query engine with as_query_engine.
#
# Without a framework, I would need to implement several steps manually, including 
# text splitting, generating embeddings through API calls, storing vectors, 
# calculating cosine similarity, ranking results, and constructing prompts that 
# ombine retrieved information with the original question. A framework makes it 
# much faster to develop a functional prototype by managing these processes behind 
# a simplified interface. However, the tradeoff is that certain default behaviors, 
# such as chunk sizes, fallback LLM configurations, and retrieving a fixed number 
# of results through top_k, are less visible. Understanding these underlying 
# processes is still important when troubleshooting errors or improving retrieval 
# performance.
#
# 2. Another use case:
# One practical application of RAG would be an internal HR or IT help desk for 
# a company. Employees frequently ask questions about topics such as vacation 
# policies, employee benefits, or setting up a company VPN. Since this information 
# is often stored in lengthy policy documents, internal wikis, and technical 
# guides, a RAG assistant could retrieve relevant information and generate 
# answers based on the most current documentation. It could also identify the 
# source documents used to generate each response, allowing employees to verify 
# the information. This would save HR and IT teams time by reducing repetitive 
# questions while providing employees with quicker access to information.
#
# 3. A failure mode RAG cannot fully prevent:
# One limitation of RAG is that retrieving the correct information does not 
# guarantee that the LLM will interpret or communicate it accurately. For 
# example, the model could incorrectly combine information from two separate 
# document chunks and generate a conclusion that is not supported by either 
# source. Even when the retrieval process works correctly, the generated 
# response can still sound confident and contain inaccurate information. 
# Additionally, if the original documents contain outdated or incorrect 
# information, RAG may retrieve and repeat those errors. This demonstrates 
# why important AI-generated answers should still be verified against their 
# original sources or reviewed by a knowledgeable person.