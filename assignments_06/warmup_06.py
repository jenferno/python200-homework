from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- RAG Concepts ---
# Concepts Q1
# Scenario A: RAG
#   RAG is the best approach because the policy library contains a large amount 
#   of information that changes every quarter. Retrieving relevant passages when a 
#   question is asked allows the model to reference the most up-to-date information 
#   without needing to be retrained. Updating the retrieval index is also more 
#   efficient and cost-effective than fine-tuning the model every time new policy 
#   documents are added.
#
# Scenario B: Fine-tuning
#   Fine-tuning is the best approach because the company wants the model to consistently 
#   follow a specific brand voice that may not be well represented in its original training 
#   data. Having 3,000 examples provides sufficient material to train the model on the 
#   desired writing style. This can help the model produce more consistent responses 
#   ithout relying entirely on detailed prompts for every interaction.
#
# Scenario C: Prompt engineering
#   Prompt engineering is the best approach because the report is only two pages long and 
#   can easily fit into the prompt as context. The model can reference the document directly 
#   when generating its response. Building a separate retrieval system or fine-tuning a 
#   model would require unnecessary time and resources for a task involving only one document.

# Concepts Q2
#   A confidently incorrect answer can be more harmful than an answer expressing uncertainty 
#   because it may cause people to trust and act on information that is inaccurate. When an 
#   AI model acknowledges that it is unsure, it encourages the user to verify the information 
#   before relying on it. However, a confident but incorrect response may cause users to 
#   accept the information without questioning its accuracy, potentially leading to poor 
#   decisions.
#
#   Tone matters because people often associate fluent, professional, and confident language 
#   with credibility and expertise. When an AI generates incorrect information, it may present 
#   that information in the same polished and authoritative manner as a factual response. 
#   Without clear indicators of uncertainty, users may have difficulty distinguishing between 
#   accurate information and fabricated content. As a result, the tone of an AI-generated 
#   response can influence how much trust users place in its accuracy.
#
#   Example: An employee asks an AI assistant for help troubleshooting a computer that is 
#   running slowly. The AI confidently recommends a solution that involves removing important 
#   system files, incorrectly claiming that they are unnecessary. If the employee trusts the 
#   response and follows the instructions without verifying them, they could damage the 
#   operating system or lose important data. This demonstrates how a confidently incorrect 
#   answer can cause real-world consequences when users mistake an authoritative tone for 
#   reliable technical knowledge.

# Concepts Q3
# Original (out of order) list:
# steps = [
#     "Generate a response from the LLM",
#     "Extract text from source documents",
#     "Receive the user's query",
#     "Retrieve the most relevant chunks",
#     "Convert text chunks into embeddings",
#     "Inject retrieved chunks into the prompt",
#     "Split text into chunks",
#     "Embed the user's query",
# ]
#
# Correct order:
# 1. Extract text from source documents
#    Raw text is pulled out of the source files (PDFs, docs, etc.) so it can be processed.
# 2. Split text into chunks
#    The extracted text is divided into smaller pieces so each one fits in the model's
#    context window and covers a focused idea.
# 3. Convert text chunks into embeddings
#    Each chunk is turned into a numeric vector that captures its meaning, and the
#    vectors are stored in an index.
# 4. Receive the user's query
#    The user asks a question, which starts the retrieval part of the pipeline.
# 5. Embed the user's query
#    The question is converted into a vector using the same embedding model as the chunks.
# 6. Retrieve the most relevant chunks
#    The query vector is compared to the chunk vectors, and the closest matches are returned.
# 7. Inject retrieved chunks into the prompt
#    The retrieved text is added to the prompt as context alongside the user's question.
# 8. Generate a response from the LLM
#    The LLM writes an answer grounded in the supplied context.

# --- Keyword RAG ---
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# Keyword Q1
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected document: {result[0][0]}")

# Keyword Q1 explanation:
#   The function selected loyalty.txt, even though hours.txt was the correct document.
#   The filtered query tokens were ['hours', 'weekends', 'what', 'your'].
#   hours.txt matched only "weekends", while hiring.txt and loyalty.txt each
#   matched only "your", resulting in a three-way tie with a score of 1.
#   The word "hours" was not found in hours.txt because the document uses
#   "open" and "close" instead. Additionally, "your" was not included in the
#   stopword list, so it was counted as a matching keyword.
#   Since scores.sort(reverse=True) sorts tied scores by document name in
#   reverse alphabetical order, loyalty.txt was selected first.
#   Therefore, hours.txt was not selected because of the alphabetical tie-breaker,
#   rather than because another document had a higher relevance score.

# Keyword Q2
query = "Do you have anything without caffeine?"

result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected document: {result[0][0]}")

# Keyword Q2 explanation:
#   The function did not select a document and returned "None found". The 
#   filtered query tokens were ['anything', 'caffeine', 'do', 'have', 'without']. 
#   None of these words appeared in any of the four documents, resulting in an 
#   overlap score of zero for every document.
#
#   This demonstrates a limitation of keyword-based retrieval. Although menu.txt
#   is the most relevant document because it includes options such as oat milk, 
#   almond milk, and other drinks suitable for customers avoiding caffeine, it 
#   does not contain the exact words used in the query. Keyword matching relies 
#   on shared terms and cannot recognize that phrases such as "without caffeine" 
#   may relate to decaffeinated or non-coffee beverage options. Semantic retrieval 
#   using vector embeddings would likely perform better because it identifies 
#   similarities in meaning rather than relying solely on exact keyword matches.

# Keyword Q3 prediction (written before running the code):
#   I predict that the function will return "None found" because the filtered 
#   query tokens are expected to be ['do', 'how', 'i', 'rewards', 'sign', 'up'], 
#   with "for" removed as a stopword. The most relevant document is loyalty.txt, 
#   which discusses the loyalty program and earning points. However, it does not 
#   contain the exact terms "rewards" or "sign up." Although "how," "do," "i," 
#   and "up" are not included in the stopword list, I do not expect these words 
#   to appear in any of the documents either. As a result, I predict that none 
#   of the documents will receive a matching keyword score.

# Keyword Q3
query = "How do I sign up for rewards?"

result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected document: {result[0][0]}")

# Keyword Q3 result:
#   My prediction was correct: the function returned "None found" because 
#   every document received an overlap score of zero. The most relevant 
#   document is loyalty.txt, which includes phrases such as "Join our loyalty 
#   program" and references to earning "points." However, the query uses 
#   different terms, such as "sign up" and "rewards." Although these phrases 
#   have similar meanings, keyword-based retrieval cannot recognize their 
#   semantic relationship without shared words. This demonstrates the same 
#   limitation identified in Q2: keyword-based RAG relies on exact keyword 
#   matches and may fail to retrieve relevant documents when users phrase 
#   their questions differently from the source material.

# --- Semantic RAG Concepts ---
# Semantic Q1
# 1. A vector embedding is a list of numbers that represents the meaning of a
#    piece of text, so that texts with similar meanings end up with similar
#    lists of numbers (close together in vector space).
#
# 2. The chunk with the 0.85 score is more relevant. Cosine similarity measures
#    how closely two vectors point in the same direction, so 0.85 means the
#    chunk and the query are very close in meaning, while 0.30 means they are
#    only loosely related.
#
# 3. Embeddings capture meaning rather than exact spelling, so a chunk that says
#    "join our loyalty program" lands near a query about "signing up for
#    rewards" even though they share no words. The model learned during
#    training that those phrases are used in similar contexts.

# Semantic Q2
# | Feature                    | Keyword RAG                       | Semantic RAG                                       |
# |----------------------------|-----------------------------------|----------------------------------------------------|
# | What is compared?          | Exact word overlap                | Meaning, by comparing query and chunk embeddings   |
# | What is retrieved?         | Full document                     | The top-k most similar chunks (passages)           |
# | Can it handle synonyms?    | No                                | Yes, similar meanings get similar embeddings       |
# | Storage format             | Plain text dictionary             | Vector index (embeddings stored with chunk text)   |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity between the two vectors          |

# --- LlamaIndex ---
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pathlib import Path


Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

# Adjust this path to match where brightleaf_pdfs/ lives on your computer.
PDF_DIR = Path.home() / "python-200-v1" / "lessons" / "06_AI_augmentation" / "resources" / "brightleaf_pdfs"
assert PDF_DIR.exists(), f"PDF directory not found: {PDF_DIR}"

pdf_documents = SimpleDirectoryReader(PDF_DIR).load_data()
index = VectorStoreIndex.from_documents(pdf_documents)


def show_result(question, response):
    """Print the question, the answer, and each retrieved source node."""
    print(f"\nQuestion: {question}")
    print(f"Answer: {response}")
    for i, node in enumerate(response.source_nodes, start=1):
        print(f"  Source {i} | score={node.score:.4f} | {node.node.get_content()[:150]!r}")


# LlamaIndex Q1
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

query_engine = index.as_query_engine(similarity_top_k=3)
for question in questions:
    response = query_engine.query(question)
    show_result(question, response)

# LlamaIndex Q1 observations:
# Query 1 (employee benefits): Only the first retrieved chunk was relevant to 
#   the question. Source 1, with a similarity score of 0.76, contained the 
#   employee benefits policy. Sources 2 and 3, with scores of 0.59 and 0.56, 
#   discussed the EcoVolt partnership and general company information, 
#   respectively, and were unrelated to employee benefits.
#
#   The generated response was confident and specific, identifying benefits 
#   such as Blue Cross health insurance, a 401(k) matching contribution of up 
#   to 5%, twelve weeks of paid parental leave, and a learning stipend. All 
#   of these details were supported by Source 1. The two unrelated sources 
#   were retrieved because the top_k=3 setting required the system to return 
#   three results, regardless of their relevance. However, the model 
#   successfully focused on the relevant information and disregarded the 
#   unrelated content.
# Query 2 (security policies): Source 1, with a similarity score of 0.67, 
#   contained the Network and Data Security information and was the most relevant 
#   document for this query. Sources 2 and 3, with scores of 0.62 and 0.59, 
#   covered the employee benefits introduction and the EcoVolt partnership, 
#   respectively, and were not relevant to the question.
#
#   The generated response was again confident and detailed, covering multi-factor 
#   authentication (MFA), TLS 1.3 and AES-256 encryption, NIST 800-61 incident 
#   response guidelines, and ISO 27001 standards. These details were supported by 
#   the relevant security source. One unexpected result was that the employee 
#   benefits chunk received a higher similarity score than other potentially 
#   relevant content for a security-related question. This demonstrates that 
#   similarity scores indicate how closely text matches a query but do not 
#   necessarily guarantee that the retrieved information is relevant. Despite 
#   retrieving unrelated content, the system generated an accurate response 
#   because the primary security source contained most of the information needed 
#   to answer the question.

# LlamaIndex Q2
q2 = questions[0]
for k in (1, 5):
    print(f"\n=== similarity_top_k={k} ===")
    engine_k = index.as_query_engine(similarity_top_k=k)
    show_result(q2, engine_k.query(q2))

# LlamaIndex Q2 observations:
# When using top_k=1, the model retrieved only the employee benefits chunk, which 
# had a similarity score of 0.76. Even with just one source, the model generated 
# a complete response covering health insurance, financial benefits, parental 
# leave, workplace flexibility, and career development. However, the response 
# included fewer specific details.
#
# When increasing the setting to top_k=5, the response contained additional 
# details, including a $600 wellness reimbursement, employer-paid disability 
# insurance, and quarterly financial seminars. However, Sources 2 through 5, 
# with similarity scores ranging from 0.59 to 0.44, covered the EcoVolt 
# partnership, company overview, financial report, and HelioPanel X5 product 
# information. None of these sources were directly related to employee benefits. 
# Therefore, the additional details in the response came from the original 
# benefits chunk rather than the newly retrieved sources.
#
# This comparison demonstrates that retrieving more information does not always 
# improve the quality of an AI-generated response. In this case, increasing top_k 
# added irrelevant context, increased the prompt size and processing cost, and 
# introduced the possibility of distracting the model from the relevant information. 
# A higher top_k value is more beneficial when the information needed to answer 
# a question is distributed across multiple documents rather than concentrated in 
# a single relevant source.

# LlamaIndex Q3
# Edit this query if you want to try a different kind of hard question.
q3 = "How does BrightLeaf's remote work policy compare to its travel expense rules?"

print("\n=== LlamaIndex Q3 ===")
response_q3 = query_engine.query(q3)
show_result(q3, response_q3)
print("\nAll retrieved chunks:")
for i, node in enumerate(response_q3.source_nodes, start=1):
    print(f"\n--- Chunk {i} (score={node.score:.4f}) ---")
    print(node.node.get_content())

# LlamaIndex Q3 observations:
# What I expected: I expected this query to be more challenging because it involved 
# two separate topics: remote work policies and travel expense guidelines. I anticipated 
# that the retrieval system might struggle to find information covering both topics in 
# a single search. As a result, the model might provide an incomplete response or attempt 
# to fill in missing information without sufficient evidence.
# 
# What Actually Happened: The top three retrieved chunks were the employee benefits 
# policy (0.57), the EcoVolt partnership (0.45), and the company overview (0.45). 
# These similarity scores were lower than those returned for Query 1, which had a 
# top score of 0.76. Additionally, none of the retrieved sources contained information 
# about travel expense policies.
#
# The model responded by addressing only the information it could support. It referenced 
# the hybrid work arrangement and flexible scheduling described in the benefits policy, 
# acknowledged that the available documents did not contain travel expense guidelines, 
# and explained that a direct comparison could not be made. Importantly, the model did 
# not fabricate a travel expense policy or provide unsupported details, which was the 
# appropriate response given the available context.
#
# What I would change: First, I would verify whether the source documents actually contain 
# a travel expense policy. If that information is unavailable, I would instruct the model 
# to acknowledge when the provided context does not contain an answer rather than attempting 
# to generate one. I would also implement a similarity threshold to prevent unrelated chunks 
# from being retrieved and used as supporting information.
#
# If the travel expense policy does exist in the documents, I would divide the original 
# question into two separate searches: one for remote work policies and another for travel 
# expense guidelines. Each query would retrieve its own relevant information, and the 
# results could then be combined into a single response. This approach would help improve 
# retrieval accuracy because each search would focus on one specific topic rather than 
# requiring a single query to retrieve information about two different subjects.

# LlamaIndex Q4
judge_llm = OpenAI(model="gpt-4o-mini")
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)


def evaluate(q):
    """Run both evaluators on the response to q and print the scores."""
    response = query_engine.query(q)
    faith = faithfulness_evaluator.evaluate_response(response=response)
    rel = relevancy_evaluator.evaluate_response(query=q, response=response)
    print(f"\nQuery: {q}")
    print(f"Response: {response}")
    print(f"Faithfulness score: {faith.score} (passing={faith.passing})")
    print(f"Relevancy score: {rel.score} (passing={rel.passing})")


q = "What employee benefits does BrightLeaf offer?"
evaluate(q)

# A question that is clearly not covered by the Brightleaf documents.
evaluate("What is the best recipe for chocolate chip cookies?")

# LlamaIndex Q4 answers:
# 1. A faithfulness score measures how well the information in an AI-generated 
#    response is supported by the retrieved context. A score of 1.0 means that all 
#    claims in the response are supported by the retrieved documents, while a score 
#    of 0.0 indicates that the response contains information that the retrieved 
#    context does not support.
# 2. Faithfulness and relevancy measure two different aspects of an AI-generated 
#    response. Faithfulness evaluates whether the answer is supported by the retrieved 
#    sources, while relevancy evaluates whether the response and retrieved information 
#    address the original question. A response can be faithful to its sources but 
#    irrelevant to the question, or relevant to the question but contain information 
#    that is not supported by the sources.
#
# 3. Yes, the evaluation scores changed significantly between the two questions. 
#    For the employee benefits question, both faithfulness and relevancy received a 
#    score of 1.0, indicating that the retrieved document contained the necessary 
#    information and the model generated an answer that was both accurate and relevant.
#
#    However, for the cookie-related question, both scores dropped to 0.0. The 
#    retrieved documents contained no information about cookies, so the model 
#    appropriately declined to provide an unsupported answer rather than attempting 
#    to fabricate information. Relevancy received a score of 0.0 because neither the 
#    retrieved context nor the response addressed the question. Faithfulness also 
#    received a score of 0.0 because the evaluation found no supporting information 
#    in the retrieved context for the response. In this situation, a low faithfulness 
#    score does not necessarily mean the model fabricated information. Instead, it 
#    demonstrates that the evaluation metrics can produce low scores when the available 
#    context does not contain the information needed to answer a question.
#
#    This comparison illustrates that faithfulness and relevancy measure different aspects 
#    of RAG performance and can both receive low scores when a question falls outside the 
#    scope of the available documents.
#
# 4. LLM-as-a-Judge is an evaluation method that uses a second large language model (LLM) 
#    to assess the quality of another model's response based on a specific set of evaluation 
#    criteria or rubric. For example, the evaluating model might be asked to determine whether 
#    every claim in a generated response is supported by the retrieved context.
#
#    This approach is useful for evaluating RAG systems because their responses are typically 
#    written in natural language and can be expressed in many different ways. Unlike 
#    traditional accuracy measurements, which often rely on comparing an output against 
#    one exact expected answer, LLM-as-a-Judge can evaluate whether a response is accurate, 
#    relevant, and supported by the available sources, even when the wording differs.
#
#    However, one limitation is that the evaluating model can also make mistakes or produce 
#    inconsistent judgments. Therefore, its evaluations should not be treated as completely 
#    reliable, and some results should be manually reviewed to verify that the assigned 
#    scores are reasonable.