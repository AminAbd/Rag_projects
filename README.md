# RAG Projects

This directory contains Retrieval-Augmented Generation (RAG) projects implemented using LangChain and LangGraph.

## Projects

### Basic RAG

**File:** `Basic_RAG.ipynb`

A foundational RAG implementation that demonstrates the core components of a Retrieval-Augmented Generation system. This notebook provides a simple, straightforward example of how to build a basic RAG pipeline from scratch.

#### Overview

This project implements a basic RAG system that:
1. **Loads** documents from web sources
2. **Splits** documents into manageable chunks
3. **Embeds** and stores documents in a vector database
4. **Retrieves** relevant documents based on queries
5. **Generates** answers using retrieved context

#### Features

- **Web Document Loading**: Uses `WebBaseLoader` to extract content from web pages
- **Text Splitting**: Implements `RecursiveCharacterTextSplitter` with configurable chunk size (1000 chars) and overlap (200)
- **Vector Storage**: Uses ChromaDB for efficient similarity search
- **RAG Chain**: Combines retrieval, formatting, and generation into a single chain
- **Simple Interface**: Easy-to-use chain that takes a question and returns an answer

#### Workflow

```
Question → Retrieve Documents → Format Context → Generate Answer
```

#### Key Components

1. **Document Loader**: Extracts content from web pages using BeautifulSoup
2. **Text Splitter**: Breaks documents into chunks (1000 chars, 200 overlap)
3. **Vector Store**: ChromaDB with OpenAI embeddings
4. **Retriever**: Semantic search over embedded documents
5. **RAG Chain**: Orchestrates retrieval and generation

#### Usage

1. Set up your `.env` file with API keys:
   ```
   OPENAI_API_KEY=your_key_here
   LANGCHAIN_API_KEY=your_key_here (optional, for tracing)
   ```

2. Run the notebook cells in order:
   - Environment setup
   - Document loading and indexing
   - RAG chain creation
   - Query execution

3. Ask questions:
   ```python
   rag_chain.invoke({"question": "What is Task Decomposition?"})
   ```

#### Example Questions

- "What is Task Decomposition?"
- "How do agents use tools?"
- Any question related to the indexed documents

#### Requirements

- OpenAI API key
- Python packages (see main `requirements.txt`)

#### Key Dependencies

- `langchain`
- `langchain-openai`
- `langchain-community`
- `langchain-text-splitters`
- `chromadb`

---

### RAG with Multi-Query Generator

**File:** `Rag_and_multi_query_generator..ipynb`

An advanced RAG implementation that uses **multi-query generation** to improve retrieval quality by generating multiple query variations from a single user question.

#### Overview

Traditional RAG systems retrieve documents based on a single query, which can miss relevant information due to the limitations of distance-based similarity search. This implementation addresses this by:

1. **Generating multiple query variations** from the original question
2. **Retrieving documents** for each query variation
3. **Combining unique documents** from all retrievals
4. **Generating the final answer** using the comprehensive context

#### How It Works

The multi-query generator creates 5 different perspectives of the user's question, helping overcome limitations of similarity search by:
- Using different phrasings and terminology
- Exploring various angles of the same question
- Capturing semantic variations that might be missed by a single query

#### Workflow

```
Original Question
    ↓
Generate 5 Query Variations
    ↓
Retrieve Documents for Each Query
    ↓
Get Unique Union of All Documents
    ↓
Generate Final Answer with Combined Context
```

#### Key Components

1. **Multi-Query Generator**: Uses an LLM to generate 5 alternative versions of the user question
2. **Parallel Retrieval**: Retrieves documents for each generated query simultaneously
3. **Document Deduplication**: Combines and deduplicates documents from all retrievals
4. **RAG Chain**: Generates the final answer using the comprehensive document set

#### Code Structure

**Indexing:**
- Loads documents from web sources
- Splits documents into chunks (300 chars, 50 overlap)
- Creates embeddings and stores in ChromaDB

**Multi-Query Generation:**
```python
generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)
```

**Retrieval Chain:**
```python
retrieval_chain = generate_queries | retriever.map() | get_unique_union
```

**Final RAG Chain:**
- Combines retrieval with question
- Generates answer using all retrieved context

#### Benefits

- **Better Coverage**: Retrieves more relevant documents by exploring multiple query angles
- **Improved Accuracy**: Reduces the chance of missing important information
- **Semantic Diversity**: Captures different ways the same concept might be expressed
- **Robust Retrieval**: Less dependent on the exact wording of the original question

#### Usage

1. Set up your `.env` file with API keys:
   ```
   OPENAI_API_KEY=your_key_here
   LANGCHAIN_API_KEY=your_key_here (optional, for tracing)
   ```

2. Run the notebook cells in order:
   - Environment setup
   - Document indexing
   - Multi-query generator setup
   - Retrieval chain creation
   - Final RAG chain execution

3. Ask questions:
   ```python
   question = "What is task decomposition for LLM agents?"
   answer = final_rag_chain.invoke({"question": question})
   ```

#### Example

**Original Question:** "What is task decomposition for LLM agents?"

**Generated Queries:**
- "How do LLM agents break down tasks?"
- "What methods are used for task decomposition in language model agents?"
- "Explain task decomposition strategies for AI agents"
- "What is the process of decomposing tasks in LLM systems?"
- "How are complex tasks divided in language model agent architectures?"

Each query retrieves potentially different documents, which are then combined for a more comprehensive answer.

#### Requirements

- OpenAI API key
- Python packages (see main `requirements.txt`)

#### Key Dependencies

- `langchain`
- `langchain-openai`
- `langchain-community`
- `langchain-text-splitters`
- `chromadb`

---
## RAG with Multi-Query Fusion (RAG-Fusion + RRF)

**File:** `RAG-Fusion.ipynb`

This project extends multi-query RAG by improving **document selection and ordering** using **Reciprocal Rank Fusion (RRF)**. Instead of blindly concatenating all retrieved documents, it ranks and filters them before generation, producing a cleaner and more informative context for the LLM.

---

### Motivation

Standard RAG systems rely on a **single query → single retrieval** step, which is fragile due to the limitations of distance-based similarity search. Multi-query RAG improves recall by generating multiple query variations, but a naive implementation often concatenates all retrieved documents, which can:

- introduce irrelevant or redundant chunks  
- exceed context limits  
- dilute important information  
- reduce answer quality  

This implementation addresses these issues by **ranking retrieved documents across queries before concatenation**.

---

### Core Idea

Instead of treating all retrieved documents equally, this approach:

1. Generates **4 semantically related queries** from the original question  
2. Retrieves **ranked document lists** for each query  
3. **Fuses the rankings** using Reciprocal Rank Fusion (RRF)  
4. Selects and orders only the **most consistently relevant documents**  
5. Uses the ordered documents as context for generation  

The result is a **smaller, higher-quality, and order-aware context**.

---

### What Is Reciprocal Rank Fusion (RRF)?

Reciprocal Rank Fusion combines multiple ranked lists into a single global ranking using **rank positions only**, not raw similarity scores.

For each document:

score = Σ 1 / (rank + k)


Where:
- `rank` is the document’s position in a retrieval list  
- `k` is a smoothing constant (typically 60)  

Documents that:
- appear across **multiple query results**, and/or  
- appear **highly ranked** in at least one query  

receive higher fused scores.

**Important:** RRF scores are used **only to rank documents**, not to weight or truncate document content. Full document chunks are passed to the LLM.

---

### Why Ranking Matters

LLMs do not treat all context equally:
- earlier context has more influence than later context  
- irrelevant chunks reduce answer quality  
- long, unordered context dilutes attention  

By ranking documents before concatenation, this approach:
- prioritizes the most informative chunks  
- removes low-value noise  
- improves factual grounding  
- produces more consistent and accurate answers  

---

### Workflow

Original Question
↓
Generate Multiple Query Variations
↓
Parallel Vector Retrieval (per query)
↓
Rank Fusion with RRF
↓
Select Top-K Documents
↓
Concatenate Ordered Context
↓
Generate Final Answer


---

### Code Structure

**Indexing:**
- Loads documents from web sources
- Splits documents into chunks (300 chars, 50 overlap)
- Creates embeddings and stores in ChromaDB

**Query Generation:**
```python
generate_queries = (
    prompt_rag_fusion 
    | ChatOpenAI(temperature=0)
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)
```

**Reciprocal Rank Fusion:**
```python
def reciprocal_rank_fusion(results: list[list], k=60):
    # Fuses multiple ranked lists using RRF formula
    # Returns reranked documents with fused scores
```

**Retrieval Chain:**
```python
retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
```

**Final RAG Chain:**
- Combines fused retrieval results with question
- Generates answer using ranked context

### Key Differences vs Normal RAG

| Aspect | Normal RAG | RAG-Fusion + RRF |
|------|-----------|------------------|
| Queries | Single | Multiple (4 queries) |
| Retrieval | One ranked list | Multiple ranked lists |
| Document Selection | Top-K from one query | Consensus across queries |
| Ranking | Vector similarity only | Rank-based fusion (RRF) |
| Context Quality | Noisy, order-dependent | Filtered, ordered |
| Robustness | Low | High |

---

### Usage

1. Set up your `.env` file with API keys:
   ```
   OPENAI_API_KEY=your_key_here
   LANGCHAIN_API_KEY=your_key_here (optional, for tracing)
   ```

2. Run the notebook cells in order:
   - Environment setup
   - Document indexing
   - Query generation setup
   - RRF implementation
   - Retrieval chain creation
   - Final RAG chain execution

3. Ask questions:
   ```python
   question = "What is task decomposition for LLM agents?"
   answer = final_rag_chain.invoke({"question": question})
   ```

**Note:** The notebook uses `itemgetter` from `operator` module - make sure to import it:
```python
from operator import itemgetter
```

### Summary

RAG-Fusion with Reciprocal Rank Fusion improves Retrieval-Augmented Generation by replacing single-query retrieval with **multi-query consensus ranking**, ensuring that only the most consistently relevant documents are used—and used in the right order—before generation.


---
### RAG with Question Decomposition

**File:** `Rag_Decomposition.ipynb`

An advanced RAG implementation that uses **question decomposition** to break down complex questions into simpler sub-questions, answer them individually, and then synthesize a comprehensive final answer.

#### Overview

Complex questions often require answering multiple related sub-questions to provide a complete answer. This implementation addresses this by:

1. **Decomposing** complex questions into 3 simpler sub-questions
2. **Answering** each sub-question using RAG
3. **Synthesizing** the individual answers into a final comprehensive answer

#### How It Works

Instead of trying to answer a complex question directly, this approach:
- Breaks the question into manageable sub-problems
- Retrieves relevant documents for each sub-question
- Answers each sub-question independently
- Combines the answers to form a complete response

#### Two Approaches

**1. Recursive Answering:**
- Answers sub-questions sequentially
- Uses previously answered Q&A pairs as context for subsequent questions
- Builds up knowledge incrementally
- Each answer can reference previous answers

**2. Individual Answering:**
- Answers each sub-question independently
- No dependency between sub-questions
- Synthesizes all answers at the end
- More parallelizable approach

#### Workflow

```
Complex Question
    ↓
Decompose into 3 Sub-Questions
    ↓
For each sub-question:
    ↓
Retrieve Relevant Documents
    ↓
Generate Answer
    ↓
Synthesize All Answers
    ↓
Final Comprehensive Answer
```

#### Key Components

1. **Question Decomposer**: Uses an LLM to break complex questions into 3 sub-questions
2. **Sub-Question RAG**: Applies RAG to each sub-question independently
3. **Answer Synthesis**: Combines individual answers into a final comprehensive response
4. **Context Building**: (Recursive approach) Uses previous Q&A pairs as context

#### Code Structure

**Question Decomposition:**
```python
generate_queries_decomposition = (
    prompt_decomposition 
    | llm 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)
```

**Recursive Approach:**
- Answers sub-questions sequentially
- Uses previous Q&A pairs as context
- Builds knowledge incrementally

**Individual Approach:**
- Answers each sub-question independently
- Formats Q&A pairs
- Synthesizes final answer from all pairs

#### Benefits

- **Better Coverage**: Ensures all aspects of a complex question are addressed
- **Improved Accuracy**: Breaking down questions leads to more focused retrieval
- **Comprehensive Answers**: Synthesizes multiple perspectives into one answer
- **Handles Complexity**: Better suited for multi-part questions
- **Modular Approach**: Each sub-question can be answered with relevant context

#### Usage

1. Set up your `.env` file with API keys:
   ```
   OPENAI_API_KEY=your_key_here
   LANGCHAIN_API_KEY=your_key_here (optional, for tracing)
   ```

2. Run the notebook cells in order:
   - Environment setup
   - Document indexing
   - Question decomposition setup
   - Choose approach (recursive or individual)
   - Execute and synthesize

3. Ask complex questions:
   ```python
   question = "What are the main components of an LLM-powered autonomous agent system?"
   questions = generate_queries_decomposition.invoke({"question": question})
   # Then use recursive or individual approach
   ```

#### Example

**Original Question:** "What are the main components of an LLM-powered autonomous agent system?"

**Decomposed Sub-Questions:**
1. "What is LLM technology and how does it work in autonomous agent systems?"
2. "What are the specific components that make up an LLM-powered autonomous agent system?"
3. "How do the main components of an LLM-powered autonomous agent system interact with each other to enable autonomous behavior?"

Each sub-question is answered independently, then synthesized into a comprehensive final answer.

#### Requirements

- OpenAI API key
- Python packages (see main `requirements.txt`)

#### Key Dependencies

- `langchain`
- `langchain-openai`
- `langchain-community`
- `langchain-text-splitters`
- `chromadb`

---

### RAG with Step-Back Prompting

**File:** `Rag_Step_Back.ipynb`

An advanced RAG implementation that uses **step-back prompting** to improve answer quality by retrieving both specific and general context. Instead of only searching for documents directly related to the question, it first generates a more generic "step-back" question to retrieve broader foundational context.

#### Overview

Step-back prompting addresses the limitation that specific questions may not retrieve enough foundational or general knowledge. This implementation:

1. **Generates a step-back question** - A more generic version of the original question
2. **Retrieves dual context** - Documents for both the original and step-back questions
3. **Combines contexts** - Uses both specific and general context to generate comprehensive answers

#### How It Works

The step-back approach:
- Takes a specific question (e.g., "What is task decomposition for LLM agents?")
- Generates a more generic question (e.g., "What is the process of breaking down tasks for LLM agents?")
- Retrieves documents for both questions
- Uses the combined context to provide a more comprehensive answer

#### Key Concept

**Step-Back Questions** are more general versions that:
- Capture the broader concept behind the specific question
- Retrieve foundational knowledge and context
- Provide background information that enriches the answer
- Help avoid overly narrow retrieval that misses important context

#### Workflow

```
Original Question
    ↓
Generate Step-Back Question (more generic)
    ↓
Retrieve Context for Original Question
    ↓
Retrieve Context for Step-Back Question
    ↓
Combine Both Contexts
    ↓
Generate Comprehensive Answer
```

#### Key Components

1. **Few-Shot Step-Back Generator**: Uses examples to learn how to generate generic questions
2. **Dual Retrieval**: Retrieves documents for both original and step-back questions
3. **Context Combination**: Merges both contexts before generation
4. **Comprehensive Answering**: Uses combined context to provide thorough answers

#### Code Structure

**Few-Shot Examples:**
```python
examples = [
    {"input": "Could the members of The Police perform lawful arrests?",
     "output": "what can the members of The Police do?"},
    {"input": "Jan Sindel's was born in what country?",
     "output": "what is Jan Sindel's personal history?"}
]
```

**Step-Back Question Generation:**
```python
generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
```

**Dual Context Retrieval:**
```python
chain = (
    {
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        "step_back_context": generate_queries_step_back | retriever,
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)
```

#### Benefits

- **Broader Context**: Retrieves foundational knowledge, not just specific answers
- **Better Grounding**: Provides background information that enriches answers
- **Comprehensive Answers**: Combines specific and general knowledge
- **Reduced Narrowness**: Avoids missing important context due to overly specific queries
- **Improved Understanding**: Helps the LLM understand the broader context of the question

#### Usage

1. Set up your `.env` file with API keys:
   ```
   OPENAI_API_KEY=your_key_here
   LANGCHAIN_API_KEY=your_key_here (optional, for tracing)
   ```

2. Run the notebook cells in order:
   - Environment setup
   - Document indexing
   - Few-shot examples setup
   - Step-back question generator
   - Dual retrieval chain
   - Execution

3. Ask questions:
   ```python
   question = "What is task decomposition for LLM agents?"
   answer = chain.invoke({"question": question})
   ```

#### Example

**Original Question:** "What is task decomposition for LLM agents?"

**Step-Back Question:** "What is the process of breaking down tasks for LLM agents?"

The step-back question retrieves broader context about task decomposition processes, while the original question retrieves specific information. Both contexts are combined to provide a comprehensive answer.

#### Requirements

- OpenAI API key
- Python packages (see main `requirements.txt`)

#### Key Dependencies

- `langchain`
- `langchain-openai`
- `langchain-community`
- `langchain-text-splitters`
- `chromadb`

---

### RAG with Hypothetical Document Embeddings (HyDE)

**File:** `Rag_HyDE.ipynb`

An advanced RAG implementation that uses **Hypothetical Document Embeddings (HyDE)** to improve retrieval by generating a hypothetical answer document and using it for retrieval instead of the original question.

#### Overview

HyDE addresses the semantic gap between questions and documents. Instead of searching with the question directly, this approach:

1. **Generates a hypothetical document** - Creates a passage that would answer the question
2. **Uses hypothetical document for retrieval** - Searches for documents similar to the hypothetical answer
3. **Retrieves relevant documents** - Finds documents that match the hypothetical answer's style and content
4. **Generates final answer** - Uses retrieved documents to answer the original question

#### How It Works

The HyDE approach:
- Takes a question (e.g., "What is task decomposition for LLM agents?")
- Generates a hypothetical scientific paper passage that would answer it
- Uses that hypothetical passage to retrieve similar documents
- Finds documents that are semantically similar to what an answer would look like
- Uses retrieved documents to generate the final answer

#### Key Concept

**Hypothetical Documents** are:
- Generated passages that represent what an ideal answer would look like
- Written in the style and format of the target documents (e.g., scientific papers)
- Used as queries for semantic search instead of the original question
- More effective at finding relevant documents because they match the document format

#### Workflow

```
Original Question
    ↓
Generate Hypothetical Answer Document
    ↓
Use Hypothetical Document for Retrieval
    ↓
Retrieve Similar Documents
    ↓
Generate Final Answer Using Retrieved Documents
```

#### Key Components

1. **Hypothetical Document Generator**: Uses an LLM to generate a passage that would answer the question
2. **HyDE Retrieval**: Uses the hypothetical document to search the vector store
3. **Document Retrieval**: Finds documents similar to the hypothetical answer
4. **RAG Chain**: Generates the final answer using retrieved documents

#### Code Structure

**Hypothetical Document Generation:**
```python
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)

generate_docs_for_retrieval = (
    prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser()
)
```

**Retrieval Chain:**
```python
retrieval_chain = generate_docs_for_retrieval | retriever
retrieved_docs = retrieval_chain.invoke({"question": question})
```

**Final RAG Chain:**
- Uses retrieved documents and original question
- Generates comprehensive answer

#### Benefits

- **Better Semantic Matching**: Hypothetical documents match the format and style of target documents
- **Improved Retrieval**: Finds documents that are similar to what an answer would look like
- **Reduced Semantic Gap**: Bridges the gap between question format and document format
- **More Relevant Results**: Retrieves documents that are semantically closer to ideal answers
- **Format-Aware**: Can be tailored to match specific document types (scientific papers, articles, etc.)

#### Usage

1. Set up your `.env` file with API keys:
   ```
   OPENAI_API_KEY=your_key_here
   LANGCHAIN_API_KEY=your_key_here (optional, for tracing)
   ```

2. Run the notebook cells in order:
   - Environment setup
   - Document indexing
   - Hypothetical document generator setup
   - Retrieval chain creation
   - Final RAG chain execution

3. Ask questions:
   ```python
   question = "What is task decomposition for LLM agents?"
   retrieved_docs = retrieval_chain.invoke({"question": question})
   answer = final_rag_chain.invoke({"context": retrieved_docs, "question": question})
   ```

#### Example

**Original Question:** "What is task decomposition for LLM agents?"

**Hypothetical Document Generated:**
A scientific paper-style passage explaining task decomposition, its methods, and applications in LLM agents.

**Retrieval:** Uses the hypothetical passage to find similar documents in the vector store.

**Final Answer:** Generated using the retrieved documents that match the hypothetical answer format.

#### Requirements

- OpenAI API key
- Python packages (see main `requirements.txt`)

#### Key Dependencies

- `langchain`
- `langchain-openai`
- `langchain-community`
- `langchain-text-splitters`
- `chromadb`

---

### Corrective RAG (CRAG)

**File:** `Corrective RAG (CRAG).ipynb`

Corrective-RAG (CRAG) is an advanced RAG strategy that incorporates self-reflection and self-grading on retrieved documents. This implementation uses LangGraph to create a workflow that:

1. **Retrieves** documents from a vector database
2. **Grades** the relevance of retrieved documents
3. **Transforms** the query if documents are irrelevant
4. **Searches** the web (using Tavily) when needed
5. **Generates** the final answer using the most relevant context

#### Features

- **Document Relevance Grading**: Uses an LLM to assess whether retrieved documents are relevant to the question
- **Query Transformation**: Automatically rewrites queries for better web search results when needed
- **Web Search Integration**: Uses Tavily Search API to supplement retrieval when documents are not relevant
- **LangGraph Workflow**: Implements a state-based graph that orchestrates the entire RAG pipeline

#### Workflow

```
START → Retrieve → Grade Documents → Decision
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            Transform Query                    Generate
                    ↓
            Web Search
                    ↓
                Generate → END
```

#### Requirements

- OpenAI API key (set in `.env` file)
- Tavily API key (for web search functionality)
- Python packages (see main `requirements.txt`)

#### Key Components

1. **Retrieval Grader**: Evaluates document relevance using structured LLM output
2. **RAG Chain**: Standard RAG pipeline with retrieval, formatting, and generation
3. **Question Rewriter**: Optimizes queries for web search
4. **Graph State**: Manages state across nodes (question, documents, generation, web_search flag)

#### Usage

1. Set up your `.env` file with API keys:
   ```
   OPENAI_API_KEY=your_key_here
   TAVILY_API_KEY=your_key_here
   ```

2. Run the notebook cells in order:
   - Setup and index creation
   - LLM configuration
   - Graph definition
   - Graph compilation
   - Execution

3. Test with different questions:
   ```python
   inputs = {"question": "Your question here"}
   for output in app.stream(inputs):
       # Process output
   ```

#### Example Questions

- "What are the types of agent memory?"
- "How does the AlphaCodium paper work?"
- Any question that may require web search when local documents are insufficient

## Setup

Make sure you have all required packages installed:

```bash
pip install -r ../requirements.txt
```

Key dependencies:
- `langgraph`
- `langchain`
- `langchain-openai`
- `langchain-community`
- `langchain-text-splitters`
- `chromadb`
- `tiktoken`
- `langchainhub`

## Notes

- The notebooks use ChromaDB for vector storage
- Web search (in CRAG) is triggered automatically when retrieved documents are not relevant
- Multi-query generator creates 5 query variations by default (can be modified)
- The implementation skips the knowledge refinement phase mentioned in the CRAG paper (can be added as an additional node if needed)

## References

- [Corrective RAG Paper](https://arxiv.org/abs/2401.15884)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)




