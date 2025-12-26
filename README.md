# RAG Projects

A comprehensive collection of Retrieval-Augmented Generation (RAG) implementations, techniques, and applications built with LangChain and LangGraph. This repository serves as both a learning resource and a practical toolkit for building production-ready RAG systems.

## 📚 Overview

This repository contains:

- **RAG Techniques**: Advanced RAG strategies and implementations for improving retrieval and generation quality
- **RAG Applications**: Production-ready RAG systems for specific use cases (e.g., SQL-RAG)
- **Educational Resources**: Well-documented code with theoretical explanations and practical examples

## 🗂️ Repository Structure

```
rag_projects/
├── rag-techniques/          # Advanced RAG techniques and strategies
│   ├── Basic_RAG.ipynb
│   ├── Corrective RAG (CRAG).ipynb
│   ├── Rag_HyDE.ipynb
│   ├── Rag_Step_Back.ipynb
│   ├── Rag_Routing.ipynb
│   ├── Rag_Decomposition.ipynb
│   ├── Rag_and_multi_query_generator..ipynb
│   ├── RAG-Fusion.ipynb
│   ├── Rag_Multi_Representation_Indexing.ipynb
│   ├── Query_structuring.ipynb
│   ├── images/              # Supporting images and diagrams
│   └── README.md            # Detailed documentation for all techniques
│
├── sql-rag/                 # SQL-RAG: Natural language to SQL agent
│   ├── sql_rag_agent.py    # Main agent implementation
│   ├── logger.py            # Logging utilities
│   ├── db/                  # Sample database (Chinook)
│   └── README.md            # SQL-RAG documentation
│
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (or compatible LLM API)
- (Optional) Tavily API key for web search features

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag_projects
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   LANGCHAIN_API_KEY=your_langchain_api_key_here  # Optional, for tracing
   TAVILY_API_KEY=your_tavily_api_key_here        # Optional, for CRAG web search
   ```

## 📖 Projects

### 🎯 RAG Techniques (`rag-techniques/`)

A collection of advanced RAG strategies and implementations. Each notebook demonstrates a specific technique for improving RAG performance:

| Technique | Description | Use Case |
|-----------|-------------|----------|
| **Basic RAG** | Foundational RAG implementation | Learning the basics |
| **Multi-Query Generator** | Generates multiple query variations | Improving recall |
| **RAG-Fusion** | Uses Reciprocal Rank Fusion (RRF) | Better document ranking |
| **HyDE** | Hypothetical Document Embeddings | Bridging semantic gaps |
| **Step-Back Prompting** | Retrieves broader context | Comprehensive answers |
| **Routing** | Intelligent query routing | Multi-domain systems |
| **Decomposition** | Breaks complex questions into sub-questions | Complex queries |
| **Multi-Representation Indexing** | Indexes summaries, retrieves full docs | Cost optimization |
| **Query Structuring** | Converts NL to structured queries | Complex search systems |
| **Corrective RAG (CRAG)** | Self-grading and web search fallback | Robust retrieval |

📚 **See [rag-techniques/README.md](rag-techniques/README.md) for detailed documentation.**

### 🔍 SQL-RAG (`sql-rag/`)

A production-ready natural language to SQL agent with:

- **Text-to-SQL Conversion**: Converts natural language questions into SQL queries
- **Role-Based Access Control**: Enforces security policies based on user roles
- **Self-Correction**: Automatically fixes SQL errors using LLM feedback
- **Comprehensive Logging**: Tracks all queries, errors, and performance metrics
- **Interactive CLI**: User-friendly command-line interface

**Quick Start:**
```bash
cd sql-rag
python sql_rag_agent.py
```

📚 **See [sql-rag/README.md](sql-rag/README.md) for detailed documentation.**

## 🎓 Learning Path

### For Beginners

1. Start with **Basic RAG** to understand core concepts
2. Explore **Multi-Query Generator** to see query expansion
3. Try **RAG-Fusion** to understand ranking and fusion
4. Experiment with **Step-Back Prompting** for context retrieval

### For Intermediate Users

1. Study **Routing** for multi-domain systems
2. Learn **Decomposition** for complex queries
3. Explore **Multi-Representation Indexing** for optimization
4. Implement **Query Structuring** for structured search

### For Advanced Users

1. Deep dive into **HyDE** for semantic matching
2. Build production systems with **SQL-RAG**
3. Customize **CRAG** for your use case
4. Combine multiple techniques for optimal performance

## 🛠️ Key Technologies

- **LangChain**: Framework for building LLM applications
- **LangGraph**: State-based workflows and agent orchestration
- **OpenAI**: LLM provider (GPT-4, GPT-3.5, etc.)
- **ChromaDB**: Vector database for embeddings
- **SQLite**: Relational database (for SQL-RAG)
- **Tavily**: Web search API (for CRAG)

## 📝 Adding New Projects

This repository is designed to grow! When adding new projects:

### For RAG Techniques

1. Create a new notebook in `rag-techniques/`
2. Follow the naming convention: `Rag_<TechniqueName>.ipynb` or `RAG-<TechniqueName>.ipynb`
3. Update `rag-techniques/README.md` with your technique
4. Add any supporting images to `rag-techniques/images/`

### For New Applications

1. Create a new folder in the root: `your-project-name/`
2. Include:
   - Main implementation files
   - `README.md` with documentation
   - Any required data or configuration files
3. Update this README with your project description
4. Add project-specific dependencies to `requirements.txt` if needed

### Documentation Standards

- Include theoretical background
- Explain code components clearly
- Provide usage examples
- Document dependencies and setup
- Add diagrams/images when helpful

## 🔧 Development

### Running Notebooks

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Document functions and classes
- Include docstrings for complex logic

### Testing

- Test notebooks can be run interactively
- SQL-RAG includes logging for debugging
- Use LangChain tracing for detailed analysis

## 📊 Project Status

### ✅ Completed

- [x] Basic RAG
- [x] Multi-Query Generator
- [x] RAG-Fusion
- [x] HyDE
- [x] Step-Back Prompting
- [x] Routing
- [x] Decomposition
- [x] Multi-Representation Indexing
- [x] Query Structuring
- [x] Corrective RAG (CRAG)
- [x] SQL-RAG Agent

### 🚧 In Progress

- [ ] Additional RAG techniques (coming soon)
- [ ] More production applications (coming soon)

### 💡 Planned

- [ ] Graph RAG
- [ ] Adaptive RAG
- [ ] Self-RAG
- [ ] RAG with re-ranking
- [ ] Multi-modal RAG
- [ ] RAG evaluation framework

## 🤝 Contributing

Contributions are welcome! When contributing:

1. Follow the existing code structure
2. Add comprehensive documentation
3. Include examples and use cases
4. Update relevant README files
5. Test your implementations

## 📚 Resources

### Documentation

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

### Research Papers

- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884)
- [HyDE: Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496)

### Learning Resources

- [LangChain Course](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)
- [RAG Tutorials](https://python.langchain.com/docs/use_cases/question_answering/)

## 📄 License

This repository is for educational and research purposes. Please check individual project licenses if applicable.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses the [Chinook](https://github.com/lerocha/chinook-database) sample database
- Inspired by research papers and the RAG community

## 📧 Contact

For questions, suggestions, or contributions, please open an issue or submit a pull request.

---

**Happy RAG Building! 🚀**
