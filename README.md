# Agentic Corrective RAG (CRAG)
### Project Overview

This project showcases a small and simple implementation of **[Corrective Retrieval Augmented Generation (CRAG)](https://arxiv.org/pdf/2401.15884)**  utilizing the 🦙 llama-agents 🤖 framework. The objective of this project is to enhance the robustness and accuracy of AI-generated responses by correcting potential retrieval errors in the standard **Retrieval-Augmented Generation (RAG)** process.

### What is CRAG?
**Corrective Retrieval Augmented Generation (CRAG)** is an advanced AI technique designed to address the common issue of irrelevant or incorrect document retrieval in standard RAG systems. CRAG introduces a corrective mechanism that evaluates the relevance of retrieved documents and takes appropriate actions to ensure the generation of accurate and reliable information. 

The core components of CRAG include:

- **Retrieval Evaluator:** Assesses the quality and relevance of retrieved documents.
- **Confidence-Based Actions:** Determines the confidence level in the retrieved documents and triggers corrective actions as needed.
- **Knowledge Refinement:** Refines and composes retrieved information to improve relevance and accuracy.
- **Web Search Integration:** Performs web searches to gather additional information when necessary.



## Getting Started

#### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Kirouane-Ayoub/Agentic-Corrective-RAG.git
   cd Agentic-Corrective-RAG
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

#### Configuration

- Embedding and llm models configurations are managed in `src/settings.py`. Adjust the settings according to your environment and requirements.
- Create a `.env` file with the following content:

    ```
    CO_API_KEY=...
    ```
- Move your data files (pdf, csv, md, etc.) to the `src/data_source` folder .
#### Run the agent : 
- To run the agent locally use this command :

    ```
    python src/local_agent.py
    ```
