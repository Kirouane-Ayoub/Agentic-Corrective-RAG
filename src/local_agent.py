from typing import Dict, Optional

import nest_asyncio
from langchain_community.tools import DuckDuckGoSearchResults
from llama_agents import (
    ComponentService,
    ControlPlaneServer,
    PipelineOrchestrator,
    ServiceComponent,
    SimpleMessageQueue,
)
from llama_agents.launchers import LocalLauncher
from llama_index.core import SummaryIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_pipeline import FnComponent, InputComponent, QueryPipeline
from llama_index.core.schema import Document
from models import llm
from retriever import retriever


def duckduckgo_search(query: str) -> str:
    search = DuckDuckGoSearchResults(max_results=15)
    return search.run(query)


nest_asyncio.apply()


message_queue = SimpleMessageQueue()


def to_service_component(component, message_queue, service_name, description):
    server = ComponentService(
        component=component,
        message_queue=message_queue,
        description=description,
        service_name=service_name,
    )
    service_component = ServiceComponent.from_component_service(server)
    return service_component, server


relevancy_prompt_tmpl = PromptTemplate(
    template="""As a grader, your task is to evaluate the relevance of a document retrieved in response to a user's question.

        Retrieved Document:
        -------------------
        {context_str}

        User Question:
        --------------
        {query_str}

        Evaluation Criteria:
        - Consider whether the document contains keywords or topics related to the user's question.
        - The evaluation should not be overly stringent; the primary objective is to identify and filter out clearly irrelevant retrievals.

        Decision:
        - Assign a binary score to indicate the document's relevance.
        - Use 'yes' if the document is relevant to the question, or 'no' if it is not.

        Please provide your binary score ('yes' or 'no') below to indicate the document's relevance to the user question."""
)
query_transform_tmpl = PromptTemplate(
    template="""Your task is to refine a query to ensure it is highly effective for retrieving relevant search results. \n
        Analyze the given input to grasp the core semantic intent or meaning. \n
        Original Query:
        \n ------- \n
        {query_str}
        \n ------- \n
        Your goal is to rephrase or enhance this query to improve its search performance. Ensure the revised query is concise and directly aligned with the intended search objective. \n
        Respond with the optimized query only:"""
)


def run_summarization(retrieved_text: str, search_text: Optional[str] = None) -> str:
    """Run summarization."""
    # use summary index to perform summarization
    search_text = search_text or ""
    documents = [Document(text=retrieved_text + "\n" + search_text)]
    index = SummaryIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    return str(query_engine.query(query_str))


def run_web_search(input_str: str) -> str:
    """Run Web Search."""
    query_transform_qp = QueryPipeline(chain=[query_transform_tmpl, llm])
    transformed_query_str = query_transform_qp.run(query_str=input_str).message.content

    # Conduct a search with the transformed query string and collect the results.
    search_results = duckduckgo_search(transformed_query_str)
    return search_results


def run_retrieval(input_str: str) -> Dict:
    """Run Retrieval."""
    relevancy_qp = QueryPipeline(chain=[relevancy_prompt_tmpl, llm])
    # retrieves a set of nodes
    retrieved_nodes = retriever.retrieve(input_str)
    # runs a relevancy check
    relevancy_results = []
    for node in retrieved_nodes:
        relevancy = relevancy_qp.run(context_str=node.text, query_str=query_str)
        relevancy_results.append(relevancy.message.content.lower().strip())
    contains_irrelevant = "no" in relevancy_results
    # get relevant texts
    relevant_texts = [
        retrieved_nodes[i].text
        for i, result in enumerate(relevancy_results)
        if result == "yes"
    ]
    relevant_text = "\n".join(relevant_texts)
    # returns a dictionary of items
    return {
        "relevant_text": relevant_text,
        "contains_irrelevant": contains_irrelevant,
        "input_str": input_str,
    }


retrieval_component = FnComponent(fn=run_retrieval)
retrieval_component_s, retrieval_server = to_service_component(
    retrieval_component,
    message_queue,
    "Runs a retrieval + relevancy check",
    "retrieval_service",
)

web_search_component = FnComponent(fn=run_web_search)
web_search_component_s, web_server = to_service_component(
    web_search_component, message_queue, "Runs web search", "web_search_service"
)
summary_component = FnComponent(fn=run_summarization)
summary_component_s, summary_server = to_service_component(
    summary_component, message_queue, "Run summarization", "summarization_service"
)

pipeline = QueryPipeline(
    module_dict={
        "input": InputComponent(),
        "retrieval_server": retrieval_component_s,
        "web_server": web_search_component_s,
        "summary_server_no_web": summary_component_s,
        "summary_server_web": summary_component_s,
    }
)
pipeline.add_link("input", "retrieval_server")
pipeline.add_link(
    "retrieval_server",
    "web_server",
    condition_fn=lambda x: x["contains_irrelevant"],
    input_fn=lambda x: x["input_str"],
)
# if web search is called
pipeline.add_link(
    "retrieval_server",
    "summary_server_web",
    dest_key="retrieved_text",
    condition_fn=lambda x: x["contains_irrelevant"],
    input_fn=lambda x: x["relevant_text"],
)
pipeline.add_link("web_server", "summary_server_web", dest_key="search_text")

# if web search is not called
pipeline.add_link(
    "retrieval_server",
    "summary_server_no_web",
    dest_key="retrieved_text",
    condition_fn=lambda x: not x["contains_irrelevant"],
    input_fn=lambda x: x["relevant_text"],
)

pipeline_orchestrator = PipelineOrchestrator(pipeline)

control_plane = ControlPlaneServer(
    message_queue=message_queue,
    orchestrator=pipeline_orchestrator,
)

## Define Launcher
launcher = LocalLauncher(
    [retrieval_server, web_server, summary_server],
    control_plane,
    message_queue,
)

while True:
    query_str = input("Enter query: ")
    result = launcher.launch_single(query_str)
    print(str(result))
