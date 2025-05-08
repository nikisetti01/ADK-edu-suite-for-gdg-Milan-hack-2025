from langchain.graphs import GraphRAGRetriever
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Placeholder setup (graph retriever integration)
graph_retriever = GraphRAGRetriever.from_graph_data_source("hackathon-graph")

llm = OpenAI(temperature=0)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=graph_retriever
)

query = "Come funziona GraphRAG?"
result = rag_chain.run(query)
print(result)