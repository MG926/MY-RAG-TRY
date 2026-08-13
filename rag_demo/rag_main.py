from src import loader,retriever,reranker,embedding,llm
my_loader = loader.Loader()
pdf_path = "F:/my_RAG/rag_demo/data/bianselong.pdf"
all_chunks = my_loader.read_pdf(pdf_path)
print(len(all_chunks))
bge_path = "F:/my_RAG/rag_demo/model/bge-small-zh-v1.5"
my_embedding = embedding.Embedding(bge_path=bge_path)
pdf_name = "bianselong"
my_embedding.bge_embedding(all_chunks=all_chunks,collection_name=pdf_name)
my_retriever = retriever.Retriever(bge_path=bge_path)
query = "谁的手指被咬了？"
res = my_retriever.recall(query=query,top_k=5,collection_name=pdf_name)
for re in res:
    print("======================================================================")
    print("content:",re["content"])
    print("metadata:",re["metadata"])
    print("distance:",re["distance"])
my_reranker = reranker.Reranker(bge_path=bge_path)
res = my_reranker.rerank(query=query,retrieved_lists=res,top_k=3)
print(len(res))
for re in res:
    print("======================================================================")
    print("content:",re["content"])
    print("metadata:",re["metadata"])
    print("distance:",re["distance"])
    print("rerank_score",re["rerank_score"])
llm = llm.LLM()
answer = llm.generate_answer(query,res)
print(answer)
