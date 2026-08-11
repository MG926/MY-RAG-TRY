#MG314
"""该模块用于召回"""
from sentence_transformers import SentenceTransformer
import chromadb
# import embedding
# import loader
class Retriever:
    def recall(self, query:str, top_k:int =5, bge_path:str ="BAAI/bge-small-zh-v1.5", collection_name:str = "default", db_path:str= "F:/my_RAG/rag_demo/chroma_db"):
        #BGE的问题必须要加提示
        instruct_query = f"为这个句子生成表示以用于检索：{query}"
        #开启模型
        recall_model = SentenceTransformer(bge_path,device="cpu")
        #开启向量库
        client = chromadb.PersistentClient(path=db_path)
        recall_collection = client.get_or_create_collection(name=collection_name)
        #给问题进行编码
        query_embedding = recall_model.encode(
            instruct_query,
            normalize_embeddings=True
        )
        #开始召回查询
        res = recall_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        output = []
        for filename,metadata,dist in zip(res["documents"][0],res["metadatas"][0],res["distances"][0]):
            output.append({
                "content": filename,
                "metadata": metadata,
                "distance": dist
            })
        return output
# my_loader = loader.Loader()
# pdf_path = "F:/my_RAG/rag_demo/data/bianselong.pdf"
# all_chunks = my_loader.read_pdf(pdf_path)
# print(len(all_chunks))
# bge_path = "F:/my_RAG/rag_demo/model/bge-small-zh-v1.5"
# my_embedding = embedding.Embedding()
# pdf_name = "bianselong"
# my_embedding.bge_embedding(all_chunks=all_chunks,bge_path=bge_path,collection_name=pdf_name)
# my_retriever = Retriever()
# query = "谁的手指被咬了？"
# res = my_retriever.recall(query=query,top_k=5,bge_path=bge_path,collection_name=pdf_name)
# for re in res:
#     print("======================================================================")
#     print("content:",re["content"])
#     print("metadata:",re["metadata"])
#     print("distance:",re["distance"])
    
