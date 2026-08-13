#MG314
"""该模块负责向量化操作"""
from sentence_transformers import SentenceTransformer
import chromadb
# import loader
class Embedding:
    def __init__(self, bge_path:str ="BAAI/bge-small-zh-v1.5", db_path:str="F:/my_RAG/rag_demo/chroma_db", device:str="cpu"):
        #设置device
        self.device = device
        #采用BGE方法，默认联网下载，如果有本地模型则上传至bge_path，默认采用cpu
        self.bge_model = SentenceTransformer(bge_path,device=self.device)
        #本地化存储
        self.client = chromadb.PersistentClient(path=db_path)
    def bge_embedding(self, all_chunks:list[dict], collection_name:str = "default"):
        #输入的chunks格式为[{"content":chunks,"metadata":{"page_num": idx}}]
        #拆分文本和文本信息
        texts = [item["content"] for item in all_chunks]
        ids = [f"{collection_name}_{i}" for i in range(len(texts))]
        metadatas = [item["metadata"] for item in all_chunks]

        # 开启向量化
        embeddings = self.bge_model.encode(
            texts,
            normalize_embeddings= True,
            batch_size= 8,
            show_progress_bar= True
        ).tolist()

        # 存入向量库
        #向量表名字命名为传入的pdf名字，没有则为default
        collection = self.client.get_or_create_collection(name=collection_name)
        #将数据存入向量库
        collection.add(
            ids= ids,
            embeddings= embeddings,
            metadatas= metadatas,
            documents= texts
        ) 
        print(f"内容数量:{len(texts)}")
'''测试代码'''
# if __name__ == "__main__":
#     my_loader = loader.Loader()
#     pdf_path = "F:/my_RAG/rag_demo/data/bianselong.pdf"
#     all_chunks = my_loader.read_pdf(pdf_path)
#     print(len(all_chunks))
#     bge_path = "F:/my_RAG/rag_demo/model/bge-small-zh-v1.5"
#     my_embedding = Embedding()
#     pdf_name = "bianselong"
#     my_embedding.bge_embedding(all_chunks=all_chunks,bge_path=bge_path,collection_name=pdf_name)