#MG314
"""该模块用于重排"""
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
# import loader
# import embedding
# import retriever
class Reranker:
    def __init__(self,bge_path:str ="BAAI/bge-small-zh-v1.5", device:str="cpu", batch_size:int=8):
        self.device = device
        #模型批处理分类大小
        self.batch_size = batch_size
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(bge_path)
        self.tokenizer = AutoTokenizer.from_pretrained(bge_path)
        self.reranker_model.to(self.device)
        self.reranker_model.eval()

    def rerank(self,query:str, retrieved_lists:list, top_k:int=3):
        if(not retrieved_lists):
            return []
        all_scores = []
        pairs = [[query,item["content"]] for item in retrieved_lists]

        with torch.no_grad():
            for pairs_start in range(0,len(pairs),self.batch_size):
                batch_pairs = pairs[pairs_start:pairs_start+self.batch_size] if pairs_start+self.batch_size<=len(pairs) else pairs[pairs_start:len(pairs)]
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,  #开启补齐
                    truncation=True,  #开启截断
                    return_tensors="pt",  #转换张量
                    max_length=512  #最大长度512token
                ).to(self.device)
                outputs = self.reranker_model(**inputs)
                batch_scores = outputs.logits.squeeze(dim=-1).cpu().numpy()
                all_scores.extend(batch_scores.tolist())

        scored_result = []
        for item,score in zip(retrieved_lists,all_scores):
            new_item = item.copy()
            new_item["rerank_score"] = float(score)
            scored_result.append(new_item)

        scored_result.sort(key=lambda x:x["rerank_score"],reverse=True)
        return scored_result[:top_k]
'''测试代码'''
#if __name__ == "__main__":
    # my_loader = loader.Loader()
    # pdf_path = "F:/my_RAG/rag_demo/data/bianselong.pdf"
    # all_chunks = my_loader.read_pdf(pdf_path)
    # print(len(all_chunks))
    # bge_path = "F:/my_RAG/rag_demo/model/bge-small-zh-v1.5"
    # my_embedding = embedding.Embedding(bge_path=bge_path)
    # pdf_name = "bianselong"
    # my_embedding.bge_embedding(all_chunks=all_chunks,collection_name=pdf_name)
    # my_retriever = retriever.Retriever(bge_path=bge_path)
    # query = "谁的手指被咬了？"
    # res = my_retriever.recall(query=query,top_k=5,collection_name=pdf_name)
    # for re in res:
    #     print("======================================================================")
    #     print("content:",re["content"])
    #     print("metadata:",re["metadata"])
    #     print("distance:",re["distance"])
    # my_reranker = Reranker(bge_path=bge_path)
    # res = my_reranker.rerank(query=query,retrieved_lists=res,top_k=3)
    # print(len(res))
    # for re in res:
    #     print("======================================================================")
    #     print("content:",re["content"])
    #     print("metadata:",re["metadata"])
    #     print("distance:",re["distance"])
    #     print("rerank_score",re["rerank_score"])
