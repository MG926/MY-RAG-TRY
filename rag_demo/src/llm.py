#MG314
"""该模块用于调用 DeepSeek API 生成答案"""
import os
from openai import OpenAI

class LLM:
    def __init__(self, api_key: str = "", base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", temperature: float = 0.1, max_tokens: int = 2048):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("未提供 API Key，请设置环境变量 DEEPSEEK_API_KEY 或传入 api_key 参数")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def chat(self, messages: list[dict]) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"调用 LLM 失败: {e}"

    def generate_answer(self, question: str, chunks: list[dict]) -> str:
        context_parts = []
        for idx, item in enumerate(chunks, start=1):
            page = item.get("metadata", {}).get("page_num", "?")
            context_parts.append(f"[{idx}] (第{page}页) {item.get('content', '')}")
        context = "\n".join(context_parts)
        context = context[:3000]

        messages = [
            {"role": "system",
             "content": "你是一个严谨的知识库问答助手。只依据提供的资料回答，"
                        "资料中没有的内容，明确回答\"资料中未找到\"，不要编造。"},
            {"role": "user",
             "content": f"相关资料：\n{context}\n\n问题：{question}"}
        ]
        return self.chat(messages)
