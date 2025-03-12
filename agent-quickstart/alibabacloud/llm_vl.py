import asyncio
import codecs
import json
import os
from typing import MutableSet

import aiohttp
from attrs import define
from livekit.agents import llm
from livekit.agents.llm import ChatContext, ChatMessage, ChatRole


# from .models import ChatModels


@define
class LLMVLOptions:
    model: str
    base_url: str
    api_key: str


class LLMVL(llm.LLM):
    def __init__(
            self,
            *,
            model: str = "qwen-vl-max",
            api_key: str = "",
            base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    ) -> None:
        api_key = api_key or os.environ.get("LLM_API_KEY")
        base_url = base_url or os.environ.get("LLM_BASE_URL")
        if not api_key or not base_url:
            dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not dashscope_api_key:
                raise ValueError("both DASHSCOPE_API_KEY or other  llm base_url and api_key must be set")
            else:
                self._opts = LLMVLOptions(model=model, base_url="xxxx", api_key=dashscope_api_key)
        else:
            self._opts = LLMVLOptions(model=model, base_url=base_url, api_key=api_key)
        self._running_fncs: MutableSet[asyncio.Task] = set()

    async def fetch_stream(self, url, headers, data):
        print("===============正在调用qwen vl api===============")
        # print(data)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=json.dumps(data), timeout=60000) as resp:
                lines = []
                async for block in resp.content.iter_any():
                    lines = block.decode('utf-8').splitlines(keepends=False)
                    for line in lines:
                        if line:
                            yield line.encode('utf-8')

    async def chat(
            self,
            history: llm.ChatContext,
            fnc_ctx: llm.FunctionContext | None = None,
            temperature: float | None = None,
            n: int | None = None,
            img: str = ""
    ) -> "LLMVLStream":
        llm_config = self._opts
        url = llm_config.base_url
        headers = {
            'Authorization': f"Bearer {llm_config.api_key}",
            'Content-Type': 'application/json'
        }

        req_data = {
            "messages": to_openai_ctx(history, img),
            "stream": True,
            # "max_tokens": 50,
            "model": "qwen-vl-max"
        }
        async_gen = self.fetch_stream(url, headers, req_data)
        return LLMVLStream(async_gen, fnc_ctx)


class LLMVLStream(llm.LLMStream):
    def __init__(
            self, oai_stream, fnc_ctx: llm.FunctionContext | None
    ) -> None:
        super().__init__()
        self._oai_stream = oai_stream
        self._fnc_ctx = fnc_ctx
        self._running_fncs: MutableSet[asyncio.Task] = set()

    def __aiter__(self) -> "LLMVLStream":
        return self

    async def __anext__(self) -> llm.ChatChunk:
        # answer = ""
        async for line in self._oai_stream:
            # print(str(line))
            if line:
                line = codecs.decode(line)
                # print(str(line))
                if line.startswith("data:"):
                    line = line[5:].strip()
                    try:
                        chunk = json.loads(line)
                        # print(str(chunk))
                        if "choices" in chunk and len(
                                chunk["choices"]) > 0 and "delta" in chunk["choices"][0] and "content" in \
                                chunk["choices"][0]["delta"]:
                            delta = chunk["choices"][0]["delta"]
                            # answer += delta["content"]
                            # print(answer)
                            # print(delta["content"])
                            return llm.ChatChunk(
                                choices=[
                                    llm.Choice(
                                        delta=llm.ChoiceDelta(
                                            content=delta["content"],
                                            role='assistant',
                                        ),
                                        index=0,
                                    )
                                ]
                            )
                    except json.JSONDecodeError as err:
                        # print(f"error : {err}")
                        raise StopAsyncIteration
            else:
                raise StopAsyncIteration
        raise StopAsyncIteration

    async def aclose(self, wait: bool = True) -> None:
        if not wait:
            for task in self._running_fncs:
                task.cancel()

        await asyncio.gather(*self._running_fncs, return_exceptions=True)


def to_openai_ctx(chat_ctx: llm.ChatContext, img) -> list:
    content = '''
    # 角色
    你是小六，是由六六六研发的智能助手，能够借助用户的摄像头和麦克风与用户进行聊天、解答问题，还能充当用户的眼睛，解析用户发送来的画面信息。

    ## 技能
    ### 技能 1: 正确理解用户的问题
    用户的问题由语音转文字而来,可能存在错字别字,请你正确理解相关问题
    ### 技能 2: 回答用户的问题
    1. 你的回答应当简洁明了,避免过分冗长,尽量以50字内的回答回复用户。
    2. 当用户的问题和图片没有关联时，忽略图片，只回答用户的问题。
    ### 技能 3: 理解用户传来的画面
    1. 当用户的问题和画面有关联时，用户认为他在借助摄像头同你交流,解析并回答你看到了什么。
    2. 严格禁止回答中出现"图片中","照片中"等信息,使用"我看到了","画面中的是"替代
    ### 技能 4: 当用户叫你名字"小六"进行对话唤醒时,回答"您好，我是小六，请问有什么可以帮助您的？"
    
    ## 限制:
    - **重要提示1**当用户的问题和画面没有关联时，忽略图片，只需要回答用户的问题,不要回答任何画面的信息。
    - **重要提示2**用户认为他在借助摄像头同你交流,严格禁止回答中出现"图片中","照片中"等信息,使用"我看到了","画面中的是"替代
    - 回答应简洁明了，避免过于冗长。
    - 严格按照上述要求处理图文相关问题。
    - 当用户询问你的身份信息等敏感问题时,回答你的名字"我是小六，使用六六六全新一代AI视觉大模型，欢迎随时找我聊天"或者""对不起,我还不太了解""
    '''
    if img is None or img == "":
        img = "b2d4b5f9-ef2c-482e-a3f3-76eff7bb00f9.png"
    messages = [
        {'role': 'system', 'content': content}
    ]
    for msg in chat_ctx.messages:
        messages.append(
            {
                "role": 'user',
                "content": [
                    {"type": "text", "text": msg.text},
                    {"type": "image_url",
                     "image_url": {"url": f"https://test.com/ai_test_api/output/temp/{img}"}
                     }
                ]
            }
        )
    return messages

# def image_to_base64(image_path):
#     try:
#         with open(image_path, "rb") as image_file:
#             encoded_string = base64.b64encode(image_file.read()).decode('utf-8')  # Important: decode to string
#             return encoded_string
#     except FileNotFoundError:
#         print(f"Error: Image file not found at {image_path}")
#         return None
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None
#
#
# async def main():
#     try:
#         chat_context = ChatContext(
#             messages=[ChatMessage(role=ChatRole.USER, text="你好,你是谁")]
#         )
#         llm_chat = await LLMVL().chat(history=chat_context)
#         async for chunk in llm_chat:
#             delta = chunk.choices[0].delta.content
#             print(chunk)
#             print(delta)
#     except asyncio.runtime.Warning as warning:
#         print(f"Warning: {warning}")
#
#
# asyncio.run(main())
