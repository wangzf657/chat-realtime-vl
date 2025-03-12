# inference_job.py
from __future__ import annotations

import asyncio
import logging
import uuid
from enum import Enum
from typing import List

from attr import define
from livekit import agents, rtc
from livekit.agents.llm import ChatContext, ChatMessage, ChatRole

from alibabacloud.llm import LLM
from alibabacloud.llm_vl import LLMVL
from alibabacloud.tts_new import TTS


class InferenceJob:
    def __init__(
            self,
            transcription: str,
            audio_source: rtc.AudioSource,
            chat_history: List[ChatMessage],
            force_text_response: str | None = None,
            img_url: str | None = None,
            has_video_stream: bool = True,
    ):
        self._id = uuid.uuid4()
        self._audio_source = audio_source
        self._transcription = transcription
        self._current_response = ""
        self._chat_history = chat_history
        self._tts = TTS()
        self._tts_stream = self._tts.stream()
        self._llm = LLM()
        if has_video_stream:
            self._llm = LLMVL()
        self._run_task = asyncio.create_task(self._run())
        self._output_queue = asyncio.Queue[rtc.AudioFrame | None]()
        self._speaking = False
        self._finished_generating = False
        self._event_queue = asyncio.Queue[Event | None]()
        self._done_future = asyncio.Future()
        self._cancelled = False
        self._force_text_response = force_text_response
        self.img_url = img_url

    @property
    def id(self):
        return self._id

    @property
    def transcription(self):
        return self._transcription

    @property
    def current_response(self):
        return self._current_response

    @current_response.setter
    def current_response(self, value: str):
        self._current_response = value
        if not self._cancelled:
            self._event_queue.put_nowait(
                Event(
                    type=EventType.AGENT_RESPONSE,
                    finished_generating=self.finished_generating,
                    speaking=self.speaking,
                )
            )

    @property
    def finished_generating(self):
        return self._finished_generating

    @finished_generating.setter
    def finished_generating(self, value: bool):
        self._finished_generating = value
        if not self._cancelled:
            self._event_queue.put_nowait(
                Event(
                    finished_generating=value,
                    type=EventType.AGENT_RESPONSE,
                    speaking=self.speaking,
                )
            )

    async def acancel(self):
        print("Cancelling inference job")
        self._cancelled = True
        # 取消子任务
        if hasattr(self, '_llm_task_instance') and self._llm_task_instance:
            self._llm_task_instance.cancel()
        if hasattr(self, '_tts_task_instance') and self._tts_task_instance:
            self._tts_task_instance.cancel()
        if hasattr(self, '_audio_capture_task_instance') and self._audio_capture_task_instance:
            self._audio_capture_task_instance.cancel()

        self._run_task.cancel()
        try:
            await self._run_task  # 等待主任务完成
        except asyncio.CancelledError:
            pass
        await self._done_future
        print("Cancelled inference job")

    @property
    def speaking(self):
        return self._speaking

    @speaking.setter
    def speaking(self, value: bool):
        if value == self._speaking:
            return
        self._speaking = value
        if not self._cancelled:
            self._event_queue.put_nowait(
                Event(
                    speaking=value,
                    type=EventType.AGENT_SPEAKING,
                    finished_generating=self.finished_generating,
                )
            )

    async def _run(self):
        print(
            "Running inference with user transcription: %s", self.transcription
        )
        try:
            # 创建并屏蔽子任务
            self._llm_task_instance = asyncio.create_task(self._llm_task())
            self._tts_task_instance = asyncio.create_task(self._tts_task())
            self._audio_capture_task_instance = asyncio.create_task(self._audio_capture_task())

            await asyncio.gather(
                self._llm_task_instance,
                self._tts_task_instance,
                self._audio_capture_task_instance,
            )

        except asyncio.CancelledError:
            pass  # 预期在中断时发生
        except Exception as e:
            print("Exception in inference %s", e)
        finally:
            self._done_future.set_result(True)  # 确保任务完成

    async def _llm_task(self):
        try:
            if self._cancelled:  # 在循环开始处检查取消状态
                return
            if self._force_text_response:
                self._tts_stream.push_text(self._force_text_response)
                self.current_response = self._force_text_response
                self.finished_generating = True
                await self._tts_stream.flush()
                return
            chat_context = ChatContext(
                messages=self._chat_history
                         + [ChatMessage(role=ChatRole.USER, text=self.transcription)]
            )
            self.finished_generating = False

            llm_chat = await self._llm.chat(history=chat_context, img=self.img_url)
            async for chunk in llm_chat:
                delta = chunk.choices[0].delta.content
                if delta:
                    self._tts_stream.push_text(delta)
                    self.current_response += delta
                await self._tts_stream.flush()
                await self._tts_stream._queue.join()  # 等待所有文本推送到 TTS 队列
            self.finished_generating = True
            await self._tts_stream.end()
            print('LLM RESPONSE END')
        except asyncio.CancelledError:
            pass  # 预期在中断时发生
        except Exception as e:
            print("_llm_task error: %s", e)

    async def _tts_task(self):
        try:
            async for event in self._tts_stream:
                if self._cancelled:
                    break
                if event.type == agents.tts.SynthesisEventType.AUDIO:
                    await self._output_queue.put(
                        event.audio.data if event.audio else event.audio
                    )
                elif event.type == agents.tts.SynthesisEventType.FINISHED:
                    break
            print('DONE tts')
            self._output_queue.put_nowait(None)  # 发出 TTS 结束信号
        except asyncio.CancelledError:
            pass  # 预期在中断时发生
        except Exception as e:
            print("_tts_task error: %s", e)

    async def interrupt_tts(self):
        print("Interrupting TTS")
        self._cancelled = True
        await self._tts_stream.aclose()  # 立即关闭 TTS 流
        self._output_queue.put_nowait(None)  # 清空输出队列

    async def _audio_capture_task(self):
        try:
            while True:
                if self._cancelled:
                    break
                audio_frame = await self._output_queue.get()
                if audio_frame is None:
                    break  # 当收到 None 时退出
                self.speaking = True
                await self._audio_source.capture_frame(audio_frame)
            self.speaking = False
        except asyncio.CancelledError:
            pass  # 预期在中断时发生
        except Exception as e:
            print("_audio_capture_task error: %s", e)
        finally:
            self._event_queue.put_nowait(None)  # 全部完成

    def __aiter__(self):
        return self

    async def __anext__(self):
        e = await self._event_queue.get()
        if e is None:
            print('DONE job')
            raise StopAsyncIteration
        return e


class EventType(Enum):
    AGENT_RESPONSE = 1
    AGENT_SPEAKING = 2


@define(kw_only=True)
class Event:
    type: EventType
    speaking: bool
    finished_generating: bool
