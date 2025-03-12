import asyncio
import contextlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import AsyncIterable, Optional

from dashscope.audio.tts_v2 import *
from livekit import rtc
from livekit.agents import tts

STREAM_EOS = "EOS"


@dataclass
class TTSOptions:
    api_key: str
    model: str
    voice: str
    # 合成音频的语速，取值范围：0.5~2。
    speech_rate: int
    # 合成音频的音量，取值范围：0~100。
    volume: int


class Callback(ResultCallback):
    def __init__(self, _tts: tts.SynthesizeStream):
        self._tts = _tts

    def on_open(self):
        print("语音输出TTS开始...")

    def on_complete(self):
        print("语音输出TTS输出完成")

    def on_error(self, message: str):
        print(f"语音输出TTS任务失败, {message}")
        self._tts._queue.task_done()

    def on_close(self):
        print('语音输出TTS已关闭')
        try:
            self._tts._queue.task_done()
        except ValueError:
            pass

    def on_data(self, data: bytes) -> None:
        print('------------语音输出-----------')
        if data is not None:
            audio_frame = rtc.AudioFrame(
                data=data,
                sample_rate=24000,
                num_channels=1,
                samples_per_channel=len(data) // 2,
            )

            self._tts._event_queue.put_nowait(
                tts.SynthesisEvent(
                    type=tts.SynthesisEventType.AUDIO,
                    audio=tts.SynthesizedAudio(text="", data=audio_frame),
                )
            )



class TTS(tts.TTS):
    def __init__(
            self,
            *,
            api_key: Optional[str] = '',
            sample_rate: int = 24000,
            num_channels: int = 1,
            latency: int = 2,
            voice: str = "longcheng_v2",
            model: str = "cosyvoice-v2",
            speech_rate: int = 1,
            volume: int = 100,
    ) -> None:
        super().__init__(streaming_supported=True, sample_rate=sample_rate, num_channels=num_channels)
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY must be set")

        self._config = TTSOptions(
            model=model,
            api_key=api_key,
            voice=voice,
            speech_rate=speech_rate,
            volume=volume
        )

    def synthesize(
            self,
            text: str,
    ) -> AsyncIterable[tts.SynthesizedAudio]:
        async def generator():
            pass

        return generator()

    def stream(
            self,
    ) -> "SynthesizeStream":
        return SynthesizeStream(self._config)


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
            self,
            config: TTSOptions,
    ):
        self._config = config
        self._executor = ThreadPoolExecutor()
        self._queue = asyncio.Queue[str]()
        self._tts_queue = asyncio.Queue[str]()
        self._event_queue = asyncio.Queue[tts.SynthesisEvent]()
        self._closed = False

        self._main_task = asyncio.create_task(self._run())

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                print(f"dashscope synthesis task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)
        self._text = ""

    def push_text(self, token: str | None) -> None:
        if self._closed:
            raise ValueError("cannot push to a closed stream")

        if not token or len(token) == 0:
            return
        self._text += token

    async def _run(self) -> None:

        callback = Callback(self)
        started = False
        model = self._config.model
        voice = self._config.voice
        speech_rate = self._config.speech_rate
        volume = self._config.volume
        synthesizer = SpeechSynthesizer(
            model=model,
            voice=voice,
            format=AudioFormat.PCM_22050HZ_MONO_16BIT,
            speech_rate=speech_rate,
            volume=volume,
            callback=callback
        )
        while True:
            text = None
            try:
                text = await self._queue.get()
                tts_str = str(text).strip()
                print(f"语音输出TTS: {tts_str}")
                if not started:
                    self._event_queue.put_nowait(
                        tts.SynthesisEvent(type=tts.SynthesisEventType.STARTED)
                    )
                    started = True
                if (tts_str != STREAM_EOS and tts_str != ""):
                    synthesizer.streaming_call(text)
                if tts_str == STREAM_EOS:
                    synthesizer.streaming_complete()
                    self._event_queue.put_nowait(
                        tts.SynthesisEvent(type=tts.SynthesisEventType.FINISHED)
                    )
                self._queue.task_done()
            except asyncio.CancelledError:
                print("TTS已经被中断")
                break
            except Exception as e:
                print(e)
                break

    async def end(self) -> None:
        self._queue.put_nowait(STREAM_EOS)

    async def flush(self) -> None:
        _text = self._text
        self._queue.put_nowait(_text)
        self._text = ""

    async def aclose(self) -> None:
        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def __anext__(self) -> tts.SynthesisEvent:
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()
