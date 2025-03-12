# agent.py
import asyncio
import json
import time
import uuid

import numpy as np
import psutil
from PIL import Image
from livekit import agents, rtc
from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
)

from alibabacloud.stt import STT
from inference_job import EventType, InferenceJob
from state_manager import StateManager

SIP_INTRO = ""
PROMPT = ""
INTRO = ""

# 唤醒词列表
WAKE_WORDS = ["小六"]


async def entrypoint(job_context: JobContext):
    # LiveKit Entities
    source = rtc.AudioSource(24000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    # Plugins
    stt = STT()
    stt_stream = None

    # Agent state
    state = StateManager(job_context.room, PROMPT)  # 初始化 StateManager

    # Inference task and related variables
    inference_task: asyncio.Task | None = None
    current_transcription = ""
    inference_job = None

    # Screenshot variables (if needed)
    screenshot_counter = 0
    last_screenshot_time = time.time()

    # Audio and Video stream flags and futures
    audio_stream_active = False
    latest_video_frame_event: rtc.VideoFrameEvent | None = None
    audio_stream_future = asyncio.Future[rtc.AudioStream]()
    video_stream_future = asyncio.Future[rtc.VideoStream]()
    # 使用 asyncio.Lock 来保护共享资源
    transcription_lock = asyncio.Lock()
    # 标记是否存在视频流/音频流
    has_video_stream = False
    has_audio_stream = False
    # 语音唤醒相关变量
    agent_awake = False
    wake_word_timer: asyncio.TimerHandle | None = None

    def reset_wake_word_timer():
        """重置唤醒词倒计时"""
        nonlocal agent_awake, wake_word_timer
        if wake_word_timer:
            wake_word_timer.cancel()
        wake_word_timer = asyncio.get_event_loop().call_later(30, set_agent_sleep)

    def set_agent_sleep():
        """设置 Agent 进入休眠状态"""
        nonlocal agent_awake
        agent_awake = False
        print("------------------进入休眠状态-------------------")
        if wake_word_timer:  # 取消可能存在的计时器
            wake_word_timer.cancel()

    async def start_new_inference(force_text: str | None = None):
        nonlocal current_transcription, inference_job, inference_task, has_video_stream

        # Cancel any existing inference task
        if inference_task:
            inference_task.cancel()
            try:
                await inference_task
            except asyncio.CancelledError:
                pass

        file_name = None
        if has_video_stream and latest_video_frame_event:  # 使用最新视频帧
            file_name = f"{uuid.uuid4()}.png"
            try:
                # 截图使用最新视频帧
                print("开始截图")
                await take_screenshot(latest_video_frame_event, file_name)
            except Exception as e:
                print(f"Screenshot failed: {e}")
        state.agent_thinking = True
        async with transcription_lock:
            inference_job = InferenceJob(
                transcription=current_transcription,
                audio_source=source,
                chat_history=state.chat_history,
                force_text_response=force_text,
                img_url=file_name,
                has_video_stream=has_video_stream,
                # has_audio_stream=has_audio_stream
            )
        inference_task = asyncio.create_task(_run_inference_job(inference_job))

    async def _run_inference_job(inference_job: InferenceJob):
        """Helper function to run the inference job and handle events."""
        nonlocal current_transcription
        try:
            agent_done_thinking = False
            agent_has_spoken = False
            committed_agent = False

            def commit_agent_text_if_needed():
                nonlocal agent_has_spoken, agent_done_thinking, committed_agent
                if agent_done_thinking and agent_has_spoken and not committed_agent:
                    committed_agent = True
                    state.commit_agent_response(inference_job.current_response)

            async for e in inference_job:
                if e.type == EventType.AGENT_RESPONSE:
                    if e.finished_generating:
                        state.agent_thinking = False
                        agent_done_thinking = True
                        commit_agent_text_if_needed()
                elif e.type == EventType.AGENT_SPEAKING:
                    state.agent_speaking = e.speaking
                    if e.speaking:
                        agent_has_spoken = True
                        if not inference_job.transcription == "":
                            state.commit_user_transcription(inference_job.transcription)
                        commit_agent_text_if_needed()
                        async with transcription_lock:
                            current_transcription = ""
        except asyncio.CancelledError:
            print("Inference job cancelled")
            if inference_job:
                await inference_job.acancel()

    def on_track_subscribed(track: rtc.Track, *_):
        nonlocal has_video_stream, has_audio_stream
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            has_audio_stream = True
            audio_stream_future.set_result(rtc.AudioStream(track))
            print("-----------音频流已订阅")
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            has_video_stream = True  # Set video stream flag to True
            video_stream_future.set_result(rtc.VideoStream(track))
            print("-----------视频流已订阅")

    def on_data(dp: rtc.DataPacket):
        async def handle_data():
            nonlocal current_transcription, inference_job

            if dp.topic != "lk-chat-topic":
                return
            try:
                payload = json.loads(dp.data.decode('utf-8'))
                message = payload["message"]
                async with transcription_lock:
                    current_transcription = message
                if inference_job:
                    await inference_job.interrupt_tts()
                await start_new_inference()
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error processing data1: {e}")
            except Exception as e:
                print(f"Error processing data2: {e}")

        asyncio.create_task(handle_data())

    # 断开连接处理
    def on_disconnected():
        print("正在断开连接...")

        async def _disconnect_task():
            nonlocal inference_task  # 如果在内部函数中修改外部变量，需要 nonlocal
            if inference_task:
                inference_task.cancel()
                try:
                    await inference_task  # 等待任务完成
                except asyncio.CancelledError:
                    pass
            await job_context.room.disconnect()
            print("Agent disconnected.")

        asyncio.create_task(_disconnect_task())  # 创建并运行异步任务

    job_context.room.on("disconnected", on_disconnected)
    # Subscribe to existing tracks
    for participant in job_context.room.participants.values():
        for track_pub in participant.tracks.values():
            if track_pub.track is None:
                continue
            if track_pub.kind == rtc.TrackKind.KIND_AUDIO:
                audio_stream_future.set_result(rtc.AudioStream(track_pub.track))
            elif track_pub.kind == rtc.TrackKind.KIND_VIDEO:
                has_video_stream = True  # Set video stream flag to True
                video_stream_future.set_result(rtc.VideoStream(track_pub.track))

    # Publish agent mic after waiting for user audio
    await job_context.room.local_participant.publish_track(track, options)

    # 封装的截图方法 (if needed, keep this function)
    async def take_screenshot(video_frame_event: rtc.VideoFrameEvent, screenshot_filename: str) -> str:
        """从视频流中截取图片并保存为文件"""
        nonlocal screenshot_counter
        frame = video_frame_event.frame
        width = frame.width
        height = frame.height
        yuv_data = frame.data

        # Convert YUV to RGB
        y_plane = yuv_data[:width * height]
        u_plane = yuv_data[width * height: width * height + (width // 2) * (height // 2)]
        v_plane = yuv_data[width * height + (width // 2) * (height // 2):]

        # Reshape planes
        y_plane = np.frombuffer(y_plane, dtype=np.uint8).reshape((height, width))
        u_plane = np.frombuffer(u_plane, dtype=np.uint8).reshape((height // 2, width // 2))
        v_plane = np.frombuffer(v_plane, dtype=np.uint8).reshape((height // 2, width // 2))

        # Upsample U and V planes
        u_plane = u_plane.repeat(2, axis=0).repeat(2, axis=1)
        v_plane = v_plane.repeat(2, axis=0).repeat(2, axis=1)

        # Combine YUV planes
        yuv = np.dstack((y_plane, u_plane, v_plane)).astype(np.uint8)
        rgb_image = Image.fromarray(yuv, mode="YCbCr").convert("RGB")

        # 保存图片,这里的路径需要被外网url映射到,url在llm_vl.py中修改
        rgb_image.save(f"D:/comfyui_api/output/temp/{screenshot_filename}")  # 更改为你实际的保存路径
        print(f"Screenshot saved as {screenshot_filename}")
        screenshot_counter += 1
        return screenshot_filename

    # Task to handle audio stream (runs concurrently)
    async def audio_stream_task():
        nonlocal audio_stream_active, has_audio_stream, audio_stream
        audio_stream_active = True
        try:
            if audio_stream is None:
                print("audio_stream_task 启动，但 audio_stream 为 None。退出。")
                return
            async for audio_frame_event in audio_stream:
                stt_stream.push_frame(audio_frame_event.frame)
        except Exception as e:
            has_audio_stream = False
            print(f"解析音频流异常: {e}")
        finally:
            audio_stream_active = False

    # Task to process STT stream (runs concurrently)
    async def stt_stream_task():
        nonlocal current_transcription, inference_task, inference_job, agent_awake

        try:
            async for stt_event in stt_stream:
                if stt_event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT:
                    text = stt_event.alternatives[0].text
                    if text == "":
                        continue
                    if not agent_awake:
                        # 语音唤醒检测
                        for word in WAKE_WORDS:
                            if word in text.lower():
                                agent_awake = True
                                reset_wake_word_timer()
                                break
                    if not agent_awake:
                        print("接收到对话:{},没有匹配唤醒词", text)
                        continue
                    ## 打断上段对话
                    if inference_job:
                        try:
                            await inference_job.interrupt_tts()
                        except Exception as e:
                            print("TTS打断失败: %s", e)

                    async with transcription_lock:
                        current_transcription += text
                        print("接收到对话:{},开始对话", text)
                    if agent_awake:
                        reset_wake_word_timer()
                        await start_new_inference()
        except Exception as e:  # 捕获STT可能出现的错误
            print(f"STT 任务异常: {e}")

    # Task to handle video stream (runs concurrently)
    async def video_stream_task():
        """实时处理视频流并更新最新帧"""
        nonlocal latest_video_frame_event, has_video_stream
        try:
            async for video_frame_event in video_stream:
                # 实时更新最新视频帧
                latest_video_frame_event = video_frame_event
        except Exception as e:
            has_video_stream = False
            print(f"解析视频流异常: {e}")

    # Add event listeners
    job_context.room.on("track_subscribed", on_track_subscribed)
    job_context.room.on("data_received", on_data)
    job_context.room.on("disconnected", on_disconnected)
    try:
        audio_stream = await asyncio.wait_for(audio_stream_future, timeout=3.0)
        print("-----------检测到音频流")
        has_audio_stream = True
    except asyncio.TimeoutError:
        print("-----------没有检测到音频流")
        has_audio_stream = False

    try:
        video_stream = await asyncio.wait_for(video_stream_future, timeout=3.0)
        print("-----------检测到视频流")
        has_video_stream = True
    except asyncio.TimeoutError:
        print("-----------没有检测到视频流")
        has_video_stream = False

    if has_audio_stream:
        _ = asyncio.create_task(audio_stream_task())
        stt_stream = stt.stream()
        _ = asyncio.create_task(stt_stream_task())
        print("-----------音频任务创建")

    if has_video_stream:
        _ = asyncio.create_task(video_stream_task())
        print("-----------视频任务创建")

    # Keep the agent alive (replace job_context.room.run())
    try:
        while True:
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        print("Agent正在关闭...")
    finally:
        if inference_task:
            inference_task.cancel()
            try:
                await inference_task
            except asyncio.CancelledError:
                pass
        await job_context.room.disconnect()
        print("Agent断开连接.")


## 0.2 当有新的通话或交互需要 Agent 处理时，LiveKit 服务器会向 Worker 发送一个 JobRequest。request_fnc 函数被触发，处理这个请求。指定 entrypoint 函数作为处理该 Job 的入口点。
async def request_fnc(req: JobRequest) -> None:
    await req.accept(entrypoint, auto_subscribe=agents.AutoSubscribe.SUBSCRIBE_ALL)


def cpu_load_fnc() -> float:
    return 0.0

## 0.1 当通过 cli.run_app 启动 Worker 时，它会监听指定的端口,Worker 准备好接收来自 LiveKit 服务器的 Job 请求。
if __name__ == "__main__":
    cli.run_app(WorkerOptions(load_fnc=cpu_load_fnc, load_threshold=0.8, request_fnc=request_fnc, port=8080))
