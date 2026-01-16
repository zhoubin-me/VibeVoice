#!/usr/bin/env python3
"""
Stream text from a local OpenAI-compatible server to VibeVoice WebSocket TTS,
receive streamed PCM16 audio, and play locally.

Requirements:
  pip install openai websockets sounddevice
"""

import argparse
import asyncio
import json
import queue
import sys
import threading
from typing import Iterable, Iterator

import numpy as np
import sounddevice as sd
import websockets
from openai import OpenAI

SAMPLE_RATE = 24000
DEFAULT_BASE_URL = "http://192.168.1.3:8901/v1"
DEFAULT_API_KEY = "local"
DEFAULT_TEXT_MODEL = None
DEFAULT_WS_URL = "ws://192.168.1.3:3000/stream-text"


def stream_text(client: OpenAI, prompt: str, model: str) -> Iterator[str]:
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer concisely for text-to-speech."},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def chunk_text(
    tokens: Iterable[str], min_chars: int = 60, max_chars: int = 200
) -> Iterator[str]:
    buffer = ""
    for token in tokens:
        buffer += token
        if len(buffer) >= min_chars and buffer.endswith((".", "?", "!", "\n")):
            yield buffer.strip()
            buffer = ""
        elif len(buffer) >= max_chars:
            split_idx = buffer.rfind(" ")
            if split_idx <= 0:
                split_idx = len(buffer)
            yield buffer[:split_idx].strip()
            buffer = buffer[split_idx:].lstrip()
    if buffer.strip():
        yield buffer.strip()


def resample_chunk(chunk: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    if in_rate == out_rate or chunk.size == 0:
        return chunk
    ratio = out_rate / in_rate
    new_len = max(1, int(round(chunk.size * ratio)))
    x_old = np.linspace(0, 1, num=chunk.size, endpoint=False)
    x_new = np.linspace(0, 1, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, chunk).astype(np.float32, copy=False)


def audio_player(
    audio_queue: "queue.Queue[bytes | None]",
    stop_event: threading.Event,
    input_rate: int,
    output_rate: int,
    output_device: int | None,
) -> None:
    with sd.OutputStream(
        samplerate=output_rate,
        channels=1,
        dtype="float32",
        device=output_device,
    ) as stream:
        while not stop_event.is_set() or not audio_queue.empty():
            try:
                payload = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if payload is None:
                break
            chunk = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
            chunk = resample_chunk(chunk, input_rate, output_rate)
            stream.write(chunk)


def produce_text_chunks(
    client: OpenAI,
    prompt: str,
    model: str,
    out_queue: "asyncio.Queue[str | None]",
    loop: asyncio.AbstractEventLoop,
) -> None:
    try:
        tokens = stream_text(client, prompt, model)
        for text_chunk in chunk_text(tokens):
            loop.call_soon_threadsafe(out_queue.put_nowait, text_chunk)
    finally:
        loop.call_soon_threadsafe(out_queue.put_nowait, None)


async def send_text_stream(
    ws: websockets.WebSocketClientProtocol,
    out_queue: "asyncio.Queue[str | None]",
) -> None:
    while True:
        text_chunk = await out_queue.get()
        if text_chunk is None:
            break
        await ws.send(json.dumps({"type": "text", "data": text_chunk}))
    await ws.send(json.dumps({"type": "end"}))


async def receive_audio_stream(
    ws: websockets.WebSocketClientProtocol,
    audio_queue: "queue.Queue[bytes | None]",
) -> None:
    async for message in ws:
        if isinstance(message, (bytes, bytearray)):
            audio_queue.put(bytes(message))


async def run_async(
    prompt: str,
    text_model: str,
    base_url: str,
    api_key: str,
    ws_url: str,
    cfg_scale: float,
    inference_steps: int | None,
    voice: str | None,
    output_device: int | None,
    output_sample_rate: int | None,
) -> None:
    client = OpenAI(base_url=base_url, api_key=api_key)
    params = []
    if cfg_scale is not None:
        params.append(f"cfg={cfg_scale:.3f}")
    if inference_steps is not None:
        params.append(f"steps={inference_steps}")
    if voice:
        params.append(f"voice={voice}")
    ws_full_url = ws_url + ("?" + "&".join(params) if params else "")

    device_info = (
        sd.query_devices(output_device, "output") if output_device is not None else None
    )
    device_rate = int(device_info["default_samplerate"]) if device_info else SAMPLE_RATE
    output_rate = output_sample_rate or device_rate

    audio_queue: "queue.Queue[bytes | None]" = queue.Queue()
    stop_event = threading.Event()
    player_thread = threading.Thread(
        target=audio_player,
        args=(audio_queue, stop_event, SAMPLE_RATE, output_rate, output_device),
        daemon=True,
    )
    player_thread.start()

    try:
        async with websockets.connect(ws_full_url, max_size=None) as ws:
            text_queue: "asyncio.Queue[str | None]" = asyncio.Queue()
            loop = asyncio.get_running_loop()
            producer_thread = threading.Thread(
                target=produce_text_chunks,
                args=(client, prompt, text_model, text_queue, loop),
                daemon=True,
            )
            producer_thread.start()

            send_task = asyncio.create_task(send_text_stream(ws, text_queue))
            recv_task = asyncio.create_task(receive_audio_stream(ws, audio_queue))
            await send_task
            await recv_task
            producer_thread.join()
    finally:
        stop_event.set()
        audio_queue.put(None)
        player_thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream OpenAI text to VibeVoice WS TTS."
    )
    parser.add_argument(
        "--prompt", required=True, help="Prompt to generate spoken response."
    )
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--output-device", type=int, default=3)
    parser.add_argument("--output-sample-rate", type=int, default=None)
    args = parser.parse_args()

    try:
        asyncio.run(
            run_async(
                prompt=args.prompt,
                text_model=args.text_model,
                base_url=args.base_url,
                api_key=args.api_key,
                ws_url=args.ws_url,
                cfg_scale=args.cfg_scale,
                inference_steps=args.inference_steps,
                voice=args.voice,
                output_device=args.output_device,
                output_sample_rate=args.output_sample_rate,
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)


if __name__ == "__main__":
    main()
