#!/usr/bin/env python3
"""
Stream text from a local OpenAI-compatible server, synthesize with VibeVoice, and play locally.

Requirements:
  pip install openai sounddevice
"""

import argparse
import queue
import sys
import threading
from typing import Iterable, Iterator

import numpy as np
import sounddevice as sd
from openai import APIConnectionError, OpenAI
import httpx

from demo.web.app import StreamingTTSService

SAMPLE_RATE=24000

DEFAULT_TEXT_MODEL = None
DEFAULT_MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"
DEFAULT_DEVICE = "cuda"
DEFAULT_BASE_URL = "http://192.168.1.3:8901/v1"
DEFAULT_API_KEY = "local"


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


def chunk_text(tokens: Iterable[str], min_chars: int = 60, max_chars: int = 200) -> Iterator[str]:
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
    audio_queue: "queue.Queue[np.ndarray | None]",
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
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                break
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32, copy=False)
            chunk = resample_chunk(chunk, input_rate, output_rate)
            stream.write(chunk)


def run(
    prompt: str,
    text_model: str,
    base_url: str,
    api_key: str,
    model_path: str,
    device: str,
    voice: str | None,
    cfg_scale: float,
    inference_steps: int | None,
    do_sample: bool,
    temperature: float,
    top_p: float,
    refresh_negative: bool,
    output_device: int | None,
    output_sample_rate: int | None,
) -> None:
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        models_url = base_url.rstrip("/") + "/models"
        httpx.get(models_url, timeout=5.0)
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Failed to reach OpenAI-compatible server at {base_url}. "
            f"Is it running? ({exc})"
        ) from exc

    tts = StreamingTTSService(model_path=model_path, device=device)
    tts.load()

    input_rate = tts.sample_rate if hasattr(tts, "sample_rate") else SAMPLE_RATE
    device_info = sd.query_devices(output_device, "output") if output_device is not None else None
    device_rate = int(device_info["default_samplerate"]) if device_info else input_rate
    output_rate = output_sample_rate or device_rate

    audio_queue: "queue.Queue[np.ndarray | None]" = queue.Queue()
    stop_event = threading.Event()
    player_thread = threading.Thread(
        target=audio_player,
        args=(audio_queue, stop_event, input_rate, output_rate, output_device),
        daemon=True,
    )
    player_thread.start()

    try:
        try:
            tokens = stream_text(client, prompt, text_model)
            for text_chunk in chunk_text(tokens):
                print(text_chunk, end="", flush=True)
                for audio_chunk in tts.stream(
                    text_chunk,
                    cfg_scale=cfg_scale,
                    inference_steps=inference_steps,
                    voice_key=voice,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    refresh_negative=refresh_negative,
                ):
                    audio_queue.put(audio_chunk)
        except APIConnectionError as exc:
            raise RuntimeError(
                f"OpenAI-compatible server at {base_url} refused the connection. "
                "Start it and retry."
            ) from exc
        print()
    finally:
        stop_event.set()
        audio_queue.put(None)
        player_thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream local OpenAI text + VibeVoice streaming TTS playback."
    )
    parser.add_argument("--prompt", required=True, help="Prompt to generate spoken response.")
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "cuda", "mpx", "mps"])
    parser.add_argument("--voice", default=None, help="Voice preset key from demo/voices/streaming_model.")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-refresh-negative", dest="refresh_negative", action="store_false")
    parser.add_argument("--output-device", type=int, default=3)
    parser.add_argument("--output-sample-rate", type=int, default=None)
    parser.set_defaults(refresh_negative=True)
    args = parser.parse_args()

    try:
        run(
            prompt=args.prompt,
            text_model=args.text_model,
            base_url=args.base_url,
            api_key=args.api_key,
            model_path=args.model_path,
            device=args.device,
            voice=args.voice,
            cfg_scale=args.cfg_scale,
            inference_steps=args.inference_steps,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            refresh_negative=args.refresh_negative,
            output_device=args.output_device,
            output_sample_rate=args.output_sample_rate,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)


if __name__ == "__main__":
    main()
