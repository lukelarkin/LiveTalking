#!/usr/bin/env python3
"""
render_course.py — Turn audio files into talking avatar videos. One command.

Usage:
    python render_course.py ./lessons ./output

    This finds every .wav/.mp3/.m4a file in ./lessons, renders a
    lip-synced talking avatar video for each one, and saves the
    final MP4s to ./output.

Options:
    --avatar_id   Avatar to use (default: auto-detected from data/avatars/)
    --model       musetalk | wav2lip | ultralight (default: musetalk)
    --batch_size  Inference batch size (default: 16)

Examples:
    # Simplest — drop files in a folder, get videos out:
    python render_course.py ./lessons ./output

    # Use a specific avatar and model:
    python render_course.py ./lessons ./output --avatar_id my_face --model wav2lip

    # Render a single file:
    python render_course.py lesson_01.wav ./output
"""
import argparse
import os
import sys
import time
import queue
from pathlib import Path
from threading import Thread, Event

import cv2
import numpy as np
import resampy
import soundfile as sf
import torch
import torch.multiprocessing as mp

from logger import logger

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
CHUNK = 320        # 20 ms at 16 kHz
FPS = 50           # audio fps (fixed by LiveTalking)
VIDEO_FPS = 25

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.opus'}


# ─────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load any audio file → float32 mono at 16 kHz."""
    stream, sr = sf.read(path)
    stream = stream.astype(np.float32)
    if stream.ndim > 1:
        stream = stream[:, 0]
    if sr != SAMPLE_RATE:
        stream = resampy.resample(stream, sr, SAMPLE_RATE)
    return stream


def audio_to_chunks(audio: np.ndarray) -> list:
    """Split audio into 20 ms chunks (320 samples each)."""
    return [audio[i:i + CHUNK] for i in range(0, len(audio) - CHUNK + 1, CHUNK)]


def find_audio_files(path: str) -> list:
    """Find audio files. Accepts a single file or a directory."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
        return [str(p)]
    if p.is_dir():
        files = set()
        for ext in AUDIO_EXTENSIONS:
            files.update(str(f) for f in p.glob(f'*{ext}'))
            files.update(str(f) for f in p.glob(f'*{ext.upper()}'))
        return sorted(files)
    return []


# ─────────────────────────────────────────────────────────────────────
# Offline frame writer (replaces WebRTC / virtualcam)
# ─────────────────────────────────────────────────────────────────────

def offline_process_frames(nerfreal, quit_event, expected_frames, progress_cb=None):
    """
    Pull rendered frames from the inference pipeline and write them
    to the active recording pipes. No WebRTC, no virtualcam — just
    clean frames straight to ffmpeg.
    """
    frames_written = 0

    while not quit_event.is_set():
        try:
            res_frame, idx, audio_frames = nerfreal.res_frame_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue

        # Composite the frame (same logic as basereal.process_frames)
        if audio_frames[0][1] != 0 and audio_frames[1][1] != 0:
            nerfreal.speaking = False
            combine_frame = nerfreal.frame_list_cycle[idx]
        else:
            nerfreal.speaking = True
            try:
                combine_frame = nerfreal.paste_back_frame(res_frame, idx)
            except Exception as e:
                logger.warning(f"paste_back_frame error: {e}")
                continue

        # Write to recording pipes
        nerfreal.record_video_data(combine_frame)
        for af in audio_frames:
            frame_data, _, _ = af
            frame_data = (frame_data * 32767).astype(np.int16)
            nerfreal.record_audio_data(frame_data)

        frames_written += 1
        if progress_cb:
            progress_cb(frames_written, expected_frames)

        if frames_written >= expected_frames:
            quit_event.set()

    logger.info(f'Rendered {frames_written} frames')


# ─────────────────────────────────────────────────────────────────────
# Progress bar
# ─────────────────────────────────────────────────────────────────────

class ProgressBar:
    def __init__(self, label: str):
        self.label = label[:24].ljust(24)
        self.start = time.time()
        self._last_pct = -1

    def update(self, current, total):
        if total <= 0:
            return
        pct = int(100 * current / total)
        if pct == self._last_pct:
            return
        self._last_pct = pct
        elapsed = time.time() - self.start
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = '\u2588' * filled + '\u2591' * (bar_len - filled)
        eta = (elapsed / max(current, 1)) * (total - current)
        sys.stdout.write(f'\r  {self.label}  {bar}  {pct:3d}%  ETA {eta:.0f}s')
        sys.stdout.flush()
        if current >= total:
            elapsed = time.time() - self.start
            sys.stdout.write(f'\r  {self.label}  {bar}  100%  {elapsed:.1f}s\n')
            sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────
# Build a fresh nerfreal instance
# ─────────────────────────────────────────────────────────────────────

def make_opt(avatar_id, model_name, batch_size, session_id=0):
    """Create the options namespace that LiveTalking components expect."""
    class Opt:
        pass
    opt = Opt()
    opt.fps = FPS
    opt.l = 10
    opt.m = 8
    opt.r = 10
    opt.W = 450
    opt.H = 450
    opt.avatar_id = avatar_id
    opt.batch_size = batch_size
    opt.customvideo_config = ''
    opt.customopt = []
    opt.tts = 'edgetts'
    opt.REF_FILE = 'zh-CN-YunxiaNeural'
    opt.REF_TEXT = None
    opt.TTS_SERVER = 'http://127.0.0.1:9880'
    opt.model = model_name
    opt.transport = 'webrtc'
    opt.push_url = ''
    opt.max_session = 1
    opt.listenport = 8010
    opt.sessionid = session_id
    return opt


def build_nerfreal(opt, model, avatar):
    """Create a fresh nerfreal renderer instance."""
    if opt.model == 'musetalk':
        from musereal import MuseReal
        return MuseReal(opt, model, avatar)
    elif opt.model == 'wav2lip':
        from lipreal import LipReal
        return LipReal(opt, model, avatar)
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        return LightReal(opt, model, avatar)


# ─────────────────────────────────────────────────────────────────────
# Core: render one audio file to one MP4
# ─────────────────────────────────────────────────────────────────────

def render_one(audio_path, output_path, model_tuple, avatar_tuple,
               model_name, avatar_id, batch_size, file_index):
    """Render a single audio file into an MP4 with the talking avatar."""
    filename = Path(audio_path).stem
    duration_audio = load_audio(audio_path)
    chunks = audio_to_chunks(duration_audio)
    total_chunks = len(chunks)
    duration = len(duration_audio) / SAMPLE_RATE

    if total_chunks < 4:
        print(f'  Skipping {filename} — too short ({duration:.1f}s)')
        return False

    # Each video frame = 2 audio chunks (50 audio fps / 25 video fps)
    expected_frames = total_chunks // 2

    print(f'  {filename}  {duration:.0f}s  ~{expected_frames} frames')

    # Fresh instance per file — clean queue state
    opt = make_opt(avatar_id, model_name, batch_size, session_id=file_index)
    nerfreal = build_nerfreal(opt, model_tuple, avatar_tuple)

    # Push all audio chunks into the ASR queue
    for chunk in chunks:
        nerfreal.put_audio_frame(chunk)

    # Set dimensions for recording from the first avatar frame
    first_frame = nerfreal.frame_list_cycle[0]
    nerfreal.height, nerfreal.width, _ = first_frame.shape
    nerfreal.start_recording()

    # Start pipeline threads
    master_quit = Event()

    # TTS thread (harmless — just blocks on its empty message queue)
    nerfreal.init_customindex()
    nerfreal.tts.render(master_quit)

    # Inference thread
    infer_quit = Event()
    if model_name == 'musetalk':
        from musereal import inference
        infer_args = (infer_quit, batch_size, nerfreal.input_latent_list_cycle,
                      nerfreal.asr.feat_queue, nerfreal.asr.output_queue,
                      nerfreal.res_frame_queue,
                      nerfreal.vae, nerfreal.unet, nerfreal.pe, nerfreal.timesteps)
    elif model_name == 'wav2lip':
        from lipreal import inference
        infer_args = (infer_quit, batch_size, nerfreal.face_list_cycle,
                      nerfreal.asr.feat_queue, nerfreal.asr.output_queue,
                      nerfreal.res_frame_queue, nerfreal.model)
    elif model_name == 'ultralight':
        from lightreal import inference
        infer_args = (infer_quit, batch_size, nerfreal.face_list_cycle,
                      nerfreal.asr.feat_queue, nerfreal.asr.output_queue,
                      nerfreal.res_frame_queue, nerfreal.model)
    infer_thread = Thread(target=inference, args=infer_args)
    infer_thread.start()

    # Frame writer thread (offline — writes to ffmpeg, no WebRTC)
    pb = ProgressBar(filename)
    writer_quit = Event()
    writer_thread = Thread(target=offline_process_frames,
                           args=(nerfreal, writer_quit, expected_frames, pb.update))
    writer_thread.start()

    # Drive the ASR — this is the main processing loop
    while not master_quit.is_set() and not writer_quit.is_set():
        nerfreal.asr.run_step()

    # Tear down
    master_quit.set()
    infer_quit.set()
    infer_thread.join()
    writer_quit.set()
    writer_thread.join()
    nerfreal.stop_recording()

    # Move the muxed output to its final home
    temp_combined = 'data/record.mp4'
    if os.path.exists(temp_combined):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
        os.replace(temp_combined, output_path)

    # Clean up temp files
    for f in [f'temp{opt.sessionid}.mp4', f'temp{opt.sessionid}.aac']:
        if os.path.exists(f):
            os.remove(f)

    return True


# ─────────────────────────────────────────────────────────────────────
# Auto-detect avatar
# ─────────────────────────────────────────────────────────────────────

def detect_avatar():
    avatar_dir = './data/avatars'
    if not os.path.isdir(avatar_dir):
        return None
    avatars = [d for d in os.listdir(avatar_dir)
               if os.path.isdir(os.path.join(avatar_dir, d))]
    return avatars[0] if avatars else None


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Render talking avatar videos from audio files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument('audio_input',
                        help='Audio file or directory of audio files')
    parser.add_argument('output_dir',
                        help='Directory for output MP4 files')
    parser.add_argument('--avatar_id', default=None,
                        help='Avatar ID in data/avatars/ (auto-detected if only one)')
    parser.add_argument('--model', default='musetalk',
                        choices=['musetalk', 'wav2lip', 'ultralight'],
                        help='Lip-sync model (default: musetalk)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Inference batch size (default: 16)')

    args = parser.parse_args()

    # ── Validate inputs ──
    audio_files = find_audio_files(args.audio_input)
    if not audio_files:
        print(f'\n  No audio files found in: {args.audio_input}')
        print(f'  Supported: {", ".join(sorted(AUDIO_EXTENSIONS))}')
        sys.exit(1)

    avatar_id = args.avatar_id or detect_avatar()
    if not avatar_id:
        print('\n  No avatar found. Create one first:')
        print('    python prepare_avatar.py your_face.mp4')
        sys.exit(1)

    avatar_path = f'./data/avatars/{avatar_id}'
    if not os.path.isdir(avatar_path):
        print(f'\n  Avatar not found: {avatar_path}')
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Header ──
    total_dur = sum(len(load_audio(f)) / SAMPLE_RATE for f in audio_files)
    print()
    print('  render_course')
    print('  ' + '=' * 44)
    print(f'  Avatar:   {avatar_id}')
    print(f'  Model:    {args.model}')
    print(f'  Files:    {len(audio_files)} ({total_dur:.0f}s total audio)')
    print(f'  Output:   {os.path.abspath(args.output_dir)}/')
    print('  ' + '=' * 44)
    print()

    # ── Load model once ──
    print('  Loading model...')
    mp.set_start_method('spawn', force=True)

    if args.model == 'musetalk':
        from musereal import load_model, load_avatar, warm_up
        model_tuple = load_model()
        avatar_tuple = load_avatar(avatar_id)
        warm_up(args.batch_size, model_tuple)
    elif args.model == 'wav2lip':
        from lipreal import load_model, load_avatar, warm_up
        model_tuple = load_model('./models/wav2lip.pth')
        avatar_tuple = load_avatar(avatar_id)
        warm_up(args.batch_size, model_tuple, 256)
    elif args.model == 'ultralight':
        from lightreal import load_model, load_avatar, warm_up
        model_tuple = load_model(args)
        avatar_tuple = load_avatar(avatar_id)
        warm_up(args.batch_size, avatar_tuple, 160)

    print('  Model ready.\n')

    # ── Render each file ──
    rendered = []
    t0 = time.time()
    for i, audio_path in enumerate(audio_files, 1):
        stem = Path(audio_path).stem
        output_path = os.path.join(args.output_dir, f'{stem}.mp4')
        print(f'  [{i}/{len(audio_files)}]')
        ok = render_one(audio_path, output_path, model_tuple, avatar_tuple,
                        args.model, avatar_id, args.batch_size, i)
        if ok:
            rendered.append(output_path)
        print()

    # ── Summary ──
    elapsed = time.time() - t0
    print('  ' + '=' * 44)
    print(f'  Done. {len(rendered)} videos in {elapsed:.0f}s')
    print()
    for path in rendered:
        size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
        print(f'    {path}  ({size_mb:.1f} MB)')
    print()


if __name__ == '__main__':
    main()
