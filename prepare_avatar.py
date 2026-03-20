#!/usr/bin/env python3
"""
prepare_avatar.py — Create a talking avatar from a video or photo.

Usage:
    python prepare_avatar.py your_face.mp4
    python prepare_avatar.py your_face.mp4 --name my_avatar
    python prepare_avatar.py photo.jpg --name my_avatar --model wav2lip

This is a convenience wrapper around the existing avatar generation scripts.
It creates the avatar data in data/avatars/<name>/ so render_course.py can use it.

Tips for best results:
    - Use a 3-10 second video of yourself looking at the camera
    - Keep a neutral expression with natural blinks
    - Good lighting, plain background
    - Face centered and clearly visible
    - Video is better than a photo (gives natural movement)
"""
import argparse
import os
import sys
import subprocess
from pathlib import Path


VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def main():
    parser = argparse.ArgumentParser(
        description='Create a talking avatar from a video or photo.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument('input_file',
                        help='Video or image file of the face to use')
    parser.add_argument('--name', default=None,
                        help='Avatar name (default: derived from filename)')
    parser.add_argument('--model', default='musetalk',
                        choices=['musetalk', 'wav2lip'],
                        help='Which model to prepare for (default: musetalk)')
    parser.add_argument('--bbox_shift', type=int, default=0,
                        help='Face bounding box vertical shift (default: 0)')

    args = parser.parse_args()

    # ── Validate input ──
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f'\n  File not found: {args.input_file}')
        sys.exit(1)

    suffix = input_path.suffix.lower()
    if suffix not in VIDEO_EXTENSIONS and suffix not in IMAGE_EXTENSIONS:
        print(f'\n  Unsupported format: {suffix}')
        print(f'  Videos:  {", ".join(sorted(VIDEO_EXTENSIONS))}')
        print(f'  Images:  {", ".join(sorted(IMAGE_EXTENSIONS))}')
        sys.exit(1)

    # ── Derive avatar name ──
    avatar_name = args.name or input_path.stem.replace(' ', '_')

    avatar_dir = f'./data/avatars/{avatar_name}'
    if os.path.isdir(avatar_dir):
        print(f'\n  Avatar "{avatar_name}" already exists at {avatar_dir}')
        print(f'  To recreate, delete it first:  rm -rf {avatar_dir}')
        sys.exit(1)

    # ── Header ──
    print()
    print('  prepare_avatar')
    print('  ' + '=' * 44)
    print(f'  Input:    {args.input_file}')
    print(f'  Name:     {avatar_name}')
    print(f'  Model:    {args.model}')
    print(f'  Output:   {avatar_dir}/')
    print('  ' + '=' * 44)
    print()

    # ── Run the appropriate generation script ──
    abs_input = str(input_path.resolve())

    if args.model == 'musetalk':
        cmd = [
            sys.executable, 'musetalk/genavatar.py',
            '--file', abs_input,
            '--avatar_id', avatar_name,
            '--bbox_shift', str(args.bbox_shift),
        ]
    elif args.model == 'wav2lip':
        cmd = [
            sys.executable, 'wav2lip/genavatar.py',
            '--avatar_id', avatar_name,
            '--video_path', abs_input,
        ]

    print(f'  Running: {" ".join(cmd)}\n')
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f'\n  Avatar generation failed (exit code {result.returncode})')
        sys.exit(1)

    # ── Verify output ──
    # The wav2lip script saves to results/avatars/ instead of data/avatars/
    # Move it if needed
    alt_dir = f'./results/avatars/{avatar_name}'
    if not os.path.isdir(avatar_dir) and os.path.isdir(alt_dir):
        os.makedirs('./data/avatars', exist_ok=True)
        os.rename(alt_dir, avatar_dir)
        print(f'  Moved avatar to {avatar_dir}')

    if os.path.isdir(avatar_dir):
        file_count = sum(1 for _ in Path(avatar_dir).rglob('*') if _.is_file())
        print(f'\n  Avatar "{avatar_name}" ready ({file_count} files)')
        print(f'  Location: {os.path.abspath(avatar_dir)}')
        print()
        print(f'  Next step — render your course:')
        print(f'    python render_course.py ./lessons ./output --avatar_id {avatar_name}')
        print()
    else:
        print(f'\n  Warning: Avatar directory not found at {avatar_dir}')
        print(f'  Check the output above for errors.')
        sys.exit(1)


if __name__ == '__main__':
    main()
