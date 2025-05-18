import os
import json
import csv
import subprocess

def get_video_fps_and_duration(video_path):
    """
    Uses ffprobe to extract the frame rate and duration of the video.
    Returns (fps, duration) as floats, or (None, None) if something goes wrong.
    """
    try:
        # Get both frame rate and duration in one call
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,duration",
                "-of", "default=nokey=1:noprint_wrappers=1",
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) < 2:
            return None, None

        raw_fps, raw_duration = output_lines
        if "/" in raw_fps:
            num, denom = raw_fps.split("/")
            fps = round(float(num) / float(denom), 3)
        else:
            fps = round(float(raw_fps), 3)
        duration = round(float(raw_duration), 3)
        return fps, duration
    except Exception as e:
        print(f"Failed to get fps/duration for {video_path}: {e}")
        return None, None

def generate_csv(video_folder, jsonl_path, output_csv_path):
    rows = []
    import glob

    with open(jsonl_path, "r") as f:
        for line_idx, line in enumerate(f):
            obj = json.loads(line)
            yt_key = obj["key"]

            possible_exts = [".mp4", ".webm", ".mkv"]
            matching_files = [
                os.path.join(video_folder, f"{yt_key}{ext}")
                for ext in possible_exts
                if os.path.exists(os.path.join(video_folder, f"{yt_key}{ext}"))
            ]

            if not matching_files:
                print(f"Missing video file for key {yt_key}, row #{line_idx}")
                continue

            video_path = matching_files[0]

            # Get fps and duration
            fps, duration = get_video_fps_and_duration(video_path)
            if fps is None or duration is None:
                print(f"Could not determine FPS or duration for {yt_key}, skipping.")
                continue

            for qa in obj["qa"]:
                question_text = qa["question"]
                answer_letter = qa["answer"].strip()

                options = []
                parts = question_text.split('\n')
                for part in parts:
                    if part.strip().startswith("("):
                        options.append(part.split(")", 1)[1].strip())

                if len(options) < 4:
                    options += [""] * (4 - len(options))

                letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
                answer_idx = letter_to_idx.get(answer_letter, None)
                if answer_idx is None or answer_idx >= len(options):
                    print(f"Warning: Invalid answer {answer_letter} for key {yt_key}, skipping question.")
                    continue
                full_answer = options[answer_idx]

                clean_question = parts[0].strip()

                row = {
                    "video": yt_key,
                    "fps": fps,
                    "duration": duration,
                    "question": clean_question,
                    "answer": full_answer,
                    "a0": options[0],
                    "a1": options[1],
                    "a2": options[2],
                    "a3": options[3],
                }
                rows.append(row)

    with open(output_csv_path, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ["video", "fps", "duration", "question", "answer", "a0", "a1", "a2", "a3"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"CSV generation complete. {len(rows)} rows written.")

# Run the function with new paths
generate_csv(
    video_folder="./all_videos",
    jsonl_path="video_info.meta.jsonl",
    output_csv_path="./csvs/full.csv"
)
