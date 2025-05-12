import os
import json
import csv
import subprocess

def get_video_fps(video_path):
    """
    Uses ffprobe to extract the frame rate of the video.
    Returns it as a float, or None if something goes wrong.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=nokey=1:noprint_wrappers=1",
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        raw_fps = result.stdout.strip()
        if "/" in raw_fps:
            num, denom = raw_fps.split("/")
            return round(float(num) / float(denom), 3)
        else:
            return round(float(raw_fps), 3)
    except Exception as e:
        print(f"Failed to get fps for {video_path}: {e}")
        return None

def generate_csv(video_folder, jsonl_path, output_csv_path):
    rows = []

    import glob

    # Read the JSONL metadata file
    with open(jsonl_path, "r") as f:
        for line_idx, line in enumerate(f):
            obj = json.loads(line)
            yt_key = obj["key"]

            # Try all known extensions
            possible_exts = [".mp4", ".webm", ".mkv"]
            matching_files = [
                os.path.join(video_folder, f"{yt_key}{ext}")
                for ext in possible_exts
                if os.path.exists(os.path.join(video_folder, f"{yt_key}{ext}"))
            ]

            if not matching_files:
                print(f"Missing video file for key {yt_key}, row #{line_idx}")
                continue

            # Use the first matching file (you can sort to prefer .mp4 if you want)
            video_path = matching_files[0]

            # Extract video resolution
            width = obj["video_info"]["resolution"]["width"]
            height = obj["video_info"]["resolution"]["height"]

            # Extract frame rate using ffprobe
            fps = get_video_fps(video_path)
            if fps is None:
                print(f"Could not determine FPS for {yt_key}, skipping.")
                continue

            # Loop through the QA pairs
            for qa in obj["qa"]:
                question_text = qa["question"]
                answer_letter = qa["answer"].strip()

                # Extract answer choices from the question text
                options = []
                parts = question_text.split('\n')
                for part in parts:
                    if part.strip().startswith("("):
                        options.append(part.split(")", 1)[1].strip())

                # Pad to 4 options if needed
                if len(options) < 4:
                    options += [""] * (4 - len(options))

                # Get the full answer text based on letter
                letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
                answer_idx = letter_to_idx.get(answer_letter, None)
                if answer_idx is None or answer_idx >= len(options):
                    print(f"Warning: Invalid answer {answer_letter} for key {yt_key}, skipping question.")
                    continue
                full_answer = options[answer_idx]

                # Extract just the main question (first line)
                clean_question = parts[0].strip()

                # Construct the CSV row
                row = {
                    "video": yt_key,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "question": clean_question,
                    "answer": full_answer,
                    "type": qa.get("question_type", [""])[0],
                    "a0": options[0],
                    "a1": options[1],
                    "a2": options[2],
                    "a3": options[3],
                }
                rows.append(row)

    # Write to CSV
    with open(output_csv_path, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ["video", "width", "height", "fps", "question", "answer", "type", "a0", "a1", "a2", "a3"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"CSV generation complete. {len(rows)} rows written.")

# Run the function with new paths
generate_csv(
    video_folder="./raw_videos",
    jsonl_path="./data/video_info.meta.jsonl",
    output_csv_path="./csvs/full.csv"
)
