import os
import json
import csv

def generate_csv(video_folder, jsonl_path, output_csv_path):
    # Read all metadata JSON files
    metadata_files = {}
    json_folder = os.path.join(video_folder)
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(json_folder, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                metadata_files[data["url"]] = (data["key"], filename)

    rows = []

    with open(jsonl_path, "r") as f:
        for line_idx, line in enumerate(f):
            obj = json.loads(line)
            yt_key = obj["key"]
            constructed_url = f"https://www.youtube.com/watch?v={yt_key}"

            if constructed_url not in metadata_files:
                print(f"No matching metadata file for key {yt_key}, row #{line_idx}")
                continue

            key_id, metadata_filename = metadata_files[constructed_url]
            mp4_path = os.path.join(json_folder, f"{key_id}.mp4")
            if not os.path.exists(mp4_path):
                print(f"Metadata file found for key {yt_key}, row #{line_idx}, but missing video file {key_id}.mp4")
                continue

            # Now we can process
            # frame_count = None  # You can leave it None unless you parse video fps*duration
            width = obj["video_info"]["resolution"]["width"]
            height = obj["video_info"]["resolution"]["height"]

            for qa in obj["qa"]:
                question_text = qa["question"]
                answer_letter = qa["answer"].strip()

                # Parse options
                options = []
                parts = question_text.split('\n')
                for part in parts:
                    if part.strip().startswith("("):
                        options.append(part.split(")", 1)[1].strip())

                if len(options) < 4:
                    options += [""] * (4 - len(options))

                # Resolve full answer text
                letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
                answer_idx = letter_to_idx.get(answer_letter, None)
                if answer_idx is None or answer_idx >= len(options):
                    print(f"Warning: Invalid answer {answer_letter} for key {yt_key}, skipping question.")
                    continue
                full_answer = options[answer_idx]

                # Clean question
                clean_question = parts[0].strip()

                row = {
                    "video": key_id,
                    # "frame_count": frame_count if frame_count is not None else "",
                    "width": width,
                    "height": height,
                    "question": clean_question,
                    "answer": full_answer,
                    "type": qa.get("question_type", [""])[0],
                    "a0": options[0],
                    "a1": options[1],
                    "a2": options[2],
                    "a3": options[3],
                }
                rows.append(row)

    # Write CSV
    with open(output_csv_path, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ["video", 
                    #   "frame_count",
                      "width", "height", "question", "answer", "type", "a0", "a1", "a2", "a3"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"CSV generation complete. {len(rows)} rows written.")


generate_csv(
    video_folder="/local/minh/VIPER/datasets/LVBench/scripts/videos/00000",
    jsonl_path="/local/minh/VIPER/datasets/LVBench/data/video_info.meta.jsonl",
    output_csv_path="/local/minh/VIPER/datasets/LVBench/csvs/list.csv"
)