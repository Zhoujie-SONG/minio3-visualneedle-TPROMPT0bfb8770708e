# Mini-o3 single-image search

Sample: `TPROMPT0bfb8770708e`

Question: Please find the two parallel strings of chili peppers in the picture. On the counter below them, counting from left to right, what product is behind the third item?

Rounds: 12; saved crops: 11; stop reason: image_or_turn_cap.

- `crop_process.txt`: every action and raw model response in order.
- `trajectory_raw.json`: the original inference episode.
- `trajectory.json` / `.jsonl`: episode with offline evaluation metadata.
- `crops/`: raw and model-preprocessed crops for every executed crop.
- `steps/turn_XX/`: input prompt, input image list, model output, turn record, and crop bbox visualization.
- `observations/`: images actually included in the model conversation.
- `crop_overview.png`: all executed crop boxes on the original image.
- `crop_contact_sheet.jpg`: crops in execution order.
- `run_config.json`, `summary.json`, `integrity.json`: configuration, completion status, and checks.

A grounding action at the configured cap is recorded but is not executed as a crop, matching the existing runner.
