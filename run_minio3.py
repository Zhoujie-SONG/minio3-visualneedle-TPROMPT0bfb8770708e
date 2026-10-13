#!/usr/bin/env python3
"""Run and preserve one fresh episode using the existing Mini-o3 protocol."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont


PROJECT = Path('/home/songzhoujie/cvpr27/vstar')
OUTPUT = Path(__file__).resolve().parent
SAMPLE_ID = 'TPROMPT0bfb8770708e'
sys.path.insert(0, str(PROJECT))

import run_visualneedle_minio3_smoke20 as baseline
from run_visualneedle_minio3_smoke20_cache import CacheEnabledHFBackend


class RecordingBackend(CacheEnabledHFBackend):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.turn_number = 0

    def generate(self, messages, max_new_tokens):
        self.turn_number += 1
        step = OUTPUT / 'steps' / f'turn_{self.turn_number:02d}'
        step.mkdir(parents=True, exist_ok=True)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        (step / 'input_prompt.txt').write_text(prompt, encoding='utf-8')
        observations = OUTPUT / 'observations'
        observations.mkdir(exist_ok=True)
        image_index = 0
        inputs = []
        for message in messages:
            content = message.get('content')
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get('type') != 'image':
                    continue
                name = 'original_image' if image_index == 0 else f'observation_{image_index}'
                destination = observations / f'{name}.png'
                if not destination.exists():
                    item['image'].save(destination)
                inputs.append({'name': name, 'path': str(destination), 'size': list(item['image'].size)})
                image_index += 1
        baseline.atomic_write_json(step / 'input_images.json', inputs)
        print(f'Generating turn {self.turn_number}, input images={len(inputs)}', flush=True)
        generated = super().generate(messages, max_new_tokens)
        baseline.atomic_write_json(step / 'generation.json', generated)
        (step / 'model_output.txt').write_text(generated['raw_model_output'] + '\n', encoding='utf-8')
        print(generated['raw_model_output'], flush=True)
        return generated


def save_visualizations(episode):
    with Image.open(episode['original_image_path']) as handle:
        original = handle.convert('RGB')
    overview = original.copy()
    overview_draw = ImageDraw.Draw(overview)
    stroke = max(3, int(max(original.size) / 500))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#00a6a6']
    crops = episode['crops']
    tile_width, tile_height = 540, 430
    sheet = Image.new('RGB', (tile_width * 3, tile_height * max(1, (len(crops) + 2) // 3)), 'white')
    for index, crop in enumerate(crops):
        bbox = tuple(crop['predicted_bbox_clipped'])
        color = colors[index % len(colors)]
        individual = original.copy()
        draw = ImageDraw.Draw(individual)
        draw.rectangle(bbox, outline=color, width=stroke)
        draw.text((bbox[0], max(0, bbox[1] - 20)), f"Turn {crop['turn_index']}", fill=color)
        step = OUTPUT / 'steps' / f"turn_{crop['turn_index']:02d}"
        individual.save(step / 'bbox_on_original.png')
        overview_draw.rectangle(bbox, outline=color, width=stroke)
        overview_draw.text((bbox[0], max(0, bbox[1] - 20)), f"T{crop['turn_index']}", fill=color)
        with Image.open(crop['crop_path']) as handle:
            tile = handle.convert('RGB')
        tile.thumbnail((tile_width - 20, tile_height - 65))
        x, y = (index % 3) * tile_width, (index // 3) * tile_height
        sheet.paste(tile, (x + (tile_width - tile.width) // 2, y + 55))
        ImageDraw.Draw(sheet).text(
            (x + 10, y + 10),
            f"Turn {crop['turn_index']} | {crop['source']}\nOriginal bbox: {list(bbox)}",
            fill='black', font=ImageFont.load_default(),
        )
    overview.save(OUTPUT / 'crop_overview.png')
    if crops:
        sheet.save(OUTPUT / 'crop_contact_sheet.jpg', quality=95)


def main():
    if (OUTPUT / 'trajectory_raw.json').exists():
        raise FileExistsError('A completed episode already exists; preserve it before requesting another run.')
    seed = 42
    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sample = baseline.load_inference_samples()[SAMPLE_ID]
    baseline.atomic_write_json(OUTPUT / 'run_config.json', {
        'sample_id': SAMPLE_ID,
        'question': sample.question,
        'original_image': str(sample.image_path),
        'model_path': str(baseline.MODEL_PATH),
        'seed': seed,
        'do_sample': False,
        'use_cache': True,
        'max_rounds': baseline.MAX_ROUNDS,
        'max_images': baseline.MAX_IMAGES,
        'max_new_tokens_per_round': baseline.MAX_NEW_TOKENS_PER_ROUND,
        'total_generation_budget': baseline.TOTAL_GENERATION_BUDGET,
        'min_pixels': baseline.MIN_PIXELS,
        'max_pixels': baseline.MAX_PIXELS,
        'protocol_runner': str(PROJECT / 'run_visualneedle_minio3_smoke20.py'),
        'backend': 'CacheEnabledHFBackend with output recording only',
        'python': sys.executable,
        'torch': torch.__version__,
        'transformers': importlib.metadata.version('transformers'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'images_bbox_used_as_model_input': False,
    })
    with Image.open(sample.image_path) as handle:
        original = handle.convert('RGB')
    original.save(OUTPUT / 'original_image.png')
    backend = RecordingBackend(baseline.MODEL_PATH)
    episode = baseline.run_episode(backend, sample, OUTPUT)
    baseline.atomic_write_json(OUTPUT / 'trajectory_raw.json', episode)
    # Ground truth enters only after search has finished.
    metadata = baseline.load_evaluation_metadata()[SAMPLE_ID]
    evaluated = baseline.evaluate_episode(episode, metadata)
    baseline.atomic_write_json(OUTPUT / 'trajectory.json', evaluated)
    baseline.write_jsonl(OUTPUT / 'trajectory.jsonl', [evaluated])
    baseline.write_csv(OUTPUT / 'turn_records.csv', evaluated['turns'], baseline.TURN_FIELDS)
    baseline.write_csv(OUTPUT / 'crop_records.csv', evaluated['crops'], baseline.CROP_FIELDS)
    _, problems = baseline.integrity_checks([SAMPLE_ID], [evaluated])
    problems = [x for x in problems if not x.startswith('Expected 20 episode records')]
    baseline.atomic_write_json(OUTPUT / 'integrity.json', {'passed': not problems, 'problems': problems})
    blocks = [f"Question: {sample.question}", f"Status: {episode['status']}; stop reason: {episode['stop_reason']}"]
    for turn in episode['turns']:
        step = OUTPUT / 'steps' / f"turn_{turn['turn_index']:02d}"
        step.mkdir(parents=True, exist_ok=True)
        baseline.atomic_write_json(step / 'turn_record.json', turn)
        blocks.extend([
            f"\nTurn {turn['turn_index']} | action: {turn['action_type']}",
            turn['raw_model_output'],
            f"Source: {turn.get('source', '')}; original bbox: {turn.get('predicted_bbox_clipped')}",
            f"Raw crop: {turn.get('raw_crop_path', '')}\nModel crop: {turn.get('crop_path', '')}",
        ])
    (OUTPUT / 'crop_process.txt').write_text('\n\n'.join(blocks) + '\n', encoding='utf-8')
    save_visualizations(evaluated)
    summary = {
        'sample_id': SAMPLE_ID, 'status': episode['status'], 'stop_reason': episode['stop_reason'],
        'num_rounds': episode['num_rounds'], 'num_crops': episode['num_crops'],
        'final_answer': episode['final_answer'], 'ground_truth': metadata.gt_answer,
        'normalized_string_match': evaluated['normalized_string_match'],
        'runtime_seconds': episode['runtime_seconds'], 'integrity_passed': not problems,
    }
    baseline.atomic_write_json(OUTPUT / 'summary.json', summary)
    (OUTPUT / 'README.md').write_text(
        '# Mini-o3 single-image search\n\n'
        f"Sample: `{SAMPLE_ID}`\n\nQuestion: {sample.question}\n\n"
        f"Rounds: {episode['num_rounds']}; saved crops: {episode['num_crops']}; stop reason: {episode['stop_reason']}.\n\n"
        '- `crop_process.txt`: every action and raw model response in order.\n'
        '- `trajectory_raw.json`: the original inference episode.\n'
        '- `trajectory.json` / `.jsonl`: episode with offline evaluation metadata.\n'
        '- `crops/`: raw and model-preprocessed crops for every executed crop.\n'
        '- `steps/turn_XX/`: input prompt, input image list, model output, turn record, and crop bbox visualization.\n'
        '- `observations/`: images actually included in the model conversation.\n'
        '- `crop_overview.png`: all executed crop boxes on the original image.\n'
        '- `crop_contact_sheet.jpg`: crops in execution order.\n'
        '- `run_config.json`, `summary.json`, `integrity.json`: configuration, completion status, and checks.\n\n'
        'A grounding action at the configured cap is recorded but is not executed as a crop, matching the existing runner.\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if problems or episode['status'] in {'oom', 'error'}:
        raise RuntimeError(f"Run failed: {episode['status']}; integrity problems: {problems}")


if __name__ == '__main__':
    main()
