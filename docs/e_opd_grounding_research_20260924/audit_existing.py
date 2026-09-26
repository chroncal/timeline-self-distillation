"""Read-only CPU audit of coordinate stopping decisions in existing pure E-OPD.

Run from the repository root with .venv/bin/python. Writes only beside this
script. Does not import the trainer, load model weights, or access CUDA.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PURE = ROOT / 'outputs/research_experiments/mmgcot_timeline_training/pure_opd_v2'
TOKENIZER = Path('/mnt/sda/sujingyang/models/Qwen3.5-0.8B/tokenizer.json')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    tokenizer = Tokenizer.from_file(str(TOKENIZER))

    @lru_cache(None)
    def decode(tid):
        return tokenizer.decode([tid], skip_special_tokens=False)

    result = {'scope': 'existing pure E-OPD training rollouts; repeated sample visits, not independent images',
              'tokenizer_sha256': sha(TOKENIZER), 'runs': {}, 'examples': []}
    for path in sorted((PURE / 'formal').glob('e_opd_pure_seed*/rollouts.jsonl')):
        counts = Counter()
        for line in path.open():
            row = json.loads(line)
            counts['rollouts'] += 1
            counts['invalid_rollouts'] += not row['valid']
            pieces = [decode(t) for t in row['token_ids']]
            full = ''.join(pieces)
            assert full == tokenizer.decode(row['token_ids'], skip_special_tokens=False)
            end = full.find(']')
            seen = False
            cursor = 0
            for j, (piece, support, mask, teacher) in enumerate(zip(
                    pieces, row['support_ids'], row['numeric_mask'],
                    row['teacher_support_logits'], strict=True)):
                # Includes the decision that can terminate the last coordinate,
                # but excludes choices after the first closing bracket.
                before_close = end < 0 or cursor <= end
                alternatives = [decode(v) for v in support]
                has_digit = any(s and s[0].isdigit() for s in alternatives)
                has_end = any(s.startswith((',', ']')) for s in alternatives)
                if before_close and has_digit and has_end:
                    counts['digit_or_stop_decision_positions'] += 1
                    if mask:
                        counts['digit_or_stop_positions_with_kl'] += 1
                        assert teacher is not None
                    else:
                        counts['digit_or_stop_positions_without_kl'] += 1
                        assert piece.startswith((',', ']')), (piece, alternatives)
                        assert teacher is None
                        seen = True
                        if len(result['examples']) < 12:
                            result['examples'].append({
                                'run': path.parent.name, 'sample_id': row['sample_id'],
                                'step': row['step'], 'bbox': row['bbox'],
                                'response': full, 'position': j, 'prefix': ''.join(pieces[:j]),
                                'selected': piece, 'legal_alternatives': alternatives,
                                'numeric_mask': mask, 'saved_teacher_logits': teacher,
                            })
                cursor += len(piece)
            counts['rollouts_with_unsupervised_digit_or_stop'] += seen
        result['runs'][path.parent.name] = {'counts': dict(counts), 'sha256': sha(path)}
    total = Counter()
    for v in result['runs'].values():
        total.update(v['counts'])
    result['total'] = dict(total)
    result['fractions'] = {
        'unsupervised_among_digit_or_stop_positions':
            total['digit_or_stop_positions_without_kl'] / total['digit_or_stop_decision_positions'],
        'rollouts_with_unsupervised_digit_or_stop':
            total['rollouts_with_unsupervised_digit_or_stop'] / total['rollouts'],
    }
    # These are mathematical counterexamples, NOT model measurements.
    # A wrong candidate can be more image-sensitive than the correct one.
    present = [0.7, 0.3]  # [correct box, wrong box]
    removed = [0.95, 0.05]
    amplified = [p*p/n for p, n in zip(present, removed)]
    amplified = [v/sum(amplified) for v in amplified]
    result['synthetic_checks'] = {
        'label': 'illustrative distributions, not experimental grounding scores',
        'visual_ratio_can_promote_wrong_box': {
            'image_present': present, 'image_removed': removed,
            'contrast_alpha_1': amplified,
            'correct_box_probability_decreases': amplified[0] < present[0],
        },
        'digit_spelling_not_coordinate_distance': {
            '299_to_300': {'absolute_error': 1, 'digit_substitutions': 3},
            '300_to_900': {'absolute_error': 600, 'digit_substitutions': 1},
        },
    }
    summary = PURE / 'evaluation/dev/pure_vs_base_three_seed_paired.json'
    original = json.loads(summary.read_text())
    result['existing_dev'] = {
        'source_sha256': sha(summary), 'n_images': original['n_images'],
        'means': {k: v['mIoU'] for k, v in original['per_system'].items()},
        'paired': original['paired_deltas'],
        'note': 'Base-L is a shared frozen baseline; its repetition is not three independently trained baselines.',
    }
    groups_report = {}
    for path in sorted((PURE / 'evaluation/dev').glob('e_opd_pure_seed*.jsonl')):
        groups = defaultdict(list)
        for line in path.open():
            row = json.loads(line)
            if row['mode'] != 'greedy':
                groups[(row['sample_id'], row['trajectory_index'])].append(row)
        assert len(groups) == 60
        counts = Counter()
        means, maxima = [], []
        for rows in groups.values():
            assert len(rows) == 4 and len({r['draw'] for r in rows}) == 4
            values = [r['iou'] for r in rows]
            counts['groups'] += 1
            counts['all_zero'] += max(values) == 0
            counts['equal_reward'] += max(values) - min(values) <= 1e-12
            counts['has_reward_variation'] += max(values) - min(values) > 1e-12
            counts['has_iou_over_0_5'] += max(values) > .5
            means.append(sum(values)/4)
            maxima.append(max(values))
        groups_report[path.stem] = {
            'counts': dict(counts), 'mean_iou': sum(means)/len(means),
            'oracle_best_of_4_mean_iou': sum(maxima)/len(maxima),
            'sha256': sha(path),
        }
    result['grpo_reward_diagnostic'] = {
        'note': 'Post-training development checkpoint, four saved draws per context; NOT train groups, NOT a new GRPO result. Oracle best-of-4 uses GT selection and is not deployable performance.',
        'runs': groups_report,
    }
    (OUT / 'existing_evidence.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'total': result['total'], 'fractions': result['fractions'],
                      'synthetic_checks': result['synthetic_checks']}, indent=2))


if __name__ == '__main__':
    main()
