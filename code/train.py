"""Scheme1 training only; starts no background workers."""
import argparse
import copy
import hashlib
import json
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import make
from config import ROOT, OUT, PROTOCOL_FILE, protocol
from data import Data, build_weights, dump, sha, evaluate

def key(m):
    return (m['conditional_top1'], m['conditional_mrr'], m['full_step_acc1'])

def guarded(m, b):
    return m['conditional_mrr'] >= b['conditional_mrr'] - 0.005 - 1e-12 and all((m[k][op] >= b[k][op] - 0.01 - 1e-12 for k in ('operation_macro', 'operation_micro') for op in ('TYPE', 'SELECT')))

def learning_rate(h, cfg):
    return cfg['learning_rate_min'] + (cfg['learning_rate_max'] - cfg['learning_rate_min']) * (1 + math.cos(math.pi * (h - 1) / (cfg['max_segments'] - 1))) / 2

def loss_function(score, labels, mask, weights, pair_weight):
    valid = score.masked_fill(~mask, -1000000000.0)
    positive = score.masked_fill(~labels, -1000000000.0).logsumexp(1)
    ce = valid.logsumexp(1) - positive
    negative_mask = mask & ~labels
    hardest = score.masked_fill(~negative_mask, -1000000000.0).max(1).values
    pair = torch.where(negative_mask.any(1), F.softplus(hardest - positive), torch.zeros_like(positive))
    return (((ce + pair_weight * pair) * weights.to(score.dtype)).mean(), float(ce.detach().mean()), float(pair.detach().mean()))

def fit(name, seed, stage='development', selected=None, horizon=None):
    torch.set_num_threads(4)
    cfg = protocol()
    folder = OUT / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f'{name}_{stage}'
    if stem.with_suffix('.json').exists():
        info = json.loads(stem.with_suffix('.json').read_text())
        assert info['protocol_sha256'] == sha(PROTOCOL_FILE) and info['checkpoint_sha256'] == sha(stem.with_suffix('.pt'))
        return
    data = Data()
    model = make(name, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate_max'], weight_decay=cfg['weight_decay'])
    ids = np.asarray(cfg['train_indices']) if stage == 'development' else np.arange(len(data.labels('train')))
    train, weights, weight_info = build_weights(data.meta['train'], ids, data.labels('train'))
    val = np.asarray(cfg['validation_indices'])
    rng = np.random.default_rng(seed)
    history = []
    order = None
    first = 1
    baseline = None
    best = None
    best_state = None
    best_h = 0
    unconstrained = None
    free_h = 0
    last_significant = 0
    significant_macro = None
    selected_state = None
    if stage == 'development':
        baseline, s = evaluate(model, data, val)
        best = baseline
        unconstrained = baseline
        best_state = copy.deepcopy(model.state_dict())
        significant_macro = baseline['conditional_top1']
        np.save(folder / f'{name}_initial_scores.npy', s)
    elif selected == 0:
        selected_state = copy.deepcopy(model.state_dict())
    resume = stem.with_name(stem.name + '_resume.pt')
    if resume.exists():
        r = torch.load(resume, weights_only=True)
        assert r['protocol_sha256'] == sha(PROTOCOL_FILE) and r['seed'] == seed
        model.load_state_dict(r['model'])
        optimizer.load_state_dict(r['optimizer'])
        rng.bit_generator.state = r['rng']
        order = r['order'].numpy()
        history = r['history']
        first = r['segment'] + 1
        baseline = r['baseline']
        best = r['best']
        best_state = r['best_state']
        best_h = r['best_segment']
        unconstrained = r['unconstrained']
        free_h = r['unconstrained_segment']
        last_significant = r['last_significant']
        significant_macro = r['significant_macro']
        selected_state = r['selected_state']
    limit = cfg['max_segments'] if stage == 'development' else horizon
    reason = 'max_segments'
    if stage == 'development' and history and (len(history) >= cfg['early_stop_min_segments']) and (len(history) - last_significant >= cfg['early_stop_patience_segments']):
        limit = len(history)
        reason = 'patience'
    for h in range(first, limit + 1):
        tick = time.time()
        model.train()
        if (h - 1) % 2 == 0:
            order = rng.permutation(train)
        chunk = np.array_split(order, 2)[(h - 1) % 2]
        lr = learning_rate(h, cfg)
        for group in optimizer.param_groups:
            group['lr'] = lr
        values = []
        ces = []
        pairs = []
        norms = []
        module_norms = None
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        for start in range(0, len(chunk), 32):
            ix = chunk[start:start + 32]
            optimizer.zero_grad(set_to_none=True)
            score = model(data.cached_batch(ix))
            loss, ce, pair = loss_function(score, torch.from_numpy(data.labels('train')[ix]), torch.from_numpy(data.mask('train')[ix]), torch.from_numpy(weights[ix]), cfg['pair_loss_weight'])
            loss.backward()
            if module_norms is None:
                module_norms = {n: float(p.grad.norm()) for n, p in model.named_parameters()}
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip']))
            assert torch.isfinite(loss) and np.isfinite(norm)
            optimizer.step()
            values.append(float(loss.detach()))
            ces.append(ce)
            pairs.append(pair)
            norms.append(norm)
        row = dict(segment=h, epoch=h / 2, learning_rate=lr, loss=float(np.mean(values)), listwise_loss=float(np.mean(ces)), hard_pair_loss=float(np.mean(pairs)), mean_gradient_norm=float(np.mean(norms)), first_batch_module_gradient_norms=module_norms, module_parameter_update={n: float((p.detach() - before[n]).norm()) for n, p in model.named_parameters()}, seconds=time.time() - tick, examples=len(chunk), updates=math.ceil(len(chunk) / 32), chunk_sha256=hashlib.sha256(chunk.tobytes()).hexdigest())
        stop = False
        if stage == 'development':
            m, _ = evaluate(model, data, val)
            row.update(validation=m, guard_passed=guarded(m, baseline))
            if key(m) > key(unconstrained):
                unconstrained = m
                free_h = h
            if row['guard_passed'] and key(m) > key(best):
                best = m
                best_h = h
                best_state = copy.deepcopy(model.state_dict())
            if row['guard_passed'] and m['conditional_top1'] >= significant_macro + cfg['early_stop_min_gain'] - 1e-12:
                significant_macro = m['conditional_top1']
                last_significant = h
            stop = h >= cfg['early_stop_min_segments'] and h - last_significant >= cfg['early_stop_patience_segments']
            reason = 'patience' if stop else 'max_segments'
        elif h == selected:
            selected_state = copy.deepcopy(model.state_dict())
        history.append(row)
        state = dict(protocol_sha256=sha(PROTOCOL_FILE), seed=seed, model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng.bit_generator.state, order=torch.from_numpy(order.copy()), segment=h, history=history, baseline=baseline, best=best, best_state=best_state, best_segment=best_h, unconstrained=unconstrained, unconstrained_segment=free_h, last_significant=last_significant, significant_macro=significant_macro, selected_state=selected_state)
        temp = resume.with_suffix('.partial')
        torch.save(state, temp)
        temp.replace(resume)
        dump(stem.with_name(stem.name + '_history.json'), history)
        dump(stem.with_name(stem.name + '_status.json'), row)
        msg = f"{name} seed{seed} {stage} epoch{h / 2:g} loss{row['loss']:.5f} seconds{row['seconds']:.2f}"
        if stage == 'development':
            msg += f" macro={100 * m['conditional_top1']:.3f}% guard={row['guard_passed']} stale={h - last_significant}"
        print(msg, flush=True)
        if stop:
            break
    chosen = best_h if stage == 'development' else selected
    model.load_state_dict(best_state if stage == 'development' else selected_state)
    torch.save(model.state_dict(), stem.with_suffix('.pt'))
    info = dict(name=name, seed=seed, stage=stage, selected_segment=chosen, selected_epoch=chosen / 2, completed_segments=len(history), history=history, baseline=baseline, selected=best, unconstrained=unconstrained, unconstrained_segment=free_h, stop_reason=reason, last_significant=last_significant, training_weights=weight_info, actual_updates=sum((r['updates'] for r in history)), actual_examples=sum((r['examples'] for r in history)), protocol_sha256=sha(PROTOCOL_FILE), checkpoint_sha256=sha(stem.with_suffix('.pt')), pretrained_weights_loaded=False)
    dump(stem.with_suffix('.json'), info)
    if stage == 'development':
        m, s = evaluate(model, data, val)
        np.save(folder / f'{name}_development_scores.npy', s)
        dump(folder / f'{name}_development_metrics.json', m)
    print('DONE', name, seed, stage, 'selected_epoch', chosen / 2, 'executed_epochs', len(history) / 2, flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Train Scheme1 only in a new output directory')
    parser.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    parser.add_argument('--stage',choices=['development','final'],default='development')
    parser.add_argument('--selected',type=int)
    parser.add_argument('--horizon',type=int)
    args=parser.parse_args()
    if args.stage=='final' and (args.selected is None or args.horizon is None):
        parser.error('final needs development-frozen --selected and --horizon')
    fit('quantum_structure',args.seed,args.stage,args.selected,args.horizon)
