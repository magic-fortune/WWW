"""Scheme1 training weights and ranking metrics."""
import numpy as np

OPERATIONS=('CLICK','TYPE','SELECT')

def canonical(operation):
    return 'CLICK' if operation in ('CLICK', 'ENTER', 'HOVER') else operation

def build_weights(rows, indices, labels):
    indices = np.asarray(indices)
    train = indices[labels[indices].any(1)]
    tasks = {}
    for i in train:
        tasks[rows[i]['task_id']] = tasks.get(rows[i]['task_id'], 0) + 1
    task_weight = np.array([1 / tasks[rows[i]['task_id']] for i in train], dtype=np.float64)
    operations = np.array([canonical(rows[i]['operation']) for i in train])
    assert set(operations) <= set(OPERATIONS)
    mass = {op: float(task_weight[operations == op].sum()) for op in OPERATIONS}
    total = task_weight.sum()
    probability = {op: mass[op] / total for op in OPERATIONS}
    raw = task_weight * np.array([1 / np.sqrt(probability[op]) for op in operations])
    raw /= raw.mean()
    weights = np.zeros(len(rows), dtype=np.float64)
    weights[train] = raw
    assert np.isfinite(weights).all() and (weights[train] > 0).all()
    assert abs(weights[train].mean() - 1) < 1e-12
    info = dict(supervised_steps=len(train), trajectories=len(tasks), counts={op: int((operations == op).sum()) for op in OPERATIONS}, trajectory_weighted_operation_mass=probability, final_weight_mass={op: float(raw[operations == op].sum() / raw.sum()) for op in OPERATIONS}, weight_min=float(raw.min()), weight_max=float(raw.max()), weight_mean=float(raw.mean()))
    return (train, weights, info)

def task_weights(rows, eligible):
    counts = {}
    for row, ok in zip(rows, eligible):
        if ok:
            counts[row['task_id']] = counts.get(row['task_id'], 0) + 1
    return np.array([1 / (len(counts) * counts[r['task_id']]) if ok else 0.0 for r, ok in zip(rows, eligible)])

def conditional_metrics(scores, y, mask, rows):
    s = np.where(mask, scores, -1000000000.0)
    pred = s.argmax(1)
    hit = y[np.arange(len(y)), pred]
    recall = y.any(1)
    order = np.argsort(-s, axis=1, kind='stable')
    rank = np.take_along_axis(y, order, 1).argmax(1) + 1
    return dict(conditional_top1=float(hit @ task_weights(rows, recall)), system_top1=float(hit @ task_weights(rows, np.ones(len(y), bool))), conditional_mrr=float(1 / rank @ task_weights(rows, recall)), recall_macro=float(recall @ task_weights(rows, np.ones(len(y), bool))), recall_micro=float(recall.mean()), recalled_steps=int(recall.sum()), steps=len(y))

def measurement(scores, y, mask, rows):
    result = conditional_metrics(scores, y, mask, rows)
    pred = np.where(mask, scores, -1000000000.0).argmax(1)
    hit = y[np.arange(len(y)), pred]
    result.update(all_step_element_accuracy=float(hit.mean()), conditional_step_top1=float(hit[y.any(1)].mean()))
    result['operation_macro'] = {}
    result['operation_micro'] = {}
    result['operation_counts'] = {}
    for op in ('CLICK', 'TYPE', 'SELECT'):
        eligible = y.any(1) & np.array([canonical(r['operation']) == op for r in rows])
        count = int(eligible.sum())
        if count:
            result['operation_macro'][op] = float(hit @ task_weights(rows, eligible))
            result['operation_micro'][op] = float(hit[eligible].mean())
            result['operation_counts'][op] = count
    return result

def full_step_vectors(scores, labels, mask):
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if scores.ndim != 2 or scores.shape != labels.shape or scores.shape != mask.shape:
        raise ValueError('Matching(step,candidate) arrays required')
    if not np.isfinite(scores[mask]).all():
        raise ValueError('Nonfinite valid candidate score')
    positives = labels & mask
    recalled = positives.any(1)
    valid = np.where(mask, scores, -np.inf)
    order = np.argsort(-valid, axis=1, kind='stable')
    sorted_positive = np.take_along_axis(positives, order, axis=1)
    rank = np.where(recalled, sorted_positive.argmax(1) + 1, 0)
    return dict(recalled=recalled, rank=rank, acc1=(rank == 1).astype(np.float64), acc5=((rank > 0) & (rank <= 5)).astype(np.float64), reciprocal_rank=np.divide(1.0, rank, out=np.zeros(len(rank), dtype=np.float64), where=rank > 0), prediction=order[:, 0])

def full_step_metrics(scores, labels, mask):
    vectors = full_step_vectors(scores, labels, mask)
    n = len(vectors['rank'])
    if not n:
        raise ValueError('At least one decision step required')
    return dict(full_step_acc1=float(vectors['acc1'].mean()), full_step_acc5=float(vectors['acc5'].mean()), full_step_mrr=float(vectors['reciprocal_rank'].mean()), steps=n, recalled_steps=int(vectors['recalled'].sum()), candidate_recall=float(vectors['recalled'].mean()), averaging='step micro over all decision steps; unrecalled target contributes0', tie_break='stable original candidate order')

def metrics(scores, labels, mask, rows):
    result = measurement(scores, labels, mask, rows)
    result.update(full_step_metrics(scores, labels, mask))
    op = np.array([canonical(row['operation']) for row in rows])
    result['full_step_operations'] = {}
    for name in ('CLICK', 'TYPE', 'SELECT'):
        ids = op == name
        if ids.any():
            result['full_step_operations'][name] = full_step_metrics(scores[ids], labels[ids], mask[ids])
    return result
