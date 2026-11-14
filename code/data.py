"""Scheme1's original four-stream 384D inputs; no model or score inheritance."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from config import DATA_DIR, ROLE_CODES, ARTIFACT_DIR
from metrics import metrics, build_weights
from utils import sha, dump

class Data:
    def __init__(self):
        self.manifest=json.loads((DATA_DIR/'manifest.json').read_text())
        with np.load(DATA_DIR/'inputs.npz') as values:self.a={k:values[k] for k in values.files}
        self.meta=json.loads((DATA_DIR/'metadata.json').read_text())
        self.geometry=dict(np.load(DATA_DIR/'projection.npz'))
        self.roles=dict(np.load(ROLE_CODES))
        self.raw_vectors=np.load(DATA_DIR/'embeddings.npy',mmap_mode='r')
        cache=ARTIFACT_DIR/'data'
        self.phase_cache=np.load(cache/'phase_overlaps.npy',mmap_mode='r') if (cache/'phase_overlaps.npy').exists() else None
        self.channel_cache=np.load(cache/'channel_expectations.npy',mmap_mode='r') if (cache/'channel_expectations.npy').exists() else None

    def batch(self,split,ids):
        get=lambda key:self.a[split+'_'+key][ids]
        d=np.asarray(self.raw_vectors[get('element_index')]);c=np.asarray(self.raw_vectors[get('context_index')])
        t=np.broadcast_to(self.raw_vectors[get('intent_index')][:,None],d.shape)
        b=np.broadcast_to(self.raw_vectors[get('background_index')][:,None],d.shape)
        streams=F.normalize(torch.from_numpy(np.stack((t,d,c,b),-2).astype(np.float32)).double(),dim=-1)
        layout=(get('state')[...,self.geometry['layout_active']]-self.geometry['layout_mean'])/self.geometry['layout_std']
        role=torch.from_numpy(self.roles[split][ids]).long()[:,None].expand(-1,50)
        return {'streams':streams,'layout':torch.from_numpy(np.tanh(layout).astype(np.float32)),'role':role}

    def cached_batch(self,ids):
        value=self.batch('train',ids)
        if self.phase_cache is not None:value['overlaps']=torch.from_numpy(np.array(self.phase_cache[ids],copy=True)).double()
        if self.channel_cache is not None:value['channel_expectations']=torch.from_numpy(np.array(self.channel_cache[ids],copy=True)).double()
        return value

    def labels(self,split):return self.a[split+'_labels']
    def mask(self,split):return self.a[split+'_mask']

def evaluate(model, data, ids, split='train'):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            ix = ids[start:start + 32]
            batch = data.cached_batch(ix) if split == 'train' else data.batch(split, ix)
            parts.append(model(batch).numpy())
    scores = np.concatenate(parts)
    rows = [data.meta[split][int(i)] for i in ids]
    return (metrics(scores, data.labels(split)[ids], data.mask(split)[ids], rows), scores)

