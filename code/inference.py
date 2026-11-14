"""Load and score with Scheme1 only; no alternative models."""
import json
import torch
from model import make, counts
from config import ARTIFACT_DIR, PROTOCOL_FILE
from utils import sha

def load_grounder(seed=0,stage='final',artifact_dir=None):
    if seed not in (0,1,2) or stage not in ('development','final'):raise ValueError('Unknown Scheme1 checkpoint')
    from pathlib import Path
    root=Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    path=root/'results'/f'seed{seed}'/f'quantum_structure_{stage}.pt'
    info=json.loads(path.with_suffix('.json').read_text())
    if sha(path)!=info['checkpoint_sha256']:raise ValueError('Checkpoint checksum mismatch')
    if sha(PROTOCOL_FILE)!=info['protocol_sha256']:raise ValueError('Original protocol checksum mismatch')
    model=make(seed=seed);model.load_state_dict(torch.load(path,weights_only=True));model.eval()
    return {'model':model,'name':'quantum_structure','seed':seed,'stage':stage,'parameters':counts(model)}

def score_encoded(bundle,batch):
    if set(batch)!={'streams','layout','role'} or batch['streams'].shape[-2:]!=(4,384):
        raise ValueError('Four 384D streams, layout and role required')
    batch=dict(batch);batch['streams']=torch.nn.functional.normalize(batch['streams'].double(),dim=-1)
    with torch.no_grad():return bundle['model'](batch)
