"""Scheme1's data/checkpoint locations; no historical code imports."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WWW=Path(os.environ.get('QAFFORD_WWW', '/Users/yuhanchen/Desktop/QuanLLM/WWW')).expanduser().resolve()
DATA_DIR=Path(os.environ.get('QAFFORD_DATA', str(WWW/'验证/Top50量子匹配重构/data')))
ROLE_CODES=Path(os.environ.get('QAFFORD_ROLES', str(WWW/'验证/量子结构意图条件化/role_codes.npz')))
ARTIFACT_DIR=Path(os.environ.get('QAFFORD_ARTIFACTS', str(WWW/'验证/384维三种子电路诊断优化')))
PROTOCOL_FILE=ARTIFACT_DIR/'protocol.json'
OUT=Path(os.environ.get('QAFFORD_RUN_DIR', str(ROOT.parent/'运行输出/方案1')))

def protocol():
    saved=json.loads(PROTOCOL_FILE.read_text())
    keys=('train_indices','validation_indices','learning_rate_min','learning_rate_max','max_segments',
          'early_stop_min_segments','early_stop_patience_segments','early_stop_min_gain',
          'weight_decay','clip','pair_loss_weight')
    return {k:saved[k] for k in keys}
