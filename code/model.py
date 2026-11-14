"""Scheme1: the 49,361-parameter role/structure quantum ranker only."""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from quantum_ops import rotations, relation_features, phase_channels, overlaps, PAIRS

ROOT=Path(__file__).resolve().parent

class _SemanticQuantumCore(nn.Module):

    def __init__(self, dimension=384):
        super().__init__()
        self.dimension = dimension
        self.ncoord = math.ceil(math.log2(dimension))
        self.size = 2 ** self.ncoord
        self.input_dim = 4 * dimension + 26
        self.groups = 8
        self.readout_dim = 4 * self.size + 288 + self.ncoord + 6
        self.local = nn.Parameter(torch.zeros(4, self.ncoord, 3, dtype=torch.float64))
        self.control = nn.Parameter(torch.zeros(3, 2, 5, self.input_dim, dtype=torch.float64))
        self.readout = nn.Parameter(torch.zeros(self.readout_dim, dtype=torch.float64))
        with torch.no_grad():
            self.readout[:self.size] = 8.0
            self.readout[self.size:2 * self.size] = 0.125
            self.readout[3 * self.size:4 * self.size] = 0.125
        ix = torch.arange(self.size)
        perm = ix.clone()
        for q in range(self.ncoord):
            perm = perm ^ (perm >> self.ncoord - 1 - q & 1) << self.ncoord - 1 - (q + 1) % self.ncoord
        self.register_buffer('ring_inverse', torch.argsort(perm), persistent=False)
        z = (1 - 2 * (ix[:, None] >> torch.arange(self.ncoord - 1, -1, -1) & 1)).double()
        self.register_buffer('coordinate_z', z, persistent=False)
        self.register_buffer('phase_coordinate_z', 0.5 * z[:dimension].mean(0), persistent=False)
        self.register_buffer('source_z', torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]], dtype=torch.float64), persistent=False)
        self.register_buffer('channel_z', (1 - 2 * (torch.arange(8)[:, None] >> torch.arange(2, -1, -1) & 1)).double(), persistent=False)

    def local_state_factor(self, streams):
        x = F.pad(streams.reshape(-1, 4, self.dimension).double(), (0, self.size - self.dimension)).to(torch.complex128) / 2
        n = len(x)
        for q in range(self.ncoord):
            angle = self.local[:, q]
            u = rotations('Z', angle[:, 2]) @ rotations('Y', angle[:, 1]) @ rotations('X', angle[:, 0])
            x = x.reshape(n, 4, 2 ** q, 2, 2 ** (self.ncoord - q - 1))
            x = torch.einsum('sij,nsljr->nslir', u, x).reshape(n, 4, self.size)
        return x[:, :, self.ring_inverse]

    def local_state(self, streams):
        return self.local_state_dense(streams)

    def local_state_dense(self, streams):
        x = F.pad(streams.reshape(-1, 4, self.dimension).double(), (0, self.size - self.dimension)) / 2
        u = torch.ones(4, 1, 1, dtype=torch.complex128, device=x.device)
        for q in range(self.ncoord):
            angle = self.local[:, q]
            a = rotations('Z', angle[:, 2]) @ rotations('Y', angle[:, 1]) @ rotations('X', angle[:, 0])
            u = torch.einsum('sab,scd->sacbd', u, a).reshape(4, 2 ** (q + 1), 2 ** (q + 1))
        u = u[:, self.ring_inverse]
        return torch.complex(torch.einsum('sij,nsj->nsi', u.real, x), torch.einsum('sij,nsj->nsi', u.imag, x))

    def unitaries(self, batch):
        x = relation_features(batch).reshape(-1, self.input_dim)
        angles = math.pi / 4 * torch.tanh(x @ self.control.reshape(30, self.input_dim).T).reshape(-1, 3, 2, 5)
        u = torch.eye(4, dtype=torch.complex128, device=x.device)[None, None].expand(len(x), 3, 4, 4)
        eye = torch.eye(2, dtype=torch.complex128, device=x.device)
        generator = torch.kron(torch.diag(torch.tensor([1.0, -1.0], device=x.device)).to(torch.complex128), torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=x.device))
        for layer in range(2):
            for k, (q, axis) in enumerate(((0, 'Y'), (0, 'Z'), (1, 'Y'), (1, 'Z'))):
                r = rotations(axis, angles[:, :, layer, k])
                gate = torch.einsum('...ab,cd->...acbd', r, eye).reshape(-1, 3, 4, 4) if q == 0 else torch.einsum('ab,...cd->...acbd', eye, r).reshape(-1, 3, 4, 4)
                u = gate @ u
            a = angles[:, :, layer, 4]
            gate = (a / 2).cos()[..., None, None] * torch.eye(4, dtype=u.dtype, device=u.device) - 1j * (a / 2).sin()[..., None, None] * generator
            u = gate @ u
        return u

    def phase_gram(self, batch, u):
        overlap = batch.get('overlaps')
        if overlap is None:
            overlap = overlaps(batch['streams'])
        x = overlap.reshape(-1, 6, 3, 8).double()
        g = torch.eye(4, dtype=torch.complex128, device=x.device)[None, None, None].expand(len(x), 3, 8, 4, 4).clone()
        for k, (a, b) in enumerate(PAIRS):
            g[:, :, :, a, b] = x[:, k]
            g[:, :, :, b, a] = x[:, k]
        return u[:, :, None] @ g @ u[:, :, None].mH

    def features(self, batch):
        u = self.unitaries(batch)
        raw = u[:, 0] @ self.local_state(batch['streams'])
        g = self.phase_gram(batch, u)
        raw_obs = []
        for a, b in ((0, 1), (0, 2)):
            z = 4 * raw[:, a].conj() * raw[:, b]
            raw_obs.extend((z.real, z.imag))
        phase = []
        for a, b in PAIRS:
            phase.extend((g[:, :, :, a, b].real.flatten(1), -g[:, :, :, a, b].imag.flatten(1)))
        source_prob = 0.5 * raw.abs().square().sum(-1) + torch.diagonal(g, dim1=-2, dim2=-1).real.sum((1, 2)) / 192
        source = source_prob @ self.source_z
        coordinate = 0.5 * raw.abs().square().sum(1) @ self.coordinate_z + self.phase_coordinate_z
        channels = batch.get('channel_expectations')
        if channels is None:
            p = phase_channels(batch['streams'].reshape(-1, 4, self.dimension).double()).square().sum((1, 4)) / 24
            channels = torch.cat((p.new_full((len(p), 1), 0.5), p.flatten(1), p.new_zeros((len(p), 1))), -1) @ self.channel_z
        else:
            channels = channels.reshape(-1, 3).double()
        return torch.cat((*raw_obs, *phase, source[:, :2], channels, coordinate, source[:, 2:]), -1)

    def forward(self, batch):
        return (self.features(batch) @ self.readout).reshape(batch['streams'].shape[:-2])

    def joint_state(self, batch):
        streams = batch['streams'].reshape(-1, 4, self.dimension).double()
        u = self.unitaries(batch)
        state = torch.zeros(len(streams), 4, 8, self.size, dtype=torch.complex128, device=streams.device)
        state[:, :, 0] = u[:, 0] @ self.local_state(streams) * math.sqrt(0.5)
        p = phase_channels(streams).to(torch.complex128) * math.sqrt(1 / 6) / 2
        changed = torch.einsum('nfab,nbfci->nafci', u, p).reshape(-1, 4, 6, self.dimension)
        state[:, :, 1:7, :self.dimension] = changed
        return state.reshape(-1, 32 * self.size)

    def native_features(self, batch):
        v = self.joint_state(batch).reshape(-1, 4, 8, self.size)
        raw = v[:, :, 0] / math.sqrt(0.5)
        features = []
        for a, b in ((0, 1), (0, 2)):
            z = 4 * raw[:, a].conj() * raw[:, b]
            features.extend((z.real, z.imag))
        for a, b in PAIRS:
            z = 192 * (v[:, a, 1:7, :self.dimension].conj() * v[:, b, 1:7, :self.dimension]).reshape(-1, 3, 2, 8, self.dimension // 8).sum((2, 4))
            features.extend((z.real.flatten(1), z.imag.flatten(1)))
        prob = v.abs().square()
        source = prob.sum((2, 3)) @ self.source_z
        channels = prob.sum((1, 3)) @ self.channel_z
        coord = prob.sum((1, 2)) @ self.coordinate_z
        return torch.cat((*features, source[:, :2], channels, coord, source[:, 2:]), -1)

class _SharedSemanticState(_SemanticQuantumCore):

    def local_state(self, streams):
        if streams.ndim != 4:
            return self.local_state_dense(streams)
        b, k = streams.shape[:2]
        if not (torch.equal(streams[:, :, 0], streams[:, :1, 0].expand(-1, k, -1)) and torch.equal(streams[:, :, 3], streams[:, :1, 3].expand(-1, k, -1))):
            return self.local_state_dense(streams)
        u = torch.ones(4, 1, 1, dtype=torch.complex128, device=streams.device)
        for q in range(self.ncoord):
            a = self.local[:, q]
            r = rotations('Z', a[:, 2]) @ rotations('Y', a[:, 1]) @ rotations('X', a[:, 0])
            u = torch.einsum('sab,scd->sacbd', u, r).reshape(4, 2 ** (q + 1), 2 ** (q + 1))
        u = u[:, self.ring_inverse]
        parts = []
        for s in range(4):
            x = streams[:, 0, s] if s in (0, 3) else streams[:, :, s].reshape(-1, self.dimension)
            x = F.pad(x.double(), (0, self.size - self.dimension)) / 2
            value = torch.complex(x @ u[s].real.T, x @ u[s].imag.T)
            parts.append(value[:, None].expand(-1, k, -1).reshape(-1, self.size) if s in (0, 3) else value)
        return torch.stack(parts, 1)

def apply_four(state, gate, q):
    n = len(state)
    x = state.reshape(n, 2 ** q, 2, 2 ** (3 - q))
    return torch.einsum('nij,nljr->nlir', gate, x).reshape(n, 16)

class StructuredQuantum(_SharedSemanticState):

    def __init__(self):
        super().__init__(384)
        self.structure_angles = nn.Parameter(torch.zeros(2, 4, 3, dtype=torch.float64))
        self.cross_angles = nn.Parameter(torch.zeros(2, 4, dtype=torch.float64))
        self.base_readout_dim = self.readout_dim
        self.readout_dim += 10
        old = self.readout.detach()
        self.readout = nn.Parameter(torch.cat((old, old.new_zeros(10))))
        ix = torch.arange(16)
        z = (1 - 2 * (ix[:, None] >> torch.arange(3, -1, -1) & 1)).double()
        signs = torch.cat((z, torch.stack([z[:, a] * z[:, b] for a in range(4) for b in range(a + 1, 4)], -1)), -1)
        perm = ix.clone()
        for q in range(4):
            perm = perm ^ (perm >> 3 - q & 1) << 3 - (q + 1) % 4
        self.register_buffer('structure_z', z, persistent=False)
        self.register_buffer('structure_observables', signs, persistent=False)
        self.register_buffer('structure_ring_inverse', torch.argsort(perm), persistent=False)

    def structure_state(self, batch):
        layout = batch['layout'].reshape(-1, 10).double()
        n = len(layout)
        state = F.one_hot(batch['role'].reshape(-1), 16).to(torch.complex128)
        for q in range(4):
            state = apply_four(state, rotations('Y', math.pi / 2 + math.pi / 4 * layout[:, 4 + q]), q)
        for q in range(4):
            state = apply_four(state, rotations('Z', math.pi / 4 * layout[:, q]), q)
        for q, j in enumerate((8, 9, 0, 1)):
            state = apply_four(state, rotations('Y', math.pi / 4 * layout[:, j]), q)
        for layer in range(2):
            for q in range(4):
                for k, axis in enumerate(('X', 'Y', 'Z')):
                    state = apply_four(state, rotations(axis, self.structure_angles[layer, q, k].expand(n)), q)
            state = state[:, self.structure_ring_inverse]
        return state

    def coupling(self, probability):
        z = self.source_z[:, :2]
        difference = (z[:, None] - z[None, :]) / 2
        phase = torch.einsum('abq,qj,kj->abk', difference, self.cross_angles, self.structure_z)
        return torch.einsum('nk,abk->nab', probability.to(torch.complex128), torch.exp(1j * phase))

    def features(self, batch):
        base = super().features(batch)
        structure = self.structure_state(batch)
        prob = structure.abs().square()
        m = self.coupling(prob)
        out = []
        for j, (a, b) in enumerate(((0, 1), (0, 2))):
            start = j * 2 * self.size
            z = torch.complex(base[:, start:start + self.size], base[:, start + self.size:start + 2 * self.size]) * m[:, a, b, None]
            out.extend((z.real, z.imag))
        for j, (a, b) in enumerate(PAIRS):
            start = 4 * self.size + j * 48
            z = torch.complex(base[:, start:start + 24], base[:, start + 24:start + 48]) * m[:, a, b, None]
            out.extend((z.real, z.imag))
        return torch.cat((*out, base[:, 4 * self.size + 288:], prob @ self.structure_observables), -1)

    def joint_state(self, batch):
        semantic = _SemanticQuantumCore.joint_state(self, batch).reshape(-1, 4, 8, self.size)
        structure = self.structure_state(batch)
        phase = torch.einsum('sq,qj,aj->sa', self.source_z[:, :2], self.cross_angles, self.structure_z)
        return (semantic[..., None] * structure[:, None, None, None, :] * torch.exp(-0.5j * phase)[None, :, None, None, :]).reshape(-1, 32 * self.size * 16)

    def native_features(self, batch):
        v = self.joint_state(batch).reshape(-1, 4, 8, self.size, 16)
        raw = v[:, :, 0] / math.sqrt(0.5)
        out = []
        for a, b in ((0, 1), (0, 2)):
            z = 4 * (raw[:, a].conj() * raw[:, b]).sum(-1)
            out.extend((z.real, z.imag))
        for a, b in PAIRS:
            z = 192 * (v[:, a, 1:7, :self.dimension].conj() * v[:, b, 1:7, :self.dimension]).sum(-1).reshape(-1, 3, 2, 8, self.dimension // 8).sum((2, 4))
            out.extend((z.real.flatten(1), z.imag.flatten(1)))
        p = v.abs().square()
        source = p.sum((2, 3, 4)) @ self.source_z
        channel = p.sum((1, 3, 4)) @ self.channel_z
        coord = p.sum((1, 2, 4)) @ self.coordinate_z
        structure = p.sum((1, 2, 3)) @ self.structure_observables
        return torch.cat((*out, source[:, :2], channel, coord, source[:, 2:], structure), -1)

def make(name='quantum_structure', seed=0):
    if name!='quantum_structure':raise ValueError('This folder contains Scheme1 only')
    torch.manual_seed(seed)
    return StructuredQuantum()

def counts(model):
    return {'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total':sum(p.numel() for p in model.parameters())}
