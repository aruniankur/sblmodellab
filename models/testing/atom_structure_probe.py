"""Does MolFormer actually mix *structure* into each atom vector?

If the per-atom vectors were just a learnable per-element table, then every
carbon in a molecule would have the identical vector (cosine exactly 1.000) and
similarity would not depend on where the atom sits in the graph.

This measures, per drug:
  1. within-element cosine  -- what an nn.Embedding(n_elements) would give (1.000)
  2. cosine vs graph distance -- atoms close in the molecular graph should be
     more similar than atoms far apart, which is structure mixing and nothing else
  3. same effect after removing the molecule-wide mean vector (transformer states
     share a large common component, which masks the structural part)

    python models/testing/atom_structure_probe.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.molformer_atoms import AtomEmbeddingStore, is_atom_token

RDLogger.DisableLog("rdApp.*")

DRUGS = [
    "Aspirin",
    "Gemcitabine",
    "Bestatin (hydrochloride)",
    "Paclitaxel",
    "Bortezomib",
]


def analyse(name, vectors, smiles):
    mol = Chem.MolFromSmiles(smiles)
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    n = len(elements)
    if n != vectors.shape[0]:
        print(f"  !! {name}: {n} rdkit atoms vs {vectors.shape[0]} vectors")
        return None

    # remove the molecule-wide common component to expose the structural part
    centred = vectors - vectors.mean(0, keepdim=True)

    def stats(x):
        unit = F.normalize(x, dim=-1)
        cosine = unit @ unit.T
        cosine.fill_diagonal_(float("nan"))
        return cosine

    raw = stats(vectors)
    cen = stats(centred)

    # ---- 1. within-element similarity -------------------------------- #
    within = {}
    for element in sorted(set(elements)):
        idx = [i for i, e in enumerate(elements) if e == element]
        if len(idx) < 2:
            continue
        block = raw[np.ix_(idx, idx)].numpy()
        block_c = cen[np.ix_(idx, idx)].numpy()
        off = ~np.eye(len(idx), dtype=bool)
        within[element] = (
            len(idx),
            float(np.nanmean(block[off])),
            float(np.nanmean(block_c[off])),
        )

    # ---- 2. similarity vs graph distance ----------------------------- #
    dist = Chem.GetDistanceMatrix(mol)
    by_distance = defaultdict(list)
    by_distance_c = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            d = int(dist[i, j])
            by_distance[d].append(float(raw[i, j]))
            by_distance_c[d].append(float(cen[i, j]))

    print(f"\n=== {name}  ({n} atoms) ===")
    print(f"  mean cosine between all atom pairs : raw {np.nanmean(raw.numpy()):.4f}"
          f"   centred {np.nanmean(cen.numpy()):.4f}")
    print("  within-element similarity (a learnable per-element table = 1.0000):")
    for element, (count, r, c) in within.items():
        print(f"    {element:>2} x{count:<3} raw {r:.4f}   centred {c:.4f}")
    print("  similarity vs graph distance (bonds between the two atoms):")
    for d in sorted(by_distance)[:5]:
        values = by_distance[d]
        label = f"{d} bond{'s' if d > 1 else ''}"
        print(f"    {label:>9}  n={len(values):<4} raw {np.mean(values):.4f}"
              f"   centred {np.mean(by_distance_c[d]):.4f}")
    return raw, cen, dist


def main() -> None:
    store = AtomEmbeddingStore()
    for name in DRUGS:
        if name not in store:
            print(f"{name!r} not in the cache")
            continue
        analyse(name, store.get(name), store.smiles[name])


if __name__ == "__main__":
    main()
