"""MolFormer token cache for the 380 Tahoe-100M drugs.

What is stored per drug
-----------------------
The **whole token sequence** of the SMILES, minus the special tokens, plus the
positions of the atom tokens inside it::

    tokens      : (T, 768)   every real token: atoms and the structure tokens
                             ("=", "(", ")", "1", ".", "#", ...)
    atom_positions : [int]   which rows of ``tokens`` are atoms, in RDKit atom order

Two ways to feed it to the model, chosen at collate time:

``mode="all"`` (default)
    every token becomes a key/value in the stage-1 cross-attention. The encoder
    already folded the whole molecule into each token, so this is a strict
    superset of the atom-only view: bond order and ring closure reach the model
    even when they barely move the atom states (ethanol ``CCO`` vs acetaldehyde
    ``CC=O``: atom cosines 0.9998, i.e. atom-only cannot tell them apart).
    Cost: median 57 keys per drug vs 30, i.e. ~2x a term that is already ~870x
    cheaper than the gene self-attention (2.53e9 pairs).

``mode="atoms"``
    only the atom rows. Same number of drugs, fewer keys, and attention weights
    that map 1:1 onto RDKit atoms (used by ``testing/atom_*.py``).

Why cache at all: the encoder is 44.4 M params and frozen in ``eval()``, so its
output is bit-identical every call. The trainer would otherwise run it 480 times
per epoch; the cache makes it a one-off (~2 min, ~66 MB on disk).

Atom/token rule: a token is an atom iff it is one of the 16 bare organic-subset
symbols or a bracketed atom ``[...]``. Validated against
``Chem.MolFromSmiles(s).GetNumAtoms()`` on all 380 library SMILES: 0 mismatches,
and the atom tokens appear in RDKit atom order.

Build once::

    python models/molformer_atoms.py --out models/cache/atom_embeddings.pt

``train.py`` only ever reads the ``.pt`` file.
"""

from __future__ import annotations

import argparse
import math
import os

import torch

MODEL_NAME = "ibm/MolFormer-XL-both-10pct"
EMBED_DIM = 768
DOSE_EPS = 1e-3
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cache", "atom_embeddings.pt"
)
DEFAULT_CSV = os.environ.get(
    "TAHOE_DRUG_CSV", os.path.join(_REPO_ROOT, "Chemcial_info", "molformer_embeddings.csv")
)

# The 16 bare symbols of the SMILES organic subset; every other atom is written
# as a bracket atom.
BARE_ATOMS = frozenset(
    {"B", "Br", "C", "Cl", "F", "I", "N", "O", "P", "S", "b", "c", "n", "o", "p", "s"}
)


def dose_to_scalar(concentration: float) -> float:
    """``log10(conc + 1e-3)`` -- the four Tahoe doses are 0, 0.05, 0.5, 5 uM.

    = -3.000, -1.292, -0.300, 0.699. The epsilon keeps DMSO (0) finite and keeps
    the log monotone over the whole range.
    """
    return float(math.log10(float(concentration) + DOSE_EPS))


def is_atom_token(token: str) -> bool:
    return token in BARE_ATOMS or (token.startswith("[") and token.endswith("]"))


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def _install_transformers_shim() -> None:
    """The pinned ``ibm/MolFormer-XL-both-10pct`` revision imports a symbol that
    only exists in transformers >= 4.58; the local env is 4.57."""
    import transformers.masking_utils as masking_utils

    if hasattr(masking_utils, "create_bidirectional_mask"):
        return

    def create_bidirectional_mask(config=None, inputs_embeds=None, attention_mask=None, **_):
        mask = attention_mask[:, None, None, :].to(inputs_embeds.dtype)
        return (1.0 - mask) * torch.finfo(inputs_embeds.dtype).min

    masking_utils.create_bidirectional_mask = create_bidirectional_mask


def build_atom_embeddings(
    csv_path: str,
    out_path: str = DEFAULT_CACHE,
    batch_size: int = 32,
    device: str | None = None,
) -> dict:
    """SMILES -> token states, saved as ``{drug: (T, 768)}`` + atom positions."""
    import pandas as pd

    _install_transformers_shim()
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, deterministic_eval=True)
        .to(device)
        .eval()
    )
    specials = set(tokenizer.all_special_tokens)

    frame = pd.read_csv(csv_path)
    drugs = frame["drug"].astype(str).tolist()
    smiles = frame["smiles"].astype(str).tolist()

    tokens: dict[str, torch.Tensor] = {}
    atom_positions: dict[str, list[int]] = {}
    mismatch = 0

    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            chunk = smiles[start : start + batch_size]
            encoded = tokenizer(chunk, padding=True, return_tensors="pt")
            encoded.pop("token_type_ids", None)  # MolformerModel.forward() rejects it
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = model(**encoded).last_hidden_state.float()  # (B, T, 768)

            for offset, smiles_string in enumerate(chunk):
                length = int(encoded["attention_mask"][offset].sum())
                names = tokenizer.convert_ids_to_tokens(
                    encoded["input_ids"][offset][:length]
                )
                keep = [i for i, name in enumerate(names) if name not in specials]
                if not keep:
                    raise ValueError(f"only special tokens in {smiles_string!r}")

                kept_names = [names[i] for i in keep]
                positions = [k for k, name in enumerate(kept_names) if is_atom_token(name)]
                if not positions:
                    raise ValueError(f"no atoms in {smiles_string!r}")

                drug = drugs[start + offset]
                tokens[drug] = hidden[offset][torch.tensor(keep, device=device)].cpu()
                atom_positions[drug] = positions

                from rdkit import Chem  # local: only needed for this sanity check

                mol = Chem.MolFromSmiles(smiles_string)
                if mol is None or mol.GetNumAtoms() != len(positions):
                    mismatch += 1

    lengths = torch.tensor([v.shape[0] for v in tokens.values()])
    counts = torch.tensor([len(v) for v in atom_positions.values()])
    payload = {
        "model": MODEL_NAME,
        "dim": EMBED_DIM,
        "dose_eps": DOSE_EPS,
        "atom_rule": "bare organic subset or bracket atom",
        "tokens": tokens,
        "atom_positions": atom_positions,
        "n_special_dropped": True,
        "smiles": dict(zip(drugs, smiles)),
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(payload, out_path)

    print(f"wrote {out_path}")
    print(f"  drugs        : {len(tokens)}")
    print(f"  tokens/drug  : min {int(lengths.min())}  median {int(lengths.median())} "
          f"  max {int(lengths.max())}   (special tokens removed)")
    print(f"  atoms/drug   : min {int(counts.min())}  median {int(counts.median())} "
          f"  max {int(counts.max())}")
    print(f"  atom-position mismatches vs RDKit: {mismatch}")
    print(f"  file size    : {os.path.getsize(out_path) / 1e6:.0f} MB")
    return payload


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
class AtomEmbeddingStore:
    """MolFormer token sequences per drug + collation to ``(B, N, 769)``.

    ``mode="all"`` uses every token, ``mode="atoms"`` uses only atom rows.
    """

    def __init__(self, path: str = DEFAULT_CACHE):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- build it once with:\n"
                f"    python models/molformer_atoms.py --out {path}"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "tokens" not in payload:
            raise RuntimeError(
                f"{path} is an old atoms-only cache. Rebuild it:\n"
                f"    python models/molformer_atoms.py --out {path}"
            )
        self.dim: int = int(payload["dim"])
        self.tokens: dict[str, torch.Tensor] = payload["tokens"]
        self.atom_positions: dict[str, list[int]] = payload["atom_positions"]
        self.smiles: dict[str, str] = payload.get("smiles", {})

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, drug: str) -> bool:
        return drug in self.tokens

    def get(self, drug: str, mode: str = "atoms") -> torch.Tensor:
        """(N, 768) for one drug; ``mode="atoms"`` keeps RDKit atom alignment."""
        train = self.tokens[drug]
        if mode == "all":
            return train
        if mode == "atoms":
            return train[torch.tensor(self.atom_positions[drug])]
        raise ValueError(f"mode must be 'all' or 'atoms', got {mode!r}")

    def n_atoms(self, drug: str) -> int:
        return len(self.atom_positions[drug])

    def n_tokens(self, drug: str) -> int:
        return int(self.tokens[drug].shape[0])

    def collate(
        self,
        drugs: list[str],
        concentrations: list[float],
        device: torch.device | str | None = None,
        mode: str = "all",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad a batch of drugs to a common length.

        Returns ``chem`` (B, N, 769) -- the 769th channel is the log10 dose -- and
        ``chem_mask`` (B, 1, 1, N) bool, True where a real token/atom exists.
        """
        vectors = [self.get(d, mode=mode) for d in drugs]
        widths = [int(v.shape[0]) for v in vectors]
        width = max(widths)
        batch = len(vectors)

        chem = torch.zeros(batch, width, self.dim + 1, dtype=torch.float32)
        mask = torch.zeros(batch, 1, 1, width, dtype=torch.bool)

        for i, (vectors_i, dose) in enumerate(zip(vectors, concentrations)):
            length = vectors_i.shape[0]
            chem[i, :length, : self.dim] = vectors_i
            chem[i, :length, self.dim] = dose_to_scalar(dose)
            mask[i, 0, 0, :length] = True

        if device is not None:
            chem = chem.to(device)
            mask = mask.to(device)
        return chem, mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="drug/smiles table of the 380-drug library",
    )
    parser.add_argument("--out", default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    build_atom_embeddings(args.csv, args.out, args.batch_size, args.device)


if __name__ == "__main__":
    main()
