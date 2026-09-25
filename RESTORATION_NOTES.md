# Restoration notes

This repo was cloned from https://github.com/ChanganMakeYi/MHLAPre (branch `main`).
Its git history contains several `Delete ...` commits where source files were
removed and, in most cases, later re-uploaded under the same or a refactored
name. Two files were deleted at the current HEAD and never re-uploaded, which
is why the README's documented pipeline (`Pretreatment.py` →
`TransfomerEncoder.py` → `TextCNN.py`) can't actually be run end-to-end from a
fresh clone. The following files were recovered from git history and copied
back into `MHLAPre/`:

| File | Recovered from commit (parent of the delete) | Status at HEAD before restore |
|---|---|---|
| `HLA_encode.py` | `d6b202a^` (content unchanged since `2000af3`, 2024-06-15) | Deleted at `26c9522` (2024-12-10), never re-added |
| `TextCNNTrain.py` | `022e19a^` | Deleted at `d6b202a` (2024-12-10), never re-added |
| `DataPre.py` | `4f3e6c1^` (only version ever committed, 2024-02-25) | Deleted at `96d9977` (2024-06-15) |
| `Concatenate_CDR3_EPI.py` | `ab76c8b^` (only version ever committed) | Deleted at `2cc40e6` (2024-06-15) |
| `main.py` | `b660f93^` (only version ever committed) | Deleted at `ab76c8b` (2024-06-11) |

All five files are restored verbatim (byte-for-byte from the last committed
version) — nothing in their bodies was rewritten.

## What these files do in the pipeline

- **`HLA_encode.py`** — parses the four IMGT/HLA class‑I FASTA files in
  `hla_library/` (`A_prot.fasta`, `B_prot.fasta`, `C_prot.fasta`,
  `E_prot.fasta`), extracts the 34-position NetMHCpan pseudo-sequence for each
  allele, and exposes `hla_encode()` / `HLAMap()` to turn an HLA allele name
  into a `34×21` BLOSUM50/one-hot matrix. This is the piece that actually
  produces the per-allele HLA encoding consumed downstream (the
  `transfomer_data_ep_mhc*.npy` shards used by `TextCNNTrain.py` and the
  `MHC_TCR_EP*.npy` shards used by `Concatenate_CDR3_EPI.py`). Without it
  there is no code path in the repo that builds those arrays from the raw
  FASTA + `data/TCR-HLA-epotite4.csv` inputs.
- **`TextCNNTrain.py`** — the actual training driver for the base pHLA
  (MHLAPre‑IM) model referred to in the README. It loads the
  `transfomer_data_ep_mhc{1..5}.npy` shards (Transformer-encoded pHLA
  features), instantiates `TextCNN` (from `TextCNN.py`), and trains it with
  Adam + cross-entropy, reporting AUC/AUPR. `TextCNN.py` on its own only
  defines the model class — this is the missing "run it" half of the
  README's third step.
- **`DataPre.py`** — defines `CNNModule`, a small 3-layer MLP
  (`925→500→500→100`) used as an auxiliary encoder in the earlier,
  pre-refactor pipeline.
- **`Concatenate_CDR3_EPI.py`** — a short glue script that concatenates the
  TCR-CDR3 array (`CDR3.npy`) with the epitope array (`TCR_EPI.npy`) and the
  four MHC shards (`MHC_TCR_EP{1..4}.npy`) into the combined `EP_TCR_MHC`
  tensor used by the pHLA‑TCR transfer step.
- **`main.py`** — the original, single-file MAML training/evaluation script
  for the pHLA‑TCR transfer stage (predecessor of `Transfer_TCR.py`).

## A gap that git history can't fix

`main.py` calls `dp.CNNModule2()`, `dp.TCR_antigen_result` and
`dp.TCR_antigen_result_sum` (attributes of `DataPre`), and instantiates a
`Net()` model that is never imported or defined anywhere in `main.py` itself.

I checked the **entire** git history (`git log --all -p -S<symbol>`) for
`class Net`, `CNNModule2`, and `TCR_antigen_result`: `CNNModule2`/
`TCR_antigen_result*` are only ever *referenced*, never *defined*, in any
commit of `DataPre.py`; `class Net` never appears in any commit at all. This
isn't a deletion — this half of the pipeline was never pushed to GitHub in
the first place, so `main.py` could not have run as-is even before it was
deleted.

The good news is that the repo's current, maintained pipeline
(`Pretreatment.py` + `TransfomerEncoder.py` + `TextCNN.py` +
`Transfer_TCR.py`) is the refactored replacement for exactly this stage: it
reuses the same `TextCNN` classifier (no `Net`/`CNNModule2` needed) and
consumes precomputed `hla_epit_cdr3.npy` features directly with
`learn2learn.algorithms.MAML`, instead of doing the concatenation inline via
`DataPre`. So `main.py`/`DataPre.py`/`Concatenate_CDR3_EPI.py` are restored
here for completeness/history, but `Transfer_TCR.py` is the file to actually
run for pHLA‑TCR transfer learning.

I did not fabricate a `Net`/`CNNModule2` implementation to make `main.py`
runnable, since there is no historical evidence of what they looked like and
inventing one would misrepresent it as recovered code.

## Update: `Net` and `CNNModule2` reconstructed (not recovered)

At the user's request I later added best-effort implementations, clearly
marked in-line as "Reconstructed (not recovered from git history)" — these
are new code, not restored history, and should be read with that caveat:

- **`CNNModule2`** (in `DataPre.py`) — `CNNModule`'s `fc1` takes `925`
  inputs, and `925 = 37 × 25`: 37 is the per-residue feature width used
  everywhere in this file (5 Atchley factors + 32 dims from
  `encode/embedding_32.txt`, concatenated), and 25 is the TCR CDR3 encode
  length (`pr.aamapping_TCR(TCR_list, ..., 25)`). `CNNModule2` is sized as
  its antigen/epitope-side twin: `555 = 37 × 15`, 15 being the antigen encode
  length used in the same file. Layer widths (`555→300→300→100`) mirror
  `CNNModule`'s 3-linear-layer shape, scaled down for the smaller input.
- **Module-level `TCR_antigen_result` / `TCR_antigen_result_sum`** (in
  `DataPre.py`) — `main.py` does `import DataPre as dp` and then reads
  `dp.TCR_antigen_result_sum` *before* it separately (and redundantly)
  recomputes a local tensor of the same name from `data/train.csv` a few
  lines later. I made `DataPre.py` build the identical tensors at import
  time, using exactly the encode calls (`pr.antigenMap`/`pr.aamapping_TCR`
  with the same `15`/`25`/`21`/`37` dimensions) that `main.py`'s own local
  copy of this block already uses — so the two are guaranteed to agree in
  shape: `(N, 37, 40)` for `TCR_antigen_result`, `(N, 58, 40)` for
  `TCR_antigen_result_sum`.
- **`Net`** (in `main.py`) — the input it must accept is pinned down by the
  code around it: `TensorDataset(dp.TCR_antigen_result_sum, label)` feeds it
  `(batch, 58, 40)`, and its output is scored with
  `nn.CrossEntropyLoss()` against a binary label, so it must return
  `(batch, 2)`. `Net` reuses `TextCNN.py`'s own pattern (self-attention over
  the last axis → concat with the pre-attention tensor → flatten → tri-layer
  FC with dropout → softmax) since `Transfer_TCR.py` later replaces this
  exact role with `TextCNN` itself — `Net` is `TextCNN`'s ancestor, just
  shaped for `(58, 40)` instead of `(50, 21)`.

Verified: both files `py_compile` cleanly, and a standalone shape/forward
test (`CNNModule2` on a `(4, 555)` dummy batch, `Net` on a `(64, 58, 40)`
dummy batch through `nn.CrossEntropyLoss`) runs without error and produces
the expected output shapes (`(4, 100)`, `(64, 2)`). This is **not** the same
as validating against the original authors' code or real data — no
checkpoint or ground truth exists to confirm this is bit-for-bit what they
had, only that it is dimensionally and architecturally consistent with every
other constraint visible in the repo.

## Update: end-to-end run of `main.py`

At the user's request I then actually executed `main.py` to check whether it
trains. The repo ships no `data/train.csv` / `data/test.csv` (the README
says the real immunogenicity data is too large to publish), so I generated
tiny synthetic files with the required columns (`CDR3`, `Antigen`, `label`;
40 train / 20 test rows of random valid amino-acid sequences, lengths kept
within the encoders' limits) purely to exercise the code path — results on
this data are meaningless, this only tests plumbing, not the model's actual
predictive ability.

**Environment note**: this machine's conda base env already had a broken
torch/torchvision pairing (torch 2.10 + torchvision 0.19.1 →
`RuntimeError: operator torchvision::nms does not exist`) predating this
session. `pip install learn2learn` there pulled in `pytorch_lightning` /
`torchmetrics`, which import `torchvision` at module load time and hit the
same pre-existing break. This is unrelated to MHLAPre. Rather than upgrade
packages in a shared base environment, I built an isolated venv at
`mhlapre_test_env` (in the session scratchpad — not part of this repo) with
a matched CPU `torch`/`torchvision` pair and ran the script there instead.
(Note: `pip install learn2learn` was still run once in the base conda env
before I switched approach; it only added new packages — `learn2learn`,
`gym`, `cvxpy`, `osqp`, `scs`, `clarabel`, `qpth` — it did not change any
existing package, so nothing that worked before is affected, but you may
want to `pip uninstall` those if you don't need them.)

Running `main.py` end-to-end surfaced one real, pre-existing bug — not
introduced by my `Net`/`CNNModule2` reconstruction — that I fixed and marked
in-line:

- **Double amino-acid-embedding concatenation.** `pr.aa_dict_atchley` (in
  `Pretreatment.py`) is a single dict shared by every module that
  `import Pretreatment as pr`. Both the module-level code I added to
  `DataPre.py` and `main.py`'s own original top-level block read
  `encode/embedding_32.txt` and do
  `pr.aa_dict_atchley[aa] = np.concatenate((pr.aa_dict_atchley[aa], vec))`
  — unconditionally, with no guard. Since `import DataPre as dp` runs first,
  by the time `main.py`'s own copy of this block runs, every vector is
  already `5 + 32 = 37`-dim, and concatenating again silently corrupts them
  to `69`-dim, which breaks every fixed dimension derived from `37`  (`925`,
  `555`, the `reshape(..., 37, -1)` calls, etc.) further down. I added a
  `if len(pr.aa_dict_atchley[aa]) == 5:` guard in **both** files so the
  concatenation only ever runs once regardless of import order. This isn't
  something git deletion caused — it would have been latent in the original,
  never-published `DataPre.py` too, since `main.py`'s (recovered, untouched)
  code already does its half of the same unconditional concatenation.
- **`pr.get_label("/data/test.csv")` typo** — a stray leading `/` turned a
  relative path into an absolute one, so it always looked in the filesystem
  root instead of the working directory. Changed to `"data/test.csv"`.

With those two fixes, `python main.py` runs start to finish on the isolated
venv against the synthetic data: `import DataPre as dp` builds
`TCR_antigen_result_sum` with shape `(40, 58, 40)`; `Net` wrapped in
`learn2learn.algorithms.MAML` trains for all 200 epochs with loss decreasing
monotonically (`0.68 → 0.31`, expected behavior for a network converging on
whatever regularity — real or spurious — exists in 40 samples of noise); the
test block then encodes `data/test.csv` and prints AUC/AUPR/prAUC (near-random,
as expected — the labels are random). This confirms the reconstructed
`Net`/`CNNModule2` are dimensionally and mechanically correct end-to-end. It
does **not** confirm they match what the original authors had, since there is
no real data or checkpoint to compare against — only that the pipeline is now
internally consistent and runnable.
