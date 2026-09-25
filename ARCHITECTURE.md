# MHLAPre architecture

Paper: Chen et al., *"Meta learning for mutant HLA class I epitope
immunogenicity prediction to accelerate cancer clinical immunotherapy"*,
**Briefings in Bioinformatics**, 26(1):bbae625, 2024.
DOI/links: https://academic.oup.com/bib/article/26/1/bbae625/7916535 ,
full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11630330/

## 1. Problem framing

Two related but distinct tasks are conflated in most prior pHLA tools: (a)
peptide–HLA **binding/presentation** (does the peptide get loaded and
displayed by this HLA allele?) and (b) **immunogenicity** (does the displayed
peptide actually trigger a CD8+ T‑cell response through TCR recognition?).
Existing models are trained mostly on MS-eluted-ligand data, which is
dominated by non-immunogenic peptides, so they are good at (a) but poor at
(b). MHLAPre is built specifically around immunogenic ligandome data, then
transfers to the pHLA‑TCR interaction to model (b) directly.

## 2. Two-stage model family

- **MHLAPre‑IM** — trained on immunogenicity-labelled pHLA data (IEDB,
  47,810 samples after cleaning, 146 HLA alleles, 11,159 positive /
  36,651 negative, peptide length 8–15aa). This is the base
  allele-specific / pan-allelic presentation+immunogenicity model.
- **MHLAPre‑TT** — MHLAPre‑IM's weights are transferred and fine-tuned on
  pHLA‑TCR data (33,517 pairs after filtering, 32 alleles, 49,988
  experimentally confirmed negatives — not decoys) to model actual TCR
  recognition of the pHLA complex, i.e. immunogenicity in the strict
  immune-response sense.

## 3. Input encoding

All inputs are folded into one fixed-size matrix per sample, `50 × 21`:

| Segment | Rows | Encoding |
|---|---|---|
| Peptide/epitope | up to 15 (padded/centered to 16) | BLOSUM62, 21 columns (20 aa + 'X' padding) |
| HLA pseudo-sequence | 34 | the 34 polymorphic peptide-contact residues of the groove (NetMHCpan pseudo-sequence positions), BLOSUM62 |
| TCR CDR3 (TT stage only) | up to 30, sharing the remaining budget | BLOSUM62, same 21-column alphabet |

Code mapping:
- `HLA_encode.py` — parses `hla_library/{A,B,C,E}_prot.fasta`, extracts the
  34-residue pseudo-sequence per allele (`pseudo_seq_pos`), and encodes it via
  BLOSUM50/one-hot (`hla_encode`, `HLAMap`). (The paper text describes
  BLOSUM62; the checked-in matrix is labelled BLOSUM50 — same 21×21 shape and
  role, likely a naming holdover from an earlier iteration.)
- `Pretreatment.py` — peptide-side encoding (`peptide_encode_HLA`,
  `antigenMap`), plus the Atchley-factor alternative encoding
  (`aamapping_TCR`) used for the CDR3/peptide arrays in the TCR stage.
- `Concatenate_CDR3_EPI.py` — concatenates the encoded CDR3 array with the
  epitope array and the per-shard MHC arrays into the unified pHLA‑TCR tensor
  consumed downstream.

## 4. Transformer encoder (contextual preprocessing)

`TransfomerEncoder.py`:
- Sinusoidal positional encoding added position-wise to the `50 × 21` matrix
  (`add_position_encoding`), so the peptide/HLA(/TCR) boundary and residue
  order are visible to attention.
- 3 stacked pre-norm-style Transformer encoder layers
  (`TransformerEncoder`/`TransformerEncoderLayer`): multi-head self-attention
  (10 heads, `d_model=50`) → residual + LayerNorm → position-wise feed-forward
  (`dim_feedforward=2048`) → residual + LayerNorm.
- Input and output shape are identical (`50 × 21`) — this module produces a
  contextualized re-embedding of the raw BLOSUM-encoded sequence, not a
  dimensionality reduction. Its output is cached to disk
  (`transfomer_data_*.npy` / `transfomer_data_mhc_ep_cdr3.npy`) and reused by
  the classifier stage.

## 5. TextCNN + attention classifier

`TextCNN.py` (`TextCNN` class):
1. A `MultiheadAttention(50, 10)` layer refines the transformer output again
   (self-attention over the residue axis).
2. The attended representation is concatenated with the original
   (pre-attention) representation along the feature axis — a residual/skip
   concatenation rather than addition — then flattened.
3. Tri-layer fully-connected head: `Linear(2100→1000)` + `BatchNorm1d` + ReLU
   → dropout(0.2) → `Linear(1000→100)` + `BatchNorm1d` + ReLU →
   `Linear(100→2)` → softmax.
   (The paper describes this stage as "TextCNN": 1D conv + max-pool feature
   extraction followed by the tri-layer FC head; the convolutional branch is
   present in the file as commented-out `nn.Conv1d` blocks — the FC head
   after attention is the branch actually wired into `forward()`.)

## 6. Meta-learning (MAML) training loop

Both `TextCNNTrain.py` (MHLAPre‑IM) and `Transfer_TCR.py`/`main.py`
(MHLAPre‑TT) wrap the classifier in `learn2learn.algorithms.MAML`:

- Each peptide (or pHLA/pHLA‑TCR group) is treated as a task
  `Qᵢ = {Sᵢ, Tᵢ}` with a support set and a query set, redrawn per epoch
  ("dynamic sampling"), rather than one fixed global training set.
- **Inner loop**: `clone = maml.clone()`; forward on the support batch;
  cross-entropy loss; `clone.adapt(loss)` — a few (paper: 3) fast-weight
  gradient steps at inner learning rate.
- **Outer loop**: the loss from the adapted clone on the batch is
  backpropagated through to the meta-parameters and stepped with Adam
  (outer/meta learning rate), so the shared initialization becomes one that
  adapts quickly to any single peptide/pHLA-TCR task rather than one that
  is merely good on average.
- Reported hyperparameters (paper): Adam, initial LR 5e‑4, 350 epochs,
  batch size 128, 4:1 train/test split, inner-loop LR fast_lr (code uses
  values like `fast_lr=0.1/0.01/0.005` and `meta_lr≈5e‑5` across scripts —
  these vary slightly per script/experiment in the repo, e.g.
  `TextCNNTrain.py` doesn't actually wrap MAML around the pretraining stage
  in the version restored here — see caveat below).
- Evaluation throughout: ROC-AUC, AUPR/average-precision, PR-AUC on held-out
  pairs.

**Caveat**: the restored `TextCNNTrain.py` trains `TextCNN` directly with
plain Adam (no MAML wrapper) — the MAML loop is applied at the *transfer*
stage (`Transfer_TCR.py`/`main.py`) in the checked-in code, even though the
paper frames meta-learning as central to the base pHLA model too. This may
reflect an ablation/earlier version, or MHLAPre‑IM's meta-learning variant
living only in the unpublished, non-recoverable parts of the pipeline (see
`RESTORATION_NOTES.md`).

## 7. Transfer learning to pHLA‑TCR (MHLAPre‑TT)

`Transfer_TCR.py`:
1. Instantiate the same `TextCNN` architecture (weights are *not* explicitly
   loaded from a saved MHLAPre‑IM checkpoint in the restored script — the
   `torch.load(...)` / state-dict-loading block is present but commented
   out), so this is a transfer-*architecture* + fine-tune, with the actual
   checkpoint hand-off left as a manual step in the current repo state.
2. Feed the precomputed `hla_epit_cdr3.npy` (Transformer-encoded peptide +
   HLA + CDR3, `50×21`) through `learn2learn.algorithms.MAML(model,
   lr=fast_lr)`.
3. Train/evaluate exactly like the base stage, but on
   `data/train_paired_tcr_pmhc_data.csv` labels (`Target` column: does this
   pHLA‑TCR pair actually trigger recognition), which is the immunogenicity
   ground truth proper.

## 8. End-to-end pipeline (file → file)

```
hla_library/*.fasta ─┐
                      ├─► HLA_encode.py  ──► HLA pseudo-seq BLOSUM matrices ─┐
data/TCR-HLA-epotite4.csv ─┘                                                 │
data/10X_datasets.csv, VDJdb/McPAS csvs ─► Pretreatment.py (peptide/CDR3 enc)│
                                                                              ▼
                                              Concatenate_CDR3_EPI.py ──► combined pHLA(-TCR) tensor
                                                                              │
                                                                              ▼
                                            TransfomerEncoder.py (3-layer, 10-head, sinusoidal PE)
                                                                              │
                                                                              ▼
                              TextCNN.py (attention + tri-layer FC + softmax)
                                     │                                   │
                                     ▼                                   ▼
                     TextCNNTrain.py  (MHLAPre-IM,               Transfer_TCR.py (MHLAPre-TT,
                      pHLA binding/immunogenicity)                 MAML transfer on pHLA-TCR)
```

`main.py` / `DataPre.py` are the earlier, single-script version of the last
two boxes (predates the `Transfer_TCR.py`/`Concatenate_CDR3_EPI.py` split);
see `RESTORATION_NOTES.md` for why they can't run as committed.
