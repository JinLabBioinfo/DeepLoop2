# DeepLoop2

**DeepLoop2** is the next-generation chromatin contact enhancement framework developed from [DeepLoop](https://github.com/JinLabBioinfo/DeepLoop). It is designed to recover high-resolution chromatin interaction patterns from sparse Hi-C data, with improved stability across sequencing depths and strong performance at **5 kb resolution**, including low-depth datasets.

> **Beta release:** DeepLoop2 is currently released as a public beta for inference and evaluation. The training pipeline is not included in this release.

## Why DeepLoop2?

DeepLoop2 focuses on practical inference from sparse chromatin contact maps.

- **5 kb resolution:** designed and evaluated for higher-resolution contact enhancement.
- **Robust across input depths:** pretrained models are provided at multiple training depths.
- **Model selection without a gold standard:** users can compare predictions from several pretrained models at representative loci before running genome-wide inference.
- **Cooler-compatible output:** DeepLoop2 predictions can be converted to `.cool` for downstream visualization and analysis.

DeepLoop2 follows the same general philosophy as DeepLoop: systematic biases should be corrected before deep-learning-based signal enhancement. If starting from raw Hi-C reads, preprocessing and bias correction should therefore be completed before DeepLoop2 inference.

## Repository structure

```text
DeepLoop2.0/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── model/
│   ├── dense_unet.py
│   └── checkpoints/
│       └── README.md
│
├── inference/
│   └── predict.py
│
├── utils/
│   └── convert_to_cooler.py
│
└── examples/
    └── DeepLoop2_walkthrough.ipynb
```

## Quick start

For first-time users, we recommend starting with:

```text
examples/DeepLoop2_prediction_walkthrough.ipynb
```

The notebook is intended to demonstrate the complete user-facing workflow:

```text
processed contact data
        ↓
choose candidate pretrained models
        ↓
run DeepLoop2 at representative loci
        ↓
compare predicted contact maps
        ↓
select a suitable model
        ↓
run chromosome / genome-wide inference
        ↓
convert prediction to Cooler (optional)
```

## Installation

Clone the repository:

```bash
git clone https://github.com/dylan-plummer/DeepLoop2.0.git
cd DeepLoop2.0
```

Create or activate a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

DeepLoop2 uses PyTorch for inference. A CUDA-enabled GPU is recommended for large-scale prediction, although the exact hardware requirements depend on dataset size and the final inference configuration.

## Input data

DeepLoop2 inference operates on chromosome-level anchor-to-anchor contact data after upstream mapping, filtering, and bias correction.

A typical input table contains:

| Column | Description |
|---|---|
| `anchor_id_1` | first genomic anchor |
| `anchor_id_2` | second genomic anchor |
| `obs` | observed contact signal |
| `exp` | expected contact signal from bias correction |

The exact accepted columns and command-line arguments are defined by `inference/predict.py`.

If your data were generated through the DeepLoop/HiCorr workflow, the chromosome-level anchor-to-anchor output can be used as the starting point for DeepLoop2 after matching it to the appropriate anchor reference.

## Pretrained models

DeepLoop2 provides pretrained models corresponding to different training depths.

We recommend **not automatically assuming that the numerically closest training depth is always optimal for every dataset**. Experimental protocol, sparsity, coverage distribution, and cell type can affect the appearance of the enhanced contact map.

For this reason, DeepLoop2 supports a simple model-selection strategy:

1. Choose several candidate pretrained models near the expected depth of your dataset.
2. Run inference on a few representative genomic loci.
3. Compare the resulting contact maps.
4. Select the model that provides the most appropriate balance between signal recovery and background suppression.
5. Use the selected model for chromosome- or genome-wide inference.

The final checkpoint table and download locations will be provided here before the beta release.

| Model | Training depth | Resolution | Download |
|---|---:|---:|---|
| DeepLoop2 model(s) | To be added | 5 kb | To be added |

Downloaded checkpoints can be stored under:

```text
model/checkpoints/
```

## Run DeepLoop2

The core inference code is:

```text
inference/predict.py
```

A typical workflow requires:

- chromosome-level contact data,
- the corresponding anchor BED/reference,
- a DeepLoop2 checkpoint,
- an output directory,
- and the chromosome to predict.

Because the beta command-line interface is being finalized, we recommend using the notebook example as the canonical runnable example for this release. The README command below will be updated to match the finalized CLI before release:

```bash
python inference/predict.py \
    --input <chromosome_contact_file> \
    --anchor <anchor_bed> \
    --checkpoint <deeploop2_checkpoint> \
    --chromosome chr11 \
    --output <output_directory>
```

> The argument names above describe the intended public interface. Use the arguments shown by `python inference/predict.py --help` once the beta inference script is finalized.

## Output

DeepLoop2 produces enhanced chromosome-level anchor-to-anchor contact predictions.

Conceptually, the output contains:

| Column | Description |
|---|---|
| `anchor_id_1` | first anchor |
| `anchor_id_2` | second anchor |
| `DeepLoop2 score` | predicted/enhanced contact signal |

The prediction can subsequently be converted to Cooler format for visualization and downstream analysis.

## Convert predictions to Cooler

The utility:

```text
utils/convert_to_cooler.py
```

converts DeepLoop2 anchor-to-anchor predictions into a `.cool` contact matrix.

This is a **post-processing/export step**, not a prerequisite for running the model.

The converted `.cool` files can be used with tools in the Cooler ecosystem and compatible genome contact-map viewers.

## Recommended model-selection workflow

Model selection is particularly important for sparse data.

Rather than running every checkpoint genome-wide, we recommend first testing candidate models at a small number of loci containing a mixture of:

- visible local structures,
- relatively sparse regions,
- stronger interaction patterns,
- and representative background signal.

When comparing predictions, consider whether a model:

- recovers coherent interaction structures,
- suppresses isolated noise,
- preserves useful local variation,
- and avoids introducing overly strong artificial patterns.

The example notebook demonstrates this workflow. Model selection is intended as a qualitative aid when an external gold standard is unavailable, not as a replacement for independent validation when such data are available.

## Example notebook

Open:

```text
examples/DeepLoop2_walkthrough.ipynb
```

This is the recommended entry point for new users.

The notebook will walk through:

1. environment setup,
2. loading example/processed contact data,
3. loading a pretrained DeepLoop2 checkpoint,
4. running inference,
5. inspecting predictions,
6. comparing candidate models at selected loci,
7. selecting a model,
8. and optionally exporting predictions to Cooler.

## From raw Hi-C data

DeepLoop2 is the signal-enhancement stage of the workflow; it does not replace read mapping, contact filtering, or systematic-bias correction.

A high-level workflow is:

```text
FASTQ
  ↓
mapping and contact filtering
  ↓
bias correction / processed contact map
  ↓
DeepLoop2
  ↓
enhanced 5 kb contact map
```

Users of the original DeepLoop/HiCorr workflow can refer to [HiCorr](https://github.com/JinLabBioinfo/HiCorr) for upstream processing concepts and examples.

## Differences from DeepLoop

DeepLoop2 builds on the goal of DeepLoop while updating the model and inference strategy for higher-resolution sparse contact data.

| | DeepLoop | DeepLoop2 |
|---|---|---|
| Primary public workflow | LoopDenoise / LoopEnhance | pretrained-model inference |
| Deep-learning framework | TensorFlow | PyTorch |
| Resolution emphasized in current release | 10 kb / ~5 kb anchor workflow | 5 kb |
| Model choice | depth-matched LoopEnhance models | multi-depth pretrained models with locus-level comparison |
| Beginner workflow | prediction walkthrough | Colab-first beta workflow |
| Training code | available | not included in beta |

## Beta release scope

Included:

- DeepLoop2 DenseUNet model definition
- inference code
- pretrained model support
- model-selection guidance
- example Colab notebook
- Cooler export utility

Not included in this beta:

- model training pipeline
- training datasets
- internal benchmarking pipelines

The beta release is intended to make DeepLoop2 inference available while the broader benchmarking and documentation continue to mature.

## Citation

DeepLoop2 manuscript/citation information will be added when available.

For the original DeepLoop method, please cite:

> Zhang, S., Plummer, D., Lu, L. et al. DeepLoop robustly maps chromatin interactions from sparse allele-resolved or single-cell Hi-C data at kilobase resolution. *Nature Genetics* 54, 1013–1025 (2022).

## Contact and feedback

DeepLoop2 is currently a beta release. Bug reports, reproducibility issues, and feedback on the inference workflow are welcome through the GitHub repository.

---

**DeepLoop2 Beta** — high-resolution chromatin contact enhancement from sparse Hi-C data.
