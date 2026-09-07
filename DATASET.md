# Imported Scene Dataset

## Source

Local archive supplied by the project owner:

`/Users/mokshanagda/Downloads/archive.zip`

The archive did not contain a README, citation, source URL, or licence file. Do not publish or redistribute its images until provenance and reuse rights are confirmed.

## Contents

| Class | Training | Test |
|---|---:|---:|
| beach | 100 | 9 |
| ice | 73 | 4 |
| mars | 100 | 2 |
| moon | 100 | 2 |
| mountain | 100 | 14 |
| ocean | 100 | 7 |
| river | 100 | 8 |
| **Total** | **673** | **46** |

This is a scene-classification dataset. It is not BigEarthNet.txt and does not contain co-registered Sentinel-1/Sentinel-2 pairs, multispectral bands, VQA annotations, or ISRO Cartosat/RISAT data.

## Evaluation protocol

The original 673/46 split is not used for reporting. All images are first deduplicated by SHA-256 within each class, then divided class-by-class into a deterministic 50/50 training and test split using seed 42. For classes with an odd number of unique images, the test side receives the extra image. This gives a substantially larger held-out test set and prevents exact duplicate leakage within a class.

Run the training command below to reproduce the current measured result. The result remains a local baseline because the dataset's provenance is unknown and its classes do not match the ISRO challenge.

Current result after deduplication: **323/351 correct predictions (92.0%)**.

| Class | Held-out test images | Accuracy |
|---|---:|---:|
| beach | 54 | 77.8% |
| ice | 31 | 93.5% |
| mars | 51 | 100.0% |
| moon | 51 | 88.2% |
| mountain | 56 | 87.5% |
| ocean | 54 | 100.0% |
| river | 54 | 98.1% |

## Reproduce

```bash
python scripts/train_scene_classifier.py --data data/scene_dataset --seed 42
```

The generated model is stored in `artifacts/scene_classifier.npz` and is loaded automatically by the SatQuery VQA workflow.

## Deployment model and new test dataset

Use the complete supplied dataset for the final deployable classifier only when the test dataset will be supplied later and kept separate:

```bash
python scripts/train_scene_classifier.py --data data/scene_dataset --deployment-all
```

This trains on all 701 unique images and records `all_deduplicated_external_test` in the model file. It deliberately stores no internal accuracy. Evaluate the later dataset with one class folder per label:

```bash
python scripts/evaluate_scene_classifier.py --data /absolute/path/to/new_test_dataset
```

The new test set must not be used to adjust features, thresholds or labels before its final score is recorded. If it contains different classes, add and train those classes first rather than treating an unknown class as one of the existing seven.
