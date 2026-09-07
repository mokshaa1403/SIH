# SatQuery 

SatQuery ISRO Lite is a runnable, evidence-first prototype for natural-language analysis of satellite imagery. It routes a question to one of four specialist workflows and returns an answer, visual evidence, confidence factors, and an execution trace.

This repository is intentionally lightweight. The included analyzers are deterministic computer-vision baselines so the complete product can run locally without downloading multi-gigabyte model weights. The architecture is designed so that each baseline can later be replaced by a trained PyTorch model without changing the website or API.

## Supported workflows

- **Visual question answering:** answers controlled questions about water, vegetation, built-up land, and scene composition.
- **Grounding:** highlights water, vegetation, built-up, or bare-land regions.
- **Change detection:** compares a before/after optical pair and produces a change mask.
- **Optical–SAR fusion:** combines optical land-cover evidence with normalized SAR intensity.
- **Scene recognition:** optionally predicts beach, ice, Mars, Moon, mountain, ocean, or river when the supplied local dataset has been trained.
- **RGB-region grounding:** highlights strongly red, green, or blue pixels in ordinary PNG/JPEG RGB images for transparent overlay testing.

## Quick start

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/create_demo_data.py
python scripts/train_scene_classifier.py --data data/scene_dataset --deployment-all
python -m satquery.server
```

Open <http://127.0.0.1:8000>.

On Windows, activate the environment with `.venv\Scripts\activate`.

## Try these queries

| Input | Question |
|---|---|
| `sample_data/optical.png` | `Highlight the water body` |
| `sample_data/optical.png` | `What is visible in this image?` |
| `sample_data/before.png` + `sample_data/after.png` | `What changed between these dates?` |
| `sample_data/optical.png` + `sample_data/sar.png` | `Use optical and SAR to identify water and built-up areas` |
| `sample_data/rgb_test.png` | `Highlight the red pixels` (also try green or blue) |

Select the correct input mode in the website before running the query.

## Test

```bash
python -m unittest discover -s tests -v
```

## Architecture

```text
Browser UI
   -> JSON API
   -> Input validation and preprocessing
   -> Query router
      -> VQA baseline
      -> Grounding baseline
      -> Change detector
      -> Optical–SAR fusion baseline
   -> Evidence renderer
   -> Confidence engine
   -> Answer + overlay + execution trace
```

The router produces a structured intent. Model outputs are structured records rather than free-form prose, and the response composer only describes those records. This reduces unsupported answers.

## API

`POST /api/analyze`

```json
{
  "question": "Highlight the water body",
  "mode": "single_optical",
  "images": [
    {"name": "optical.png", "data": "data:image/png;base64,..."}
  ]
}
```

The response includes `task`, `answer`, `confidence`, `metrics`, `warnings`, `trace`, and an optional base64-encoded `evidence_image`.

`GET /api/health` returns service and model-registry status.

## Repository layout

```text
satquery/
  server.py             HTTP server and API
  router.py             Hybrid rule-based query router
  imaging.py            Decoding, validation, masks, overlays
  confidence.py         Explainable confidence aggregation
  models/               Replaceable specialist analyzers
  static/               Browser application
config/model_registry.json
scripts/create_demo_data.py
tests/
```

## What is a baseline?

The included implementation uses colour, texture, intensity, and image-difference rules. It is suitable for demonstrating routing, evidence, validation, UX, and system integration. It is **not** scientifically validated for operational decisions.

### Imported scene dataset

The locally supplied `archive.zip` is extracted to `data/scene_dataset` and contains 719 images across seven classes: beach, ice, Mars, Moon, mountain, ocean, and river. It is not BigEarthNet and contains no paired Sentinel-1/Sentinel-2 data or question-answer annotations. `scripts/train_scene_classifier.py` builds a lightweight k-nearest-neighbour scene baseline at `artifacts/scene_classifier.npz` and evaluates it on the corrected split. The raw `data/` and generated `artifacts/` directories are excluded from Git because the source archive is large and its provenance/licence was not included. The local testing ZIP contains the generated classifier file for convenience; remove it before public redistribution unless the source licence permits derived-model sharing.

Evaluation uses a duplicate-checked, deterministic, class-balanced 50/50 train/test split rather than the archive's highly uneven supplied split. See `DATASET.md` for the protocol, current results and limitations.

The packaged deployment model is trained separately on all 701 deduplicated supplied images. It has no internal test score because evaluation must use your new unseen dataset:

```text
new_test_dataset/
  beach/
  ice/
  mars/
  moon/
  mountain/
  ocean/
  river/
```

```bash
python scripts/evaluate_scene_classifier.py --data /absolute/path/to/new_test_dataset
```

The command prints overall and per-class accuracy and writes `artifacts/external_test_predictions.csv`.

## Advantages implemented in the runnable version

| Existing friction | Implemented response |
|---|---|
| Expert chooses every workflow | Natural-language router selects a supported specialist and explains the route. |
| Text result without spatial support | Grounding and change workflows return a pixel overlay. |
| Commercial or cloud dependency | The default server and models run locally without an external API. |
| Closed input ecosystem | The prototype accepts ordinary PNG and JPEG uploads. |
| Unclear reliability | Every result contains confidence factors, warnings and an execution trace. |
| One general model can hallucinate | Structured specialist outputs constrain the final answer. |
| ISRO portals remain separate from conversational analysis | The model registry provides a clean future adapter point for Bhuvan, Bhoonidhi and VEDAS services. |

`GET /api/model-card` reports the installed model, training protocol, classes and limitations. The interface shows the same status at runtime.

For an SIH submission, replace baselines in this order:

1. VQA with a remote-sensing VLM adapted using LoRA on VRSBench/RSVQA.
2. Grounding with a text-grounding model followed by segmentation.
3. Change detection with a trained bi-temporal change model.
4. Optical–SAR fusion with separate optical and SAR encoders plus late fusion.

Each replacement should implement the same `analyze(...)` contract used in `satquery/models/`.

## Safety and limitations

- Results are decision support, not authoritative geospatial conclusions.
- PNG and JPEG inputs are supported in this lightweight version.
- Single-image requests automatically identify RGB/false-colour composites versus normal natural-colour imagery and expose that decision in the result.
- Pair mode accepts two files together or in two successive selections; the Detect change shortcut selects the correct mode automatically.
- Large images are resized to protect memory.
- Change detection requires aligned images of the same area.
- Optical–SAR fusion assumes the images are already co-registered.
- Confidence is an explainable prototype score, not a calibrated probability.
- Unsupported questions are refused instead of being guessed.

## Docker

```bash
docker build -t satquery-isro-lite .
docker run --rm -p 8000:8000 satquery-isro-lite
```

## GitHub publishing

Run and verify the project locally first. Then initialize and publish it yourself:

```bash
git init
git add .
git commit -m "Initial SatQuery ISRO Lite prototype"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```
