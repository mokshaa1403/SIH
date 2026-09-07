# Run SatQuery ISRO Lite

## 1. Install the two Python packages

```bash
python3 -m pip install -r requirements.txt
```

## 2. Start the website

```bash
python3 -m satquery.server --port 8765
```

Open `http://127.0.0.1:8765` in a browser.

Try `sample_data/rgb_test.png` with **Highlight the red pixels**. The result should include a red evidence overlay, coverage measurement, confidence factors, execution trace, and a validation warning.

For water, keep **Single image — auto-detect RGB / normal** selected. The app first reports whether the upload looks like an RGB/false-colour composite or a normal natural-colour image, then applies the matching water rule.

For comparison, click **Detect change**. It automatically switches to **Before / after pair**. Select `before.png` and `after.png` together, or add them one at a time; both filenames must appear before running the analysis.

## 3. Test a completely new labelled dataset

Keep the new test images outside the training folders. Arrange them like this:

```text
my_new_test_dataset/
  beach/
  ice/
  mars/
  moon/
  mountain/
  ocean/
  river/
```

Then run:

```bash
python3 scripts/evaluate_scene_classifier.py --data /absolute/path/to/my_new_test_dataset
```

Predictions are written to `artifacts/external_test_predictions.csv`.

## What is genuinely trained

The included scene classifier was trained on all 701 unique supplied images. Eighteen exact duplicates were excluded. No internal test score is claimed because you requested that the separate future dataset be used for testing.

The supervised water segmenter uses 296 paired images: 180 from the water-body set, 20 from each of five non-water-dominant land-cover groups, and all 16 available water-dominant land-cover pairs. All remaining 3,348 paired images were kept unseen and evaluated after training. Full reserved-set results: 84.47% accuracy, 71.23% precision, 69.51% recall, 54.28% IoU and 70.36% Dice. The exact split is stored under `artifacts/water_split/`.

If `datasetsss` is beside the extracted project folder, run `python3 scripts/evaluate_water_segmenter.py`. Otherwise pass its location explicitly with `python3 scripts/evaluate_water_segmenter.py --data-root /absolute/path/to/datasetsss`.

The colour grounding, image-pair change detection, and optical–SAR fusion workflows are runnable explainable baselines. They are not presented as trained deep-learning models because the supplied archive has no masks, boxes, question-answer labels, aligned time pairs, or optical–SAR pairs. Their modules can be replaced later when suitable labelled data is available.
