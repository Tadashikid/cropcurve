# CropCurve 🌿

*See the disease before it sees your harvest.* Team **SpotItFirst**.

Photograph a leaf. CropCurve identifies the crop and the condition (38 classes across 14 crops),
reads the local weather forecast, simulates how fast the disease will spread, and shows **the last
good day to treat it**, with a matched spray recommendation.

---

## Start here: the order to do things in

1. **Run the app locally** (section 1, 5 minutes). It works immediately in demo mode.
2. **Train the model on Kaggle** (section 2, about 30 minutes, mostly waiting).
3. **Drop the model into `models/`**. The app switches to real diagnosis.
4. **Add pesticide and weather data** for the diseases you care about (section 3). This can happen in parallel with steps 2–3.
5. **Rehearse the demo** (section 4).

## 1. Run it locally (Windows / PowerShell)

```powershell
cd path\to\cropcurve
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
python -m uvicorn backend.main:app --reload --port 8000
```

Open **http://localhost:8000**.

- If activation fails with "running scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.
- On macOS/Linux, use `source .venv/bin/activate` and `cp` instead.

Until a trained model is in `models/`, the app runs in **demo mode**. The diagnosis is guessed from
the photo's file name, so a file like `PotatoEarlyBlight1.JPG` or `tomato_late_blight.jpg` works.
Everything after the diagnosis is real.

Run the tests with `python -m pytest -q`.

## 2. Train the model on Kaggle

You're using the Kaggle dataset **New Plant Diseases Dataset (Augmented)** (`vipoooool/new-plant-diseases-dataset`).

1. On Kaggle, go to **Create → New Notebook → File → Import Notebook**, and upload `ml/CropCurve_Train_Kaggle.ipynb`.
2. In the notebook's right-hand panel:
   - **Add Input:** search for "New Plant Diseases Dataset" and add it.
   - **Settings → Accelerator:** GPU T4 x2 (or P100).
   - **Settings → Internet:** On. It's needed to download the pretrained weights, and requires a phone-verified Kaggle account.
3. **Run All.** Expect about 20–30 minutes. You'll see the accuracy after each epoch, then predictions for the dataset's 33 loose test photos.
4. Download `models.zip` from the **Output** tab and unzip it so your project has:
   ```
   models/cropcurve_model.onnx
   models/labels.json
   models/samples/        <- real leaf photos, shown in the app as "try a sample"
   ```
5. Restart the server. The "prototype running without a trained model" note disappears.

**How it differs from the notebook you found.** That notebook trains a ResNet-style network from scratch: 5 epochs, about 40 minutes, 99.4% validation accuracy. `ml/train.py` fine-tunes a pretrained **MobileNetV2** instead. This is typically as accurate, faster to train, and small enough (about 9 MB) to run on a laptop CPU. It also exports to ONNX so the web backend doesn't need PyTorch.

**Useful options** (edit the training cell):

| Option | What it does |
|---|---|
| `--max-per-class 150 --epochs 1` | Quick smoke test, about 3 minutes |
| `--max-per-class 0` | Use all ~70k images (slower, slightly better) |
| `--crops tomato potato corn` | Train on a subset of crops |
| `--arch efficientnet_b0` | Larger model, a bit more accurate |

> **Be upfront with judges:** this dataset's validation images are augmented copies drawn from the same
> pool as the training images, so the 97–99% accuracy is optimistic. Real field photos with messy
> backgrounds will score lower. Fine-tuning on PlantDoc (field photos) is the natural next step.

## 3. Pesticides and weather: how to switch a disease "on"

Each of the 38 classes works at one of three levels, controlled purely by the CSV files:

| Level | What the farmer sees | What you need |
|---|---|---|
| Diagnosis only | Name, cause, symptoms, care steps | A row in `data/diseases.csv` (all 38 have one) |
| + spread forecast | The curve and "act before [date]" | The 7 weather columns filled in for that row |
| + spray advice | The "spray by [date]" verdict, a curve per spray day, the product and dose | Rows in `data/pesticides.csv` for that disease id |

**Current coverage:**
- 5 tomato diseases have all three levels.
- Potato early blight and late blight have forecasts, because they're the same pathogens as on tomato, but no pesticide rows yet.
- Everything else is diagnosis only.

The "What it recognises" section of the app underlines the conditions that have forecasts.

**Weather columns in `diseases.csv`** (one value per disease):

| Column | Meaning | Example (late blight) |
|---|---|---|
| `r_base` | Daily spread rate in ideal weather (0.1 slow to 0.5 very fast) | 0.50 |
| `t_opt` | Temperature where it spreads fastest (°C) | 19 |
| `t_width` | How tolerant it is of other temperatures (°C) | 6 |
| `h_thresh` | Humidity (%) above which it really takes off | 88 |
| `h_scale` | How sharp that humidity switch is (smaller = sharper) | 5 |
| `rain_boost` | Extra spread on rainy days (0 none, 1 doubles at 20 mm) | 0.4 |
| `economic_threshold` | Damage fraction you must stay under (0.10 = 10% of leaf area) | 0.10 |

Good sources are plant pathology extension sheets: search for "*disease* optimum temperature humidity
infection". Your state agricultural university's crop protection pages are another option.

**Pesticide rows** use the columns `disease_id`, `product`, `active_ingredient`, `mode` (`protectant` or
`systemic`), `frac_group`, `efficacy` (0–1), `protection_days`, `rainfast` (`yes`/`no`), `typical_rate`,
`cost_level` (`low`/`medium`/`high`) and `notes`. Copy an existing tomato row as a template.
Before using a product, check it's registered for that crop in India on the CIB&RC list.

**Weather** is already wired in. Add a free OpenWeatherMap key to `.env` (`OPENWEATHER_API_KEY=...`) and
restart. New keys can take up to 2 hours to activate. Without a key, farmers set conditions manually.

After editing the CSVs, run `python -m pytest -q`. It checks every id lines up and that the forecasts
still behave sensibly.

## 4. Demo checklist

- [ ] Model in `models/`, and http://localhost:8000/api/health shows `"model_loaded": true`
- [ ] Weather key working, or the "Set it myself" presets ready as a backup
- [ ] **The story:**
  1. Tomato late blight with "Cool and rainy" gives "Spray by [date]". Drag the "What if I wait" slider.
  2. The same leaf with "Hot and dry" gives "No spray needed yet". Same disease, different weather, different decision. This is your differentiator.
  3. A healthy leaf: no spray, which saves money.
  4. A leaf from another crop, to show breadth (38 conditions).
- [ ] The "Try a sample" leaves are real dataset photos, which makes for a reliable live demo
- [ ] For an offline venue, download `chart.umd.min.js` (Chart.js 4.4.1) into `frontend/vendor/`
- [ ] For a public link from your laptop: `ngrok http 8000`

## 5. Project layout

```
backend/
  main.py          API: /api/analyze, /api/simulate, /api/diseases, /api/samples, /api/health; serves frontend/
  classifier.py    ONNX inference, photo quality checks, demo mode
  weather.py       OpenWeatherMap forecast + geocoding, caching, manual fallback
  simulation.py    weather-driven disease spread model (see the docstring for the maths)
  recommend.py     pesticide choice
  data_store.py    CSV loading; disease ids = slugified dataset folder names
frontend/          index.html, styles.css, app.js (plain JS + Chart.js)
ml/
  train.py                       training + ONNX export + sample photos
  CropCurve_Train_Kaggle.ipynb   the same script, ready to run on Kaggle
data/              diseases.csv (38 classes), pesticides.csv
models/            put the trained model here
tests/
```

## 6. Known limitations

- The weather parameters are literature-informed starting values, not field-calibrated.
- The free forecast covers 5 days; days 6–14 repeat the average and are marked in the UI.
- Pesticide rates are typical label rates for reference. Always follow the actual label and local agricultural guidance.
