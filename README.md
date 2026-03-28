# Weather Classification (Streamlit + TensorFlow)

A simple **image classification** web app built with **Streamlit** and **TensorFlow** that predicts the weather category in an uploaded image.

**Classes**
- Rainy
- Cloudy
- Thunderstorm

The trained model is included in this repository as `keras_model.h5`, and the labels are stored in `labels.txt`.

---

## Project Structure

- `test.py` — Streamlit app (main entrypoint)
- `keras_model.h5` — trained Keras model
- `labels.txt` — class label mapping
- `requirements.txt` — Python dependencies

---

## Prerequisites

- **Python 3.9–3.11** (recommended)
- `pip` (or `pip3`)
- (Recommended) `venv` for a virtual environment

This project uses:
- `streamlit`
- `tensorflow-cpu==2.15.0`
- `Pillow`
- `numpy<2.0.0`

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mjamagos/weather_classification.git
   cd weather_classification
   ```

2. **Create and activate a virtual environment**

   macOS / Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   Windows (PowerShell):
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

Start the Streamlit app:

```bash
streamlit run test.py
```

Then open the local URL Streamlit prints in your terminal (usually something like `http://localhost:8501`).

---

## How It Works (High-level)

1. Upload an image (`.jpg`, `.jpeg`, or `.png`)
2. The app:
   - converts it to RGB
   - resizes/crops to **224×224**
   - normalizes pixel values to match the model’s expected input
3. The TensorFlow model predicts probabilities for each class
4. The app displays:
   - the top predicted label + confidence
   - a probability breakdown for all classes

---

## Notes / Troubleshooting

- Make sure these files exist in the repo root when running:
  - `keras_model.h5`
  - `labels.txt`
- If TensorFlow install issues occur, confirm you're using a supported Python version and that your environment is clean (virtualenv recommended).
- The app caches the model in memory using `st.cache_resource` to avoid reloading it on every interaction.

---

## License

Add a license if you plan to share or reuse this project publicly (e.g., MIT, Apache-2.0).
