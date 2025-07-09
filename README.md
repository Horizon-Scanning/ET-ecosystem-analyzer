# Ecosystem Analyzer

An interactive web dashboard (built with [Streamlit](https://streamlit.io)) that visualises the emerging technologies ecosystem on a spider (radar) diagram.

## Features

- Filter by Scientific Field (Web of Science category)
- Filter by Year range
- Filter by technology count (icon size)
- Interactive scatter-polar (radar) chart where:
  - Angle = `TRL x Adoption` dimension
  - Radius = `Impact` (values closer to 1 sit nearer the centre)
  - Marker size = `Tech_Count`
- Live data table of the filtered technologies

## Quick start

1. Install dependencies (preferably inside a virtual environment):

```bash
pip install -r requirements.txt
```

2. Launch the app:

```bash
streamlit run app.py
```

3. Your browser will open automatically at `http://localhost:8501`. Use the sidebar to explore the dataset interactively.

## File overview

- `app.py` – Streamlit front-end and data handling code.
- `emerging_technologies_processed.csv` – static dataset used by the dashboard.
- `requirements.txt` – Python dependencies.

---
Feel free to further tailor the UI/UX, add custom icons, or deploy the app on Streamlit Community Cloud for easy sharing. 