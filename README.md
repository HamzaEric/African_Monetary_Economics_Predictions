#  Macroeconomic Stress-Testing & Predictive Simulation Engine

An  predictive risk-analysis platform designed to stress-test African economies under severe systemic global shocks. This system allows economists and researchers to simulate major international crises—such as a global banking collapse or severe energy volatility—and instantly visualize the projected impact on core domestic indicators: CPI Inflation, Exchange Rate Returns, and Commercial Lending Rates.

---
##  The Tech Stack & Tooling

The architecture is built entirely on a lightweight, open-source python ecosystem engineered for speed, low latency, and ease of deployment:

* **Machine Learning Engine:** `LightGBM` (Light Gradient Boosting Machine). Selected over deep learning alternatives for its exceptional handling of tabular time-series features, strict threshold splitting, and rapid inference times.
* **Data Engineering & Formats:** `Pandas` and `NumPy` for feature engineering, matrix manipulation, handling lag-variable creation, and executing case-insensitive lookups.
* **Model Serialization:** `Joblib` / `Pickle` for saving and loading pre-trained tree structures without overhead.
* **Interactive Frontend UI:** `Streamlit`. Utilized to construct a dual-dropdown sidebar control room, responsive column layouts, and dynamic, conditional UI rendering cards (`st.metric`, `st.warning`, `st.info`).

---

### Training the LightGBM Clones
* Three individual LightGBM clones were trained as specialized target models:
  1. **The Inflation Clone:** Highly sensitive to global energy volatility shocks (`input_wop_vol`).
  2. **The Currency Clone:** Highly sensitive to "risk-off" global pressure spikes (`input_gpi`), reflecting immediate capital flight.
  3. **The Lending Rate Clone:** Captures central bank contractionary policy responses to domestic shocks.
* The decision trees memorized historic crises (like the 2008 global financial meltdown and the 2020 pandemic logistics disruptions) to form mathematical split conditions.

---
