# 🌾 AgriPattern

AI-assisted agricultural data discovery platform for:
- correlations
- anomalies
- clusters
- trends
- unexpected relationships
- automatic visual exploration
- AI-assisted explanations

## 1. Create environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## 2. Install packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Optional AI setup

Copy `.env.example` to `.env` and add your Groq API key.

```env
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile
```

## 4. Run

```powershell
python -m streamlit run app.py
```

Open the Local URL shown by Streamlit.

## 5. Recommended dataset

Use CSV/XLSX with columns such as:

Year, State, District, Crop, Rainfall, Temperature,
Soil_N, Soil_P, Soil_K, Fertilizer, Irrigation,
Yield, Production, Area, Market_Price

The app also works with generic numeric agricultural datasets.
