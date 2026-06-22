import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="GrainGuardian Analytical Computation Engine")

# Enable Cross-Origin Resource Sharing (CORS) so GitHub Pages can read this API safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your GitHub Pages domain to fetch data
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input data layout structure standard
class TelemetryDataInput(BaseModel):
    crop_id: int
    temp_t1: float
    temp_t2: float
    temp_t3: float
    rh: float
    is_storage: bool

# Crop Database Configuration Constants Matching your V3 Specifications
CROP_DB = {
    0: {"name": "Paddy (Rice)", "c1": 1.9187e-5, "c2": 51.161, "c3": 2.4484, "safe_limit": 13.5, "price_per_ton": 21840},
    1: {"name": "Wheat", "c1": 2.3007e-5, "c2": 35.853, "c3": 2.2857, "safe_limit": 13.0, "price_per_ton": 22750},
    2: {"name": "Maize (Corn)", "c1": 8.6541e-5, "c2": 49.810, "c3": 1.8634, "safe_limit": 13.5, "price_per_ton": 20900}
}

@app.post("/api/calculate")
def process_analytics(payload: TelemetryDataInput):
    crop = CROP_DB.get(payload.crop_id, CROP_DB[0])
    
    # 1. Math Equation: Calculate Equilibrium Moisture Content (EMC %)
    rh_decimal = max(0.1, min(99.9, payload.rh)) / 100.0
    term1 = -math.log(1.0 - rh_decimal)
    
    # Evaluating using the deep core sensor temperature layer (T3)
    term2 = crop["c1"] * (payload.temp_t3 + crop["c2"])
    calculated_emc = math.pow(term1 / term2, 1.0 / crop["c3"]) * 100 if term2 > 0 else 0
    
    # 2. Risk Evaluation Engine and Penalties
    score = 100
    reasons = []
    
    moisture_gap = calculated_emc - crop["safe_limit"]
    if moisture_gap > 0:
        score -= round(moisture_gap * 15)
        reasons.append(f"Moisture limit breached by {moisture_gap:.1f}%")
        
    thermal_gradient = max(abs(payload.temp_t1 - payload.temp_t2), 
                           abs(payload.temp_t2 - payload.temp_t3), 
                           abs(payload.temp_t1 - payload.temp_t3))
    if thermal_gradient > 4.0:
        score -= 25
        reasons.append(f"Thermal gradient anomaly: ΔT = {thermal_gradient:.1f}°C")
        
    peak_temp = max(payload.temp_t1, payload.temp_t2, payload.temp_t3)
    if peak_temp > 38.0:
        score -= 30
        reasons.append(f"Hyper-heating spike: {peak_temp:.1f}°C")
        
    score = max(0, min(100, score))
    status = "SAFE" if score >= 85 else ("WARNING" if score >= 60 else "CRITICAL")
    
    # 3. Socio-Economic Impact Calculations (Financial Degradation Analysis)
    total_batch_mass = 10000  # Baseline metric allocation: 10 Metric Tons
    weight_shrinkage_kg = 0.0
    financial_loss_inr = 0.0
    
    if calculated_emc > crop["safe_limit"]:
        shrinkage_rate = 0.002 * (calculated_emc - crop["safe_limit"])
        weight_shrinkage_kg = total_batch_mass * shrinkage_rate
        financial_loss_inr = (weight_shrinkage_kg / 1000.0) * crop["price_per_ton"]

    return {
        "score": score,
        "status": status,
        "emc": round(calculated_emc, 2),
        "thermal_average": round((payload.temp_t1 + payload.temp_t2 + payload.temp_t3) / 3, 1),
        "weight_loss_kg": round(weight_shrinkage_kg, 1),
        "financial_loss_rs": round(financial_loss_inr, 2),
        "reasons": " | ".join(reasons) if reasons else "Parameters nominal."
    }
