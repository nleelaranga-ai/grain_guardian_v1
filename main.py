import math
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="GrainGuardian Intelligent Engine v3",
    description="IEEE-Compliant Multi-Crop Decision Support Systems Engine",
    version="3.0.0"
)

# Robust Cross-Origin Resource Sharing (CORS) setup so GitHub Pages can fetch safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Input Validation Payload matching frontend sliders exactly
class TelemetryDataInput(BaseModel):
    crop_id: int = Field(..., description="0: Paddy, 1: Wheat, 2: Maize")
    temp_t1: float = Field(..., ge=0.0, le=75.0, description="Top Layer Temp")
    temp_t2: float = Field(..., ge=0.0, le=75.0, description="Middle Layer Temp")
    temp_t3: float = Field(..., ge=0.0, le=75.0, description="Bottom Layer Temp")
    rh: float = Field(..., ge=10.0, le=100.0, description="Relative Humidity Input")
    is_storage: bool = True

class ClpEngineStatus(BaseModel):
    clp_moisture_violation: bool
    clp_temp_violation: bool
    clp_humidity_violation: bool
    clp_duration_violation: bool
    clp_fungal_violation: bool

# Response Payload matching what index.html needs to update charts & labels
class AnalysisResponse(BaseModel):
    record_id: str
    grain_health_index: int
    fungal_risk_status: str
    biological_activity_index: float
    projected_weight_loss_kg: float
    estimated_financial_loss_inr: float
    thermal_average: float
    clp_matrix: ClpEngineStatus
    action_advisory: List[str]

# Crop Database Coefficients Matrix matching V3 specifications
CROP_DB = {
    0: {"name": "Paddy (Rice)", "c1": 1.9187e-5, "c2": 51.161, "c3": 2.4484, "safe_limit": 13.5, "crit_moisture": 16.0, "warn_moisture": 14.5, "crit_temp": 42.0, "warn_temp": 35.0, "price_per_ton": 21840},
    1: {"name": "Wheat", "c1": 2.3007e-5, "c2": 35.853, "c3": 2.2857, "safe_limit": 13.0, "crit_moisture": 15.5, "warn_moisture": 14.0, "crit_temp": 40.0, "warn_temp": 33.0, "price_per_ton": 22750},
    2: {"name": "Maize (Corn)", "c1": 8.6541e-5, "c2": 49.810, "c3": 1.8634, "safe_limit": 13.5, "crit_moisture": 15.8, "warn_moisture": 14.2, "crit_temp": 41.0, "warn_temp": 34.0, "price_per_ton": 20900}
}

# Health Route validation page check
@app.get("/")
def health():
    return {
        "status": "running",
        "service": "GrainGuardian V3 API Engine"
    }

@app.post("/api/v3/analyze", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
def execute_grain_intelligence_pass(payload: TelemetryDataInput):
    try:
        # Load profile configurations based on numeric selection index
        crop = CROP_DB.get(payload.crop_id, CROP_DB[0])
        
        # 1. Math Equation: Calculate Equilibrium Moisture Content (EMC %)
        rh_decimal = max(0.1, min(99.9, payload.rh)) / 100.0
        term1 = -math.log(1.0 - rh_decimal)
        term2 = crop["c1"] * (payload.temp_t3 + crop["c2"])
        calculated_emc = math.pow(term1 / term2, 1.0 / crop["c3"]) * 100 if term2 > 0 else 0.0

        # 2. Biological Activity Index Curve Calculations
        bai = float(((calculated_emc / crop["safe_limit"]) ** 2) * (payload.temp_t3 / 28.0))
        fungal_risk = "LOW" if bai < 1.05 else ("MEDIUM" if bai < 1.30 else "HIGH")

        # 3. Evaluate CLP Threshold boundary violation matrices
        clp_m = calculated_emc > crop["crit_moisture"]
        clp_t = max(payload.temp_t1, payload.temp_t2, payload.temp_t3) > crop["crit_temp"]
        clp_h = payload.rh > 75.0
        clp_d = False
        clp_f = fungal_risk == "HIGH"

        # 4. Compute GHI score weighted penalty matrices (60% Moisture / 40% Thermal)
        m_penalty = max(0.0, ((calculated_emc - crop["warn_moisture"]) / (crop["crit_moisture"] - crop["warn_moisture"])) * 100) if calculated_emc > crop["warn_moisture"] else 0.0
        t_max = max(payload.temp_t1, payload.temp_t2, payload.temp_t3)
        t_penalty = max(0.0, ((t_max - crop["warn_temp"]) / (crop["crit_temp"] - crop["warn_temp"])) * 100) if t_max > crop["warn_temp"] else 0.0
        
        thermal_gradient = max(abs(payload.temp_t1 - payload.temp_t2), abs(payload.temp_t2 - payload.temp_t3), abs(payload.temp_t1 - payload.temp_t3))
        gradient_penalty = 25.0 if thermal_gradient > 4.0 else 0.0

        aggregated_penalty = (0.5 * m_penalty) + (0.3 * t_penalty) + (0.2 * gradient_penalty)
        ghi_score = max(0, min(100, round(100.0 - aggregated_penalty)))

        # 5. Value Economic Mass Shrinkage (Baseline scale: 12,000 kg)
        stored_mass_kg = 12000.0
        weight_loss = 0.0
        if calculated_emc > crop["safe_limit"]:
            gap_ratio = (calculated_emc - crop["safe_limit"]) / 100.0
            severity_factor = 1.35 if clp_f else 1.0
            weight_loss = stored_mass_kg * gap_ratio * severity_factor

        loss_inr = round((weight_loss / 1000.0) * crop["price_per_ton"], 2)

        # 6. Action Advisories Construction Block
        advisories = []
        if clp_m or calculated_emc > crop["warn_moisture"]:
            advisories.append(f"CRITICAL_MOISTURE: Calculated Moisture ({calculated_emc:.1f}%) breaches limits. Engage mechanical aeration fans.")
        if clp_t or t_max > crop["warn_temp"]:
            advisories.append(f"THERMAL_SPIKE: Elevated hot spot layers detected (Max: {t_max:.1f}°C). Rotate the batch volume.")
        if clp_f:
            advisories.append("FUNGAL_OUTBREAK_RISK: High Biological Activity. Pull core layer probe samples immediately.")
        if not advisories:
            advisories.append("SAFE: All storage boundaries nominal. Maintain tight hermetic microclimates.")

        return AnalysisResponse(
            record_id=f"rec-{int(datetime.now(timezone.utc).timestamp())}",
            grain_health_index=ghi_score,
            fungal_risk_status=fungal_risk,
            biological_activity_index=round(bai, 2),
            projected_weight_loss_kg=round(weight_loss, 1),
            estimated_financial_loss_inr=loss_inr,
            thermal_average=round((payload.temp_t1 + payload.temp_t2 + payload.temp_t3) / 3, 1),
            clp_matrix=ClpEngineStatus(
                clp_moisture_violation=clp_m, clp_temp_violation=clp_t,
                clp_humidity_violation=clp_h, clp_duration_violation=clp_d, clp_fungal_violation=clp_f
            ),
            action_advisory=advisories
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Mathematical calculation pipeline error: {str(e)}"
        )
