"""
AI-Based Patient Load Prediction for Government Hospitals
Module: src/resource_planner.py

Calculates clinical resource requirements, staffing estimates, bed occupancy projections,
and consumable inventory indicators based on predicted patient volume and Indian Public
Health Standards (IPHS) guidelines.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class HospitalResourcePlanner:
    """
    Translates predicted OPD and Emergency patient loads into actionable
    hospital resource allocations:
    - Doctors (OPD and Emergency shifts)
    - Nursing staff
    - Bed occupancy and admission projections
    - Medicine and consumable buffer index
    """
    def __init__(
        self,
        opd_patients_per_doctor: float = 45.0,
        emergency_patients_per_doctor: float = 18.0,
        nurses_per_occupied_bed: float = 0.25,      # 1 nurse per 4 beds in general ward
        nurses_per_emergency_bed: float = 0.50,     # 1 nurse per 2 emergency beds
        opd_admission_rate: float = 0.045,          # ~4.5% of OPD cases require admission
        emergency_admission_rate: float = 0.22,     # ~22% of emergency cases require admission
        avg_length_of_stay_days: float = 3.5,       # Average Length of Stay (ALOS)
    ):
        self.opd_ratio = opd_patients_per_doctor
        self.emg_ratio = emergency_patients_per_doctor
        self.nurse_ward_ratio = nurses_per_occupied_bed
        self.nurse_emg_ratio = nurses_per_emergency_bed
        self.opd_adm_rate = opd_admission_rate
        self.emg_adm_rate = emergency_admission_rate
        self.alos = avg_length_of_stay_days

    def estimate_daily_resources(
        self,
        predicted_opd: float,
        predicted_emergency: float,
        total_sanctioned_beds: int = 500,
        current_occupied_beds: Optional[int] = None,
        available_doctors: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculates daily resource requirements for a single prediction instance.
        """
        pred_opd = max(0.0, float(predicted_opd))
        pred_emg = max(0.0, float(predicted_emergency))
        total_load = pred_opd + pred_emg

        # 1. Staffing requirements
        needed_opd_doctors = int(np.ceil(pred_opd / self.opd_ratio))
        needed_emg_doctors = int(np.ceil(pred_emg / self.emg_ratio))
        total_doctors_required = needed_opd_doctors + needed_emg_doctors

        doctor_deficit = 0
        if available_doctors is not None:
            doctor_deficit = max(0, total_doctors_required - available_doctors)

        # 2. Admission projections
        projected_new_admissions = int(np.round(
            (pred_opd * self.opd_adm_rate) + (pred_emg * self.emg_adm_rate)
        ))

        # 3. Bed occupancy projection
        # If current occupancy is not provided, estimate based on steady-state Little's Law (Admissions * ALOS)
        if current_occupied_beds is not None:
            projected_occupied_beds = min(
                total_sanctioned_beds,
                int(current_occupied_beds + projected_new_admissions - (current_occupied_beds / self.alos))
            )
        else:
            projected_occupied_beds = min(
                total_sanctioned_beds,
                int(np.round(projected_new_admissions * self.alos))
            )

        occupancy_rate_pct = round((projected_occupied_beds / max(1, total_sanctioned_beds)) * 100.0, 1)
        available_beds = max(0, total_sanctioned_beds - projected_occupied_beds)

        # 4. Nursing requirements
        needed_ward_nurses = int(np.ceil(projected_occupied_beds * self.nurse_ward_ratio))
        needed_emg_nurses = int(np.ceil(pred_emg * 0.15 * self.nurse_emg_ratio)) # acute care share
        total_nurses_required = needed_ward_nurses + needed_emg_nurses

        # 5. Medicine & Consumable Demand Index (relative scalar where 100 is baseline)
        # OPD uses 1.0 unit per patient, Emergency uses 3.5 units per acute patient
        consumable_units = (pred_opd * 1.0) + (pred_emg * 3.5)
        medicine_demand_index = round((consumable_units / 5000.0) * 100.0, 1)

        return {
            "predicted_opd": round(pred_opd, 1),
            "predicted_emergency": round(pred_emg, 1),
            "predicted_total_load": round(total_load, 1),
            "projected_admissions": projected_new_admissions,
            "projected_occupied_beds": projected_occupied_beds,
            "available_beds": available_beds,
            "occupancy_rate_pct": occupancy_rate_pct,
            "total_sanctioned_beds": total_sanctioned_beds,
            "needed_opd_doctors": needed_opd_doctors,
            "needed_emergency_doctors": needed_emg_doctors,
            "total_doctors_required": total_doctors_required,
            "doctor_deficit": doctor_deficit,
            "total_nurses_required": total_nurses_required,
            "medicine_demand_index": medicine_demand_index
        }

    def generate_forecast_schedule(
        self,
        forecast_df: pd.DataFrame,
        total_sanctioned_beds: int = 500
    ) -> pd.DataFrame:
        """
        Processes a multi-day forecast DataFrame (containing 'date', 'pred_opd', 'pred_emergency')
        and appends full clinical staffing and bed allocation columns.
        """
        df = forecast_df.copy()
        results = []
        for _, row in df.iterrows():
            res = self.estimate_daily_resources(
                predicted_opd=row['pred_opd'],
                predicted_emergency=row['pred_emergency'],
                total_sanctioned_beds=total_sanctioned_beds
            )
            results.append(res)

        res_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), res_df], axis=1)
