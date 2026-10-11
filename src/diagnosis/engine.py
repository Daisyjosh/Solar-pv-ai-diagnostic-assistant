"""Evidence-based diagnostic reasoning engine for Solar PV inspection.

Synthesizes thermal/optical visual observations, AI vision predictions,
and operational telemetry measurements into structured engineering assessments
aligned with IEC 62446-3, IEC 61215, and IEA-PVPS Task 13.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from src.knowledge.taxonomy import Observability, TaxonomyRegistry


class SeverityLevel(str, Enum):
    """IEC 62446-3 and engineering risk severity classification."""

    CRITICAL = "CRITICAL"  # Severe fire/safety hazard, imminent diode burnout or string disconnect
    MAJOR = "MAJOR"        # Significant yield loss (>10%), hot spot delta-T > 10C, requires scheduled intervention
    MINOR = "MINOR"        # Early-stage anomaly, delta-T < 10C, external soiling/shading, monitor during next cycle
    NORMAL = "NORMAL"      # Operating within normal thermal and electrical parameters
    INCONCLUSIVE = "INCONCLUSIVE"  # Insufficient environmental conditions or conflicting sensor data


class ReviewStatus(str, Enum):
    """Human-in-the-loop engineering validation lifecycle."""

    PENDING_REVIEW = "PENDING_REVIEW"
    CONFIRMED = "CONFIRMED"
    MODIFIED = "MODIFIED"
    REJECTED = "REJECTED"


@dataclass
class DiagnosticAssessment:
    """Comprehensive structured engineering diagnostic assessment."""

    assessment_id: str
    inspection_id: int
    asset_hierarchy: Dict[str, Any]
    image_modality: str
    visible_observations: List[str]
    model_prediction: Optional[str]
    prediction_confidence: Optional[float]
    prediction_probabilities: Dict[str, float]
    canonical_fault_code: Optional[str]
    fault_category_name: str
    observability: str
    probable_root_causes: List[str]  # Explicitly marked as hypotheses
    supporting_evidence: List[str]
    contradictory_evidence: List[str]
    severity: SeverityLevel
    severity_justification: str
    estimated_power_loss_percent: Optional[float]
    power_loss_justification: str
    corrective_actions: List[str]
    preventive_actions: List[str]
    safety_precautions: List[str]
    actions_to_avoid: List[str]
    required_additional_measurements: List[str]
    environmental_validity_note: str
    diagnostic_limitations: List[str]
    recommended_follow_up: str
    review_status: ReviewStatus
    created_at_utc: str
    reviewed_by: Optional[str] = None
    reviewed_at_utc: Optional[str] = None
    engineer_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["review_status"] = self.review_status.value
        return d


class DiagnosticEngine:
    """Domain-expert reasoning engine synthesizing multi-modal inspection evidence."""

    def __init__(self):
        self.taxonomy = TaxonomyRegistry()

    def diagnose(
        self,
        inspection_id: int,
        asset_info: Dict[str, Any],
        image_modality: str,
        model_prediction: Optional[str] = None,
        model_confidence: Optional[float] = None,
        model_probabilities: Optional[Dict[str, float]] = None,
        irradiance_w_m2: Optional[float] = None,
        ambient_temp_c: Optional[float] = None,
        module_temp_c: Optional[float] = None,
        delta_temp_c: Optional[float] = None,
        operating_voltage_v: Optional[float] = None,
        operating_current_a: Optional[float] = None,
        notes: Optional[str] = None,
        uncertainty_tier: Optional[str] = None,
        input_validation: Optional[Dict[str, Any]] = None,
        taxonomy_note: Optional[str] = None,
    ) -> DiagnosticAssessment:
        """Generate structured diagnostic assessment from multi-modal inputs."""
        timestamp = datetime.now(timezone.utc).isoformat()
        assessment_id = f"DIAG-{inspection_id}-{int(datetime.now(timezone.utc).timestamp())}"
        conf_str = f"{model_confidence:.1%}" if model_confidence is not None else "N/A"
        delta_t_str = f"{delta_temp_c:.1f}°C" if delta_temp_c is not None else "N/A"

        # 1. Environmental Validity Screening (IEC 62446-3 Clause 5.1)
        env_notes = []
        is_env_valid = True
        if irradiance_w_m2 is not None:
            if irradiance_w_m2 < 600.0:
                env_notes.append(
                    f"Irradiance ({irradiance_w_m2:.1f} W/m²) is below IEC 62446-3 minimum threshold (600 W/m²). "
                    "Thermal gradients (ΔT) are dampened; severity may be underestimated."
                )
                is_env_valid = False
            else:
                env_notes.append(f"Irradiance ({irradiance_w_m2:.1f} W/m²) meets IEC 62446-3 minimum criteria.")
        else:
            env_notes.append(
                "In-plane irradiance was not recorded. In accordance with IEC 62446-3, thermal severity cannot be certified."
            )

        # 2. Canonical Taxonomy Mapping
        pred_label = model_prediction or "No-Anomaly"
        mapping = self.taxonomy.map_label("InfraredSolarModules", pred_label)
        canonical_code = mapping.canonical_code if mapping else None
        taxonomy_entry = self.taxonomy.get_entry(canonical_code) if canonical_code else None

        # Check prediction uncertainty and competing candidate classes
        is_ambiguous_prediction = False
        competing_classes: List[Tuple[str, float]] = []
        if uncertainty_tier in ("AMBIGUOUS_HYPOTHESIS", "LOW_CONFIDENCE_HYPOTHESIS"):
            is_ambiguous_prediction = True

        if model_probabilities:
            sorted_probs = sorted(model_probabilities.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_probs) >= 2:
                top1_c, top1_p = sorted_probs[0]
                top2_c, top2_p = sorted_probs[1]
                if top1_p < 0.50 or (top1_p - top2_p) < 0.08:
                    is_ambiguous_prediction = True
                    competing_classes = sorted_probs[:3]
        elif model_confidence is not None and model_confidence < 0.50:
            is_ambiguous_prediction = True

        # 3. Rule-Based Reasoning based on Fault Category & Telemetry
        obs = []
        hypotheses = []
        sup_evidence = []
        contra_evidence = []
        corrective = []
        preventive = []
        safety = []
        avoid = []
        measurements_needed = []
        limitations = [
            "Thermal imagery alone demonstrates surface heat dissipation; internal semiconductor/interconnect defect requires electrical verification.",
            "8-bit rendered thermograms do not support absolute radiometric temperature calculation without calibrated radiometric metadata.",
        ]

        # Integrate input validation diagnostics
        if input_validation:
            if input_validation.get("aspect_ratio_deviation"):
                ar = input_validation.get("aspect_ratio", 1.0)
                obs.append(f"Visual Observation: Aspect ratio ({ar:.2f}) deviates from 0.60 benchmark (24x40).")
            if input_validation.get("is_uncropped_scene"):
                dims = input_validation.get("original_dimensions", (0, 0))
                obs.append(f"Visual Observation: Dimensions ({dims[0]}x{dims[1]}) indicate multi-module/uncropped scene.")
            if not input_validation.get("is_native_grayscale", True):
                obs.append("Visual Observation: Rendered 8-bit false-color capture; uncalibrated radiometry.")
            for w in input_validation.get("warnings", []):
                if w not in limitations:
                    limitations.append(w)

        if taxonomy_note and taxonomy_note not in hypotheses:
            hypotheses.append(f"Domain Taxonomy Context: {taxonomy_note}")

        if pred_label == "No-Anomaly":
            severity = SeverityLevel.NORMAL
            sev_just = "Thermal distribution across cell matrix appears uniform without localized heating."
            p_loss = 0.0
            p_loss_just = "Module operates within normal thermal limits with zero expected thermal power dissipation."
            fault_cat = "Normal Operating Module"
            obs.append("Uniform thermal signature across module surface.")
            hypotheses.append("Hypothesis: Module is operating normally under standard irradiance.")
            corrective.append("No corrective action required. Maintain scheduled routine inspection cycle.")
            preventive.append("Continue periodic thermographic and electrical scanning per IEC 62446-1.")
            safety.append("Standard solar plant electrical safety guidelines apply.")
            avoid.append("Do not clean hot modules during peak solar irradiance (thermal shock risk).")
            follow_up = "Re-inspect during next annual thermography flight."

        elif pred_label in ("Cell", "Hot-Spot"):
            if is_ambiguous_prediction or (model_confidence is not None and model_confidence < 0.50):
                if delta_temp_c is None:
                    severity = SeverityLevel.INCONCLUSIVE
                    sev_just = (
                        f"AI model predicted '{pred_label}' with low confidence ({conf_str}) without calibrated radiometric ΔT. "
                        "Under IEC 62446-3, severity is INCONCLUSIVE until physical thermal delta or electrical measurements are verified."
                    )
                    p_loss = None
                    p_loss_just = "Power loss cannot be quantified without verified ΔT or string electrical measurements."
                    fault_cat = f"Suspected {pred_label} (Uncorroborated AI Hypothesis)"
                else:
                    severity = SeverityLevel.MAJOR if delta_temp_c > 10.0 else SeverityLevel.MINOR
                    sev_just = f"Localized single-cell thermal elevation identified (ΔT: {delta_t_str}). Point shunt or cell reverse-biasing."
                    p_loss = 2.5
                    p_loss_just = "Single hot cell dissipates power produced by adjacent cells in substring (estimated 2-5% loss)."
                    fault_cat = "Single-Cell Hot Spot"
            else:
                severity = SeverityLevel.MAJOR if (delta_temp_c and delta_temp_c > 10.0) or (model_confidence and model_confidence > 0.8) else SeverityLevel.MINOR
                sev_just = (
                    f"Localized single-cell thermal elevation identified (ΔT: {delta_t_str}). "
                    "Points to localized internal shunt, localized micro-crack, or cell reverse-biasing."
                )
                p_loss = 2.5
                p_loss_just = "Single hot cell dissipates power produced by adjacent cells in substring (estimated 2-5% loss)."
                fault_cat = "Single-Cell Hot Spot"

            obs.append("Localized point thermal anomaly concentrated within a single solar cell boundary.")
            hypotheses.extend([
                "Hypothesis A: Localized cell microcrack or wafer fissure causing localized resistance increase.",
                "Hypothesis B: Localized point shunt dissipation under forward/reverse bias.",
                "Hypothesis C: Localized bird dropping or stubborn point soiling forcing cell into reverse bias.",
            ])
            sup_evidence.append(f"AI Vision Model predicted '{pred_label}' (confidence: {conf_str}).")
            if delta_temp_c and delta_temp_c > 10.0:
                sup_evidence.append(f"Observed temperature differential (ΔT = {delta_temp_c:.1f}°C) exceeds IEC 62446-3 Clause 5.2.2 threshold.")
            corrective.extend([
                "Inspect cell surface visually for localized soiling, burn mark, or EVA browning.",
                "Measure string Voc and I-V curve to determine shunt resistance degradation.",
                "If hotspot ΔT > 20°C and backsheet shows browning, flag module for warranty replacement.",
            ])
            preventive.extend([
                "Ensure routine module washing to prevent localized bird dropping calcification.",
                "Audit installation handling to minimize mechanical stress and micro-crack propagation.",
            ])
            safety.extend([
                "Thermal hotspot poses localized fire/delamination risk if backsheet degradation is severe.",
                "Use insulated multimeter probes rated CAT III 1500V during electrical verification.",
            ])
            avoid.extend([
                "Do NOT perform open-circuit disconnection under full load without DC isolator.",
                "Do NOT walk or apply mechanical pressure to module glass during physical examination.",
            ])
            measurements_needed.extend([
                "Calibrated radiometric ΔT under >700 W/m² irradiance.",
                "String Voc and operating current compared to parallel strings.",
                "Module I-V curve trace to evaluate shunt resistance slope.",
            ])
            follow_up = "Perform physical ground check within 14 days and evaluate for warranty claim."

        elif pred_label in ("Cell-Multi", "Hot-Spot-Multi"):
            if is_ambiguous_prediction or (model_confidence is not None and model_confidence < 0.50):
                if delta_temp_c is None and not (operating_voltage_v and asset_info.get("rated_voc_v")):
                    severity = SeverityLevel.INCONCLUSIVE
                    sev_just = (
                        f"AI vision hypothesis '{pred_label}' is unverified ({conf_str} confidence) without measured ΔT or string telemetry. "
                        "Under IEC 62446-3, severity is INCONCLUSIVE until physical measurements are corroborated."
                    )
                    p_loss = None
                    p_loss_just = "Power loss cannot be quantified without verified string telemetry or calibrated radiometry."
                    fault_cat = f"Suspected {pred_label} (Uncorroborated AI Hypothesis)"
                else:
                    severity = SeverityLevel.MAJOR
                    sev_just = "Multiple distributed or clustered cells exhibit elevated thermal dissipation."
                    p_loss = 6.0
                    p_loss_just = "Multiple hot cells dissipate power across multiple substring sections."
                    fault_cat = "Multi-Cell Hot Spot"
            else:
                severity = SeverityLevel.MAJOR
                sev_just = "Multiple distributed or clustered cells exhibit elevated thermal dissipation. Indicates severe cell cracking, manufacturing defect, or localized non-uniform loading."
                p_loss = 6.0
                p_loss_just = "Multiple hot cells dissipate power across multiple substring sections (estimated 5-10% string loss)."
                fault_cat = "Multi-Cell Hot Spot"

            obs.append("Multiple cell coordinates exhibit localized thermal peaks above ambient module baseline.")
            hypotheses.extend([
                "Hypothesis A: Interconnect ribbon fatigue or cracked busbars spanning multiple cells.",
                "Hypothesis B: Multiple micro-cracks induced by hail impact or severe wind mechanical deflection.",
            ])
            sup_evidence.append(f"Model identified multiple thermal hotspots (confidence: {conf_str}).")
            corrective.extend([
                "Perform high-resolution electroluminescence (EL) imaging to confirm wafer crack patterns.",
                "Verify string operating current with DC clamp meter.",
            ])
            preventive.append("Review site mechanical tracker stability and torque settings against high-wind vibration.")
            safety.append("Hotspot clusters accelerate EVA degradation and may breach backsheet dielectric barrier.")
            avoid.append("Avoid delaying inspection if visual browning or backsheet bubbling is observed.")
            measurements_needed.extend([
                "Electroluminescence (EL) inspection in dark conditions.",
                "Module-level open-circuit voltage Voc.",
            ])
            follow_up = "Prioritize for engineering field inspection within 7 days."

        elif pred_label in ("Diode", "Diode-Multi"):
            severity = SeverityLevel.CRITICAL if pred_label == "Diode-Multi" else SeverityLevel.MAJOR
            sev_just = (
                "Continuous thermal band spanning one or more full cell substrings. "
                "Characteristic signature of activated or short-circuited bypass diode in junction box."
            )
            p_loss = 33.3 if pred_label == "Diode" else 66.7
            p_loss_just = f"Bypass diode conduction removes {'one-third' if pred_label == 'Diode' else 'two-thirds'} of total module voltage output (33.3% to 66.7% power loss per module)."
            fault_cat = "Activated / Faulty Bypass Diode"
            obs.append("Distinct longitudinal rectangular band of cells showing uniform temperature elevation across entire substring.")
            hypotheses.extend([
                "Hypothesis A: Bypass diode permanently shorted due to junction box lightning surge or thermal avalanche breakdown.",
                "Hypothesis B: Severe permanent shading on one cell substring forcing bypass diode into continuous forward conduction.",
            ])
            sup_evidence.append(f"Characteristic substring thermal pattern detected (confidence: {conf_str}).")
            if operating_voltage_v and asset_info.get("rated_voc_v"):
                exp_v = asset_info["rated_voc_v"] * (2/3 if pred_label == "Diode" else 1/3)
                if abs(operating_voltage_v - exp_v) < 5.0:
                    sup_evidence.append(f"Measured string voltage ({operating_voltage_v:.1f} V) correlates with bypass of {'1' if pred_label == 'Diode' else '2'} substrings.")
            corrective.extend([
                "Measure module open-circuit voltage (Voc) under irradiance. A reduction of ~1/3 Voc confirms active diode.",
                "Check bypass diode forward bias voltage and reverse breakdown resistance in junction box with diode tester.",
                "Replace faulty bypass diode or replace junction box/module if non-serviceable.",
            ])
            preventive.append("Ensure surge protection devices (SPDs) in DC combiner boxes are functional and grounded.")
            safety.extend([
                "Short-circuited diodes generate continuous heat dissipation in junction box, posing junction box fire risk.",
                "De-energize string before opening junction box enclosure.",
            ])
            avoid.extend([
                "Do NOT leave active diode modules unattended in high-temperature environments.",
                "Do NOT replace diodes with non-Schottky or under-rated replacement parts.",
            ])
            measurements_needed.extend([
                "Module Voc compared to rated STC nameplate.",
                "Diode forward voltage drop (Vf) and reverse leakage current.",
            ])
            follow_up = "Immediate field intervention required: test junction box within 48 hours."

        elif pred_label == "Offline-Module":
            severity = SeverityLevel.CRITICAL
            sev_just = "Entire module is operating completely open-circuit or disconnected, appearing uniformly warmer than surrounding working modules under irradiance."
            p_loss = 100.0
            p_loss_just = "Module is completely disconnected from circuit and delivers 0W power output."
            fault_cat = "Completely Offline / Disconnected Module"
            obs.append("Entire module exhibits uniform thermal elevation relative to adjacent energized modules in string.")
            hypotheses.extend([
                "Hypothesis A: MC4 connector disconnected, burnt, or unclipped.",
                "Hypothesis B: Internal busbar open-circuit failure in junction box.",
                "Hypothesis C: Blown inline string fuse or disconnected string breaker.",
            ])
            sup_evidence.append(f"Uniform module-wide thermal delta observed (confidence: {conf_str}).")
            corrective.extend([
                "Inspect MC4 connectors on positive and negative module leads for uncoupling or thermal charring.",
                "Perform continuity and insulation resistance test on module leads.",
                "Re-mate connectors or replace damaged cable assembly.",
            ])
            preventive.append("Verify torque and lock-click engagement of all MC4 connectors during routine O&M audits.")
            safety.extend([
                "Disconnected MC4 connectors under DC potential pose serious DC arc-flash hazards.",
                "Never decouple energized DC connectors under load.",
            ])
            avoid.append("Do NOT attempt to reconnect charred MC4 connectors without replacing connector contacts.")
            measurements_needed.extend([
                "DC continuity across module leads.",
                "Individual module Voc.",
                "String operating current at combiner box.",
            ])
            follow_up = "Urgent: Dispatch technician to re-connect module within 24 hours."

        elif pred_label in ("Shadowing", "Vegetation"):
            severity = SeverityLevel.MINOR
            sev_just = f"External non-defect obstruction ({pred_label.lower()}) causing partial cell shading and localized thermal dissipation."
            p_loss = 15.0
            p_loss_just = "Shading forces bypass diode activation or cell reverse bias (estimated 10-25% string power reduction)."
            fault_cat = f"External Obstruction ({pred_label})"
            obs.append(f"Thermal gradient matches visual pattern of external {pred_label.lower()} across module lower border.")
            hypotheses.append(f"Hypothesis: Environmental {pred_label.lower()} casting shadows during inspection window.")
            sup_evidence.append(f"Model identified {pred_label} (confidence: {conf_str}).")
            corrective.append(
                "Clear vegetation / trim trees adjacent to array row." if pred_label == "Vegetation"
                else "Evaluate nearby structural obstructions or tracker tilt orientation."
            )
            preventive.append("Schedule regular mowing and vegetation management around module tables.")
            safety.append("Standard mechanical landscaping safety around high-voltage PV equipment.")
            avoid.append("Avoid using metal trimmers or weed-whackers directly in contact with module cabling.")
            measurements_needed.append("Re-inspect at solar noon to confirm absence of structural shading.")
            follow_up = "Clear obstruction during next routine groundskeeping cycle."

        elif pred_label == "Soiling":
            severity = SeverityLevel.MINOR
            sev_just = "Surface dust, pollen, or sand accumulation reduces optical transmittance."
            p_loss = 8.0
            p_loss_just = "Uniform and non-uniform optical transmission reduction (estimated 5-12% module yield loss)."
            fault_cat = "Surface Soiling"
            obs.append("Diffuse surface opacity and non-uniform thermal dissipation on module glass.")
            hypotheses.append("Hypothesis: Extended dry period or industrial dust deposition.")
            sup_evidence.append(f"Model identified Soiling (confidence: {conf_str}).")
            corrective.append("Clean module surface using demineralized water and soft PV-certified cleaning brushes.")
            preventive.append("Adjust automated cleaning robot schedule or seasonal washing intervals based on soiling ratio.")
            safety.append("Do not wash modules during peak solar hours (risk of thermal glass fracture).")
            avoid.append("Do NOT use abrasive detergents, high-pressure washers (>35 bar), or hard water.")
            measurements_needed.append("Soiling ratio measurement or clean vs soiled reference cell comparison.")
            follow_up = "Schedule module washing batch."

        elif pred_label == "Cracking":
            if is_ambiguous_prediction or (model_confidence is not None and model_confidence < 0.50) or delta_temp_c is None:
                severity = SeverityLevel.INCONCLUSIVE
                competing_desc = ""
                if len(competing_classes) > 1:
                    competing_desc = f", closely contested by '{competing_classes[1][0]}' at {competing_classes[1][1]:.1%}"
                sev_just = (
                    f"AI vision hypothesis '{pred_label}' is unverified ({conf_str} confidence{competing_desc}). "
                    "In accordance with IEC 62446-3 Clause 5.2 and IEA-PVPS Task 13, severity is classified as INCONCLUSIVE "
                    "because thermal thermography cannot verify microcracking without Electroluminescence (EL) imaging and string I-V curve corroboration."
                )
                p_loss = None
                p_loss_just = (
                    "Yield loss cannot be quantified from thermal imagery alone. Electroluminescence (EL) and string I-V curve tracing "
                    "are required to determine active cell area loss and shunt degradation."
                )
                fault_cat = "Suspected Cell Microcrack (Uncorroborated AI Hypothesis)"
                obs.append(f"Localized temperature gradient detected. AI prediction '{pred_label}' is an uncorroborated hypothesis ({conf_str}).")
                hypotheses.append("Hypothesis A (Top AI Candidate): Possible wafer microcrack or thermal stress fissure (requires EL imaging to confirm).")
                if len(competing_classes) > 1:
                    hypotheses.append(f"Hypothesis B (Alternative AI Candidate): {competing_classes[1][0]} ({competing_classes[1][1]:.1%}) — localized cell heating or external shading.")
                hypotheses.append("Hypothesis C: Point shunting or localized surface resistance anomaly.")
            else:
                severity = SeverityLevel.MAJOR if (delta_temp_c and delta_temp_c > 10.0) else SeverityLevel.MINOR
                sev_just = f"Confirmed or high-confidence ({conf_str}) cracking signature with observed thermal gradient (ΔT: {delta_t_str})."
                p_loss = 4.0
                p_loss_just = "Microcracks isolate cell fragments and generate high-resistance current bottlenecks (estimated 3-8% loss)."
                fault_cat = "Cell Wafer Microcrack"
                obs.append("Irregular localized temperature distribution along suspected crack lines.")
                hypotheses.append("Hypothesis: Mechanical stress, hail impact, or thermal cycling wafer crack.")

            sup_evidence.append(f"Model identified Cracking signature (confidence: {conf_str}).")
            corrective.append("Conduct night-time Electroluminescence (EL) imaging to map inactive cell fragments.")
            preventive.append("Enforce strict walking/stepping bans during mechanical assembly.")
            safety.append("Cracked cells can progressively develop into severe fire-risk hotspots over operating lifetime.")
            avoid.append("Do NOT apply localized mechanical pressure to cracked glass or cells.")
            measurements_needed.extend([
                "Electroluminescence (EL) image to confirm wafer crack presence and inactive fragments.",
                "Module I-V curve trace for fill factor degradation.",
                "Calibrated radiometric ΔT under >700 W/m² irradiance.",
            ])
            follow_up = "Perform EL testing during next scheduled site audit."

        else:
            severity = SeverityLevel.INCONCLUSIVE
            sev_just = f"Unrecognized or ambiguous category '{pred_label}'."
            p_loss = None
            p_loss_just = "Cannot determine power loss without recognized fault signature."
            fault_cat = "Inconclusive Anomaly"
            obs.append(f"Uncategorized visual thermal pattern: {pred_label}")
            hypotheses.append("Hypothesis: Ambiguous anomaly requiring manual specialist review.")
            corrective.append("Perform manual engineering physical inspection.")
            preventive.append("Standard O&M inspection protocol.")
            safety.append("Standard high-voltage PV safety.")
            avoid.append("Avoid making operational decisions without physical confirmation.")
            measurements_needed.append("Full electrical I-V trace and calibrated thermal scan.")
            follow_up = "Assign to senior PV engineer for manual review."

        return DiagnosticAssessment(
            assessment_id=assessment_id,
            inspection_id=inspection_id,
            asset_hierarchy=asset_info,
            image_modality=image_modality,
            visible_observations=obs,
            model_prediction=model_prediction,
            prediction_confidence=model_confidence,
            prediction_probabilities=model_probabilities or {},
            canonical_fault_code=canonical_code,
            fault_category_name=fault_cat,
            observability=(
                taxonomy_entry.observability.value
                if taxonomy_entry
                else Observability.REQUIRES_ELECTRICAL_CORROBORATION.value
            ),
            probable_root_causes=hypotheses,
            supporting_evidence=sup_evidence,
            contradictory_evidence=contra_evidence,
            severity=severity,
            severity_justification=sev_just,
            estimated_power_loss_percent=p_loss,
            power_loss_justification=p_loss_just,
            corrective_actions=corrective,
            preventive_actions=preventive,
            safety_precautions=safety,
            actions_to_avoid=avoid,
            required_additional_measurements=measurements_needed,
            environmental_validity_note=" | ".join(env_notes),
            diagnostic_limitations=limitations,
            recommended_follow_up=follow_up,
            review_status=ReviewStatus.PENDING_REVIEW,
            created_at_utc=timestamp,
            engineer_notes=notes,
        )


# Global singleton instance
_diagnostic_engine: Optional[DiagnosticEngine] = None


def get_diagnostic_engine() -> DiagnosticEngine:
    """Retrieve global DiagnosticEngine singleton."""
    global _diagnostic_engine
    if _diagnostic_engine is None:
        _diagnostic_engine = DiagnosticEngine()
    return _diagnostic_engine
