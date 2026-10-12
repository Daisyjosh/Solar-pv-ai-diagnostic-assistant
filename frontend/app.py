"""Solar PV AI Diagnostic Assistant - Operations & Maintenance Portal.

Enterprise full-stack engineering interface for solar photovoltaic inspection,
thermal anomaly analysis, telemetry ingestion, and multi-modal decision support.
Aligned with IEC 62446-3, IEC 61215, and IEA-PVPS Task 13.
"""

import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image
import streamlit as st

import importlib
import frontend.api_client
importlib.reload(frontend.api_client)
from frontend.api_client import BackendAPIClient
from src.core.formatting import format_power_loss, is_synthetic_asset, is_synthetic_telemetry

# Configure page layout
st.set_page_config(
    page_title="Solar PV AI Diagnostic Assistant",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

client = BackendAPIClient()
health = client.check_health()

# Top Header
st.title("☀️ Solar PV AI Diagnostic Assistant")
st.caption(
    "Enterprise AI Diagnostic Platform for Solar Photovoltaic Inspection, "
    "IEC 62446-3 Thermography Analysis, Electrical Telemetry Ingestion, and O&M Decision Support"
)

# Sidebar System Health
st.sidebar.title("System Health")
if health["online"]:
    if health["status"] == "healthy":
        st.sidebar.success(f"🟢 **API Online** ({health['latency_ms']} ms)")
    else:
        st.sidebar.warning(f"🟡 **API Degraded** ({health['latency_ms']} ms)")

    if health["database"] == "connected":
        st.sidebar.success("🟢 **Database:** Connected")
    else:
        st.sidebar.error("🔴 **Database:** Disconnected")
    st.sidebar.caption(f"Environment: `{health['environment']}` | Version: `{health['version']}`")
else:
    st.sidebar.error("🔴 **Backend API Offline**")
    st.sidebar.caption(f"Target: `{health['url']}`")
    if health.get("error"):
        st.sidebar.caption(f"Reason: {health['error']}")

st.sidebar.markdown("---")

def render_gemini_analysis_card(res: Optional[Dict[str, Any]]):
    """Render structured Gemini multimodal vision analysis card."""
    if not res:
        return
    status = res.get("status")
    model_name = res.get("model_name", "gemini-3.5-flash")
    msg = res.get("message", "")
    analysis_text = res.get("analysis_text")
    disclaimer = res.get("disclaimer", "AI-generated visual hypothesis for engineering review — not a certified physical diagnosis.")

    if status == "SUCCESS" and analysis_text:
        st.success(f"✅ **Gemini Vision Analysis Complete** (Model: `{model_name}`)")
        st.info("ℹ️ **AI Visual Hypothesis**: The assessment below is an exploratory visual hypothesis adhering to IEC 62446-3 guidelines. It is not a certified physical diagnosis and does not substitute for on-site engineering verification.")
        with st.container(border=True):
            st.markdown(analysis_text)
            st.markdown("---")
            st.caption(f"🛡️ **Engineering Disclaimer**: {disclaimer}")
    elif status == "API_KEY_MISSING":
        st.warning("⚠️ **Gemini API Key Required**")
        st.markdown(
            "Gemini Multimodal Vision Analysis uses Google's free-tier Gemini API.\n\n"
            "To enable this capability:\n"
            "1. Obtain a free API key from [Google AI Studio](https://aistudio.google.com/).\n"
            "2. Add `GEMINI_API_KEY=your_key_here` to the `.env` file in the project root (`C:\\sem-7-project\\.env`).\n"
            "3. Restart the backend API server."
        )
    elif status == "RATE_LIMIT_EXCEEDED":
        st.warning(f"⏳ **Rate Limit Exceeded (429)**: {msg}")
    elif status == "SERVICE_BUSY":
        st.warning(f"⏳ **Service Temporarily Busy (503)**: {msg}")
    elif status == "MODEL_ERROR":
        st.error(f"❌ **Model Error**: {msg}")
    elif status == "NETWORK_ERROR":
        st.error(f"🌐 **Network Error**: {msg}")
    elif status == "INVALID_API_KEY":
        st.error(f"🔑 **Invalid API Key**: {msg}")
    else:
        st.error(f"⚠️ **Analysis Status: {status}**: {msg}")


def render_yolo_analysis_card(res: Optional[Dict[str, Any]]):
    """Render structured YOLO object detection & localization card."""
    if not res:
        return
    status = res.get("status")
    model_path = res.get("model_path") or "Not configured"
    msg = res.get("message", "")
    detections = res.get("detections", [])
    total_detections = res.get("total_detections", len(detections))
    annotated_b64 = res.get("annotated_image_base64")
    disclaimer = res.get("disclaimer", "Automated YOLO detection hypothesis for engineering review — requires physical verification.")

    if status == "SUCCESS":
        st.success(f"🎯 **YOLO Anomaly Localization Complete** ({total_detections} detection(s) found)")
        st.info("ℹ️ **Spatial Localization**: Bounding boxes identify candidate anomaly regions according to IEC 62446-3 thermography inspection criteria.")

        c_left, c_right = st.columns([1, 1])
        with c_left:
            if annotated_b64:
                try:
                    annotated_bytes = base64.b64decode(annotated_b64)
                    st.image(Image.open(io.BytesIO(annotated_bytes)), caption="YOLO Spatial Anomaly Detections", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render annotated detection image: {e}")
            else:
                st.caption("No annotated visualization available.")

        with c_right:
            if detections:
                st.markdown("##### Detected Anomaly Candidates")
                det_rows = []
                for idx, d in enumerate(detections, 1):
                    bbox = d.get("bbox", {})
                    det_rows.append({
                        "#": idx,
                        "Category": d.get("class_name", "Unknown"),
                        "Confidence": f"{d.get('confidence', 0.0):.1%}",
                        "BBox [xmin, ymin, xmax, ymax]": f"[{bbox.get('xmin', 0):.0f}, {bbox.get('ymin', 0):.0f}, {bbox.get('xmax', 0):.0f}, {bbox.get('ymax', 0):.0f}]",
                    })
                st.dataframe(pd.DataFrame(det_rows), use_container_width=True)
            else:
                st.info("No anomalies detected above confidence threshold.")

        st.caption(f"Model Path: `{model_path}`")
        st.warning(f"🛡️ **Engineering Disclaimer**: {disclaimer}")

    elif status == "WEIGHTS_UNAVAILABLE":
        st.warning("⚠️ **YOLO Model Weights Unavailable**")
        st.markdown(
            f"{msg}\n\n"
            "**Setup Instructions**:\n"
            "1. Train or download a YOLOv8/v9/v11 model annotated with solar PV anomaly bounding boxes (e.g., Hotspot, Diode, Cell, Multi-cell, Cracking).\n"
            "2. Place the trained weights file at `models/yolo/active/best.pt` (or set `YOLO_MODEL_PATH` in `.env`).\n"
            "3. Ensure `ultralytics` is installed in your Python environment (`pip install ultralytics`).\n"
            "4. No artificial detections or fake bounding boxes are generated."
        )
    elif status == "DEPENDENCY_UNAVAILABLE":
        st.warning("⚠️ **YOLO Runtime Dependency Unavailable**")
        st.markdown(
            f"{msg}\n\n"
            "To enable YOLO inference, install `ultralytics`:\n"
            "```powershell\n"
            "pip install ultralytics\n"
            "```"
        )
    elif status == "IMAGE_ERROR":
        st.error(f"🖼️ **Image Decode Error**: {msg}")
    elif status == "FILE_NOT_FOUND":
        st.error(f"📁 **File Not Found**: {msg}")
    elif status == "INFERENCE_ERROR":
        st.error(f"❌ **YOLO Inference Error**: {msg}")
    else:
        st.error(f"⚠️ **YOLO Status: {status}**: {msg}")


# Navigation Menu
NAVIGATION_VIEWS = [
    "1. Dashboard",
    "2. Plant Management",
    "3. Asset Hierarchy",
    "4. New Inspection",
    "5. Thermal Image Upload & Gallery",
    "6. Optical RGB Upload & Gallery",
    "7. Image-Analysis Results",
    "8. Diagnostic Assessment",
    "9. Telemetry Import & Trends",
    "10. Diagnostic Chatbot",
    "11. Inspection History & Compare",
    "12. Batch Inspection Monitoring",
    "13. PDF Engineering Reports",
    "14. Model Status & Metrics",
    "15. System Config & SCADA Status",
]

selected_view = st.sidebar.radio("Navigation View", NAVIGATION_VIEWS)

st.sidebar.markdown("---")
st.sidebar.markdown("### Engineering Standards")
st.sidebar.caption("• **IEC 62446-3**: IR Thermography\n• **IEC 61215**: Design Qualification\n• **IEA-PVPS Task 13**: PV Failure Modes")

if not health["online"]:
    st.warning(
        "⚠️ **Backend API is currently unreachable.**\n\n"
        "Start the FastAPI backend service in your terminal to enable live operations:\n"
        "```powershell\n"
        "python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload\n"
        "```"
    )

# ==============================================================================
# VIEW 1: DASHBOARD
# ==============================================================================
if selected_view == "1. Dashboard":
    st.header("Executive O&M Dashboard")
    summary = client.get_dashboard_summary()
    kpis = summary.get("kpis", {})

    # Top KPI Metrics Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Plants", kpis.get("total_plants", 0))
    c2.metric("Total Inspections", kpis.get("total_inspections", 0))
    c3.metric("Media Assets", kpis.get("total_images", 0))
    c4.metric("Pending Reviews", kpis.get("pending_reviews", 0))
    c5.metric("Open High-Priority", kpis.get("open_critical_findings", 0))
    c6.metric("Telemetry Points", kpis.get("total_telemetry_records", 0))

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Fault Category Distribution")
        fault_dist = summary.get("fault_distribution", {})
        if fault_dist:
            df_faults = pd.DataFrame(list(fault_dist.items()), columns=["Anomaly Category", "Count"]).set_index("Anomaly Category")
            st.bar_chart(df_faults)
        else:
            st.info("No diagnostic assessments recorded yet. Run inspections to populate fault distributions.")

    with col_right:
        st.subheader("Severity Breakdown")
        sev_dist = summary.get("severity_distribution", {})
        if sev_dist:
            df_sev = pd.DataFrame(list(sev_dist.items()), columns=["Severity Level", "Count"]).set_index("Severity Level")
            st.bar_chart(df_sev)
        else:
            st.info("No severity classifications evaluated yet.")

    st.subheader("Recent Inspections")
    recent = summary.get("recent_inspections", [])
    if recent:
        df_recent = pd.DataFrame(recent)
        df_recent.columns = ["ID", "Code", "Plant", "Date", "Status", "Severity", "Fault Finding"]
        st.dataframe(df_recent, use_container_width=True)
    else:
        st.info("No inspection events recorded. Register a plant and create your first inspection session in the sidebar.")

# ==============================================================================
# VIEW 2: PLANT MANAGEMENT
# ==============================================================================
elif selected_view == "2. Plant Management":
    st.header("Solar PV Plant Registry")

    tab_list, tab_create = st.tabs(["Registered Plants", "Register New Plant"])

    with tab_list:
        plants = client.get_plants()
        if plants:
            df_plants = pd.DataFrame(plants)[["id", "code", "name", "location", "capacity_dc_mw", "is_active", "commissioning_date"]]
            df_plants.columns = ["ID", "Code", "Name", "Location", "Capacity (MWp)", "Active", "Commissioning Date"]
            st.dataframe(df_plants, use_container_width=True)
        else:
            st.info("No solar PV plants registered yet. Use the 'Register New Plant' tab to add your utility site.")

    with tab_create:
        with st.form("create_plant_form"):
            st.subheader("New Plant Details")
            c1, c2 = st.columns(2)
            name = c1.text_input("Plant Name *", placeholder="e.g. Desert Horizon Solar Park")
            code = c2.text_input("Unique Plant Code *", placeholder="e.g. DHSP-01")
            location = c1.text_input("Geographic Location *", placeholder="e.g. Barstow, CA, USA")
            capacity = c2.number_input("DC Nameplate Capacity (MWp) *", min_value=0.01, value=50.0, step=1.0)
            comm_date = c1.date_input("Commissioning Date", value=datetime.now().date())
            is_active = c2.checkbox("Operating & Active", value=True)

            submitted = st.form_submit_button("Register Plant")
            if submitted:
                if not name.strip() or not code.strip() or not location.strip():
                    st.error("Please fill in all required fields (Name, Code, Location).")
                else:
                    payload = {
                        "name": name.strip(),
                        "code": code.strip(),
                        "location": location.strip(),
                        "capacity_dc_mw": float(capacity),
                        "commissioning_date": comm_date.isoformat(),
                        "is_active": is_active,
                    }
                    ok, res = client.create_plant(payload)
                    if ok:
                        st.success(f"Plant '{name}' registered successfully with ID {res.get('id')}!")
                        st.rerun()
                    else:
                        st.error(f"Failed to register plant: {res}")

# ==============================================================================
# VIEW 3: ASSET HIERARCHY
# ==============================================================================
elif selected_view == "3. Asset Hierarchy":
    st.header("Asset Hierarchy (Inverters → Strings → Modules)")
    plants = client.get_plants()
    if not plants:
        st.warning("Please register a plant in 'Plant Management' first.")
    else:
        plant_options = {p["name"]: p["id"] for p in plants}
        selected_pname = st.selectbox("Select Plant", list(plant_options.keys()))
        selected_pid = plant_options[selected_pname]

        tab_tree, tab_add_inv, tab_add_str, tab_add_mod = st.tabs([
            "Asset Explorer", "Add Inverter", "Add PV String", "Add PV Module"
        ])

        with tab_tree:
            inverters = client.get_inverters(selected_pid)
            if not inverters:
                st.info("No inverters configured for this plant.")
            else:
                for inv in inverters:
                    with st.expander(f"⚡ Inverter: {inv['code']} ({inv['rated_power_kw']} kW | {inv.get('model') or 'Standard'})", expanded=False):
                        strings = client.get_strings(inv["id"])
                        if not strings:
                            st.caption("No strings configured under this inverter.")
                        else:
                            for s in strings:
                                st.markdown(f"**String `{s['code']}`** — {s['modules_count']} modules | Voc: {s['rated_voc_v']}V | Isc: {s['rated_isc_a']}A")
                                modules = client.get_modules(s["id"])
                                if modules:
                                    mod_codes = [f"`{m['code']}` (Pos {m['position_in_string']})" for m in modules]
                                    st.caption("Modules: " + ", ".join(mod_codes))

        with tab_add_inv:
            with st.form("add_inverter_form"):
                st.subheader(f"Add Inverter to {selected_pname}")
                inv_code = st.text_input("Inverter Code *", placeholder="e.g. INV-01")
                inv_model = st.text_input("Model / Manufacturer", placeholder="e.g. SMA Sunny Central 2500")
                inv_power = st.number_input("Rated AC Power (kW) *", min_value=1.0, value=2500.0, step=10.0)
                if st.form_submit_button("Add Inverter"):
                    ok, res = client.create_inverter(selected_pid, {"code": inv_code, "model": inv_model, "rated_power_kw": inv_power})
                    if ok:
                        st.success(f"Inverter {inv_code} added successfully!")
                        st.rerun()
                    else:
                        st.error(f"Error: {res}")

        with tab_add_str:
            inverters = client.get_inverters(selected_pid)
            if not inverters:
                st.info("Add an inverter first.")
            else:
                inv_opts = {inv["code"]: inv["id"] for inv in inverters}
                sel_inv_code = st.selectbox("Select Parent Inverter", list(inv_opts.keys()))
                sel_inv_id = inv_opts[sel_inv_code]

                with st.form("add_string_form"):
                    str_code = st.text_input("String Code *", placeholder="e.g. STR-01-A")
                    mod_count = st.number_input("Modules in String *", min_value=1, value=24)
                    voc = st.number_input("Rated Voc (V) *", min_value=10.0, value=960.0, step=10.0)
                    isc = st.number_input("Rated Isc (A) *", min_value=1.0, value=11.5, step=0.5)
                    if st.form_submit_button("Add String"):
                        ok, res = client.create_string(sel_inv_id, {
                            "code": str_code, "modules_count": mod_count, "rated_voc_v": voc, "rated_isc_a": isc
                        })
                        if ok:
                            st.success(f"String {str_code} added successfully!")
                            st.rerun()
                        else:
                            st.error(f"Error: {res}")

        with tab_add_mod:
            inverters = client.get_inverters(selected_pid)
            if not inverters:
                st.info("Add an inverter first.")
            else:
                all_strings = []
                for inv in inverters:
                    all_strings.extend(client.get_strings(inv["id"]))
                if not all_strings:
                    st.info("Add a string first.")
                else:
                    str_opts = {s["code"]: s["id"] for s in all_strings}
                    sel_str_code = st.selectbox("Select Parent String", list(str_opts.keys()))
                    sel_str_id = str_opts[sel_str_code]

                    with st.form("add_module_form"):
                        mod_code = st.text_input("Module Serial / Code *", placeholder="e.g. MOD-01-A-12")
                        pos = st.number_input("Position in String *", min_value=1, value=1)
                        mfg = st.text_input("Manufacturer", placeholder="e.g. Canadian Solar")
                        m_model = st.text_input("Module Model", placeholder="e.g. HiKu CS3W-400P")
                        m_power = st.number_input("Rated STC Power (W) *", min_value=10.0, value=400.0, step=10.0)
                        if st.form_submit_button("Add Module"):
                            ok, res = client.create_module(sel_str_id, {
                                "code": mod_code, "position_in_string": pos,
                                "manufacturer": mfg, "model": m_model, "rated_power_w": m_power
                            })
                            if ok:
                                st.success(f"Module {mod_code} added successfully!")
                                st.rerun()
                            else:
                                st.error(f"Error: {res}")

# ==============================================================================
# VIEW 4: NEW INSPECTION
# ==============================================================================
elif selected_view == "4. New Inspection":
    st.header("Schedule & Create Inspection Session")
    plants = client.get_plants()
    if not plants:
        st.warning("Please register a plant first.")
    else:
        plant_options = {p["name"]: p["id"] for p in plants}
        selected_pname = st.selectbox("Select Plant *", list(plant_options.keys()))
        selected_pid = plant_options[selected_pname]

        # Optional equipment linking
        inverters = client.get_inverters(selected_pid)
        inv_choices = {"None / Array-Level": None}
        for inv in inverters:
            inv_choices[f"Inverter: {inv['code']}"] = inv["id"]
        sel_inv_label = st.selectbox("Target Inverter (Optional)", list(inv_choices.keys()))
        sel_inv_id = inv_choices[sel_inv_label]

        with st.form("create_inspection_form"):
            c1, c2 = st.columns(2)
            code = c1.text_input("Inspection Code *", value=f"INSP-{datetime.now().strftime('%Y%m%d-%H%M')}")
            inspector = c2.text_input("Lead Inspector Name *", value="Chief Field Engineer")
            insp_date = c1.datetime_input("Inspection Timestamp", value=datetime.now())
            status_val = c2.selectbox("Initial Status", ["SCHEDULED", "IN_PROGRESS", "COMPLETED"])

            st.markdown("#### Environmental Parameters (IEC 62446-3)")
            c3, c4 = st.columns(2)
            amb_temp = c3.number_input("Ambient Temperature (°C)", value=25.0, step=1.0)
            irradiance = c4.number_input("In-Plane Irradiance (W/m²) [Min 600 W/m² recommended]", value=750.0, step=10.0)
            weather = st.text_input("Weather Notes", value="Clear sky, direct sunlight, wind speed < 2.5 m/s")
            notes = st.text_area("Inspection Scope & Notes", value="Aerial drone thermal thermography scan of array section.")

            if st.form_submit_button("Create Inspection Session"):
                payload = {
                    "code": code.strip(),
                    "plant_id": selected_pid,
                    "inverter_id": sel_inv_id,
                    "inspector_name": inspector.strip(),
                    "inspection_date": insp_date.isoformat(),
                    "status": status_val,
                    "ambient_temp_c": amb_temp,
                    "irradiance_w_m2": irradiance,
                    "weather_notes": weather,
                    "general_notes": notes,
                }
                ok, res = client.create_inspection(payload)
                if ok:
                    st.success(f"Inspection session '{code}' created with ID {res.get('id')}!")
                    st.info("You can now upload thermal/RGB imagery in 'Thermal Image Upload & Gallery'.")
                else:
                    st.error(f"Failed to create inspection: {res}")

# ==============================================================================
# VIEW 5: THERMAL IMAGE UPLOAD & GALLERY
# ==============================================================================
elif selected_view == "5. Thermal Image Upload & Gallery":
    st.header("Thermal Infrared Imagery Ingestion & Gallery")
    inspections = client.get_inspections()
    if not inspections:
        st.info("Create an inspection session first.")
    else:
        insp_opts = {f"#{insp['id']} - {insp['code']} ({insp['inspector_name']})": insp["id"] for insp in inspections}
        sel_insp_label = st.selectbox("Target Inspection Session", list(insp_opts.keys()))
        sel_insp_id = insp_opts[sel_insp_label]

        tab_up, tab_gal = st.tabs(["Upload Thermal Image", "Inspection Media Gallery"])

        with tab_up:
            st.markdown("#### Secure Thermal Ingestion Pipeline")
            st.caption("Enforces streaming size limits, PIL header validation, SHA-256 fingerprinting, and collision-resistant storage.")
            thermal_file = st.file_uploader("Select Infrared Thermal File (.jpg, .jpeg, .png, .tiff)", type=["jpg", "jpeg", "png", "tif", "tiff"], key="thermal_up")
            t_notes = st.text_input("Image Notes / Target Substring", placeholder="e.g. Module 12 string A, suspected hotspot")

            if st.button("Upload & Ingest Thermal Image"):
                if not thermal_file:
                    st.error("Please select a file to upload.")
                else:
                    content = thermal_file.read()
                    ok, res = client.upload_image(sel_insp_id, content, thermal_file.name, "THERMAL_IR", t_notes)
                    if ok:
                        st.success(f"Thermal image ingested successfully! Asset ID: {res.get('id')}")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {res}")

        with tab_gal:
            detail = client.get_inspection_detail(sel_insp_id)
            if detail and detail.get("images"):
                thermal_imgs = [img for img in detail["images"] if img["image_type"] == "THERMAL_IR"]
                if not thermal_imgs:
                    st.info("No thermal images uploaded for this inspection yet.")
                else:
                    cols = st.columns(3)
                    for idx, img_info in enumerate(thermal_imgs):
                        with cols[idx % 3]:
                            st.markdown(f"**Asset #{img_info['id']}** — `{img_info['original_filename']}`")
                            if is_synthetic_asset(img_info):
                                st.warning("⚠️ **SYNTHETIC TEST IMAGE** (Simulated)")
                            img_bytes = client.get_image_file_bytes(sel_insp_id, img_info["id"])
                            if img_bytes:
                                pil_im = Image.open(io.BytesIO(img_bytes))
                                st.image(pil_im, use_container_width=True)
                            st.caption(f"SHA-256: `{img_info['checksum_sha256'][:16]}...` | Size: {img_info['file_size_bytes']} B")
            else:
                st.info("No media assets linked to this inspection.")

# ==============================================================================
# VIEW 6: OPTICAL RGB UPLOAD & GALLERY
# ==============================================================================
elif selected_view == "6. Optical RGB Upload & Gallery":
    st.header("Optical RGB Imagery Ingestion & Gallery")
    inspections = client.get_inspections()
    if not inspections:
        st.info("Create an inspection session first.")
    else:
        insp_opts = {f"#{insp['id']} - {insp['code']} ({insp['inspector_name']})": insp["id"] for insp in inspections}
        sel_insp_label = st.selectbox("Target Inspection Session", list(insp_opts.keys()))
        sel_insp_id = insp_opts[sel_insp_label]

        tab_up, tab_gal = st.tabs(["Upload Optical RGB Image", "Inspection RGB Gallery"])

        with tab_up:
            rgb_file = st.file_uploader("Select Optical RGB Image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], key="rgb_up")
            rgb_notes = st.text_input("Optical Observations", placeholder="e.g. Surface discoloration, vegetative shading visible")

            if st.button("Upload & Ingest Optical RGB Image"):
                if not rgb_file:
                    st.error("Please select a file to upload.")
                else:
                    content = rgb_file.read()
                    ok, res = client.upload_image(sel_insp_id, content, rgb_file.name, "OPTICAL_RGB", rgb_notes)
                    if ok:
                        st.success(f"Optical RGB image ingested successfully! Asset ID: {res.get('id')}")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {res}")

        with tab_gal:
            detail = client.get_inspection_detail(sel_insp_id)
            if detail and detail.get("images"):
                rgb_imgs = [img for img in detail["images"] if img["image_type"] == "OPTICAL_RGB"]
                if not rgb_imgs:
                    st.info("No optical RGB images uploaded for this inspection yet.")
                else:
                    cols = st.columns(3)
                    for idx, img_info in enumerate(rgb_imgs):
                        with cols[idx % 3]:
                            st.markdown(f"**Asset #{img_info['id']}** — `{img_info['original_filename']}`")
                            if is_synthetic_asset(img_info):
                                st.warning("⚠️ **SYNTHETIC TEST IMAGE** (Simulated)")
                            img_bytes = client.get_image_file_bytes(sel_insp_id, img_info["id"])
                            if img_bytes:
                                pil_im = Image.open(io.BytesIO(img_bytes))
                                st.image(pil_im, use_container_width=True)
                            st.caption(f"SHA-256: `{img_info['checksum_sha256'][:16]}...` | Size: {img_info['file_size_bytes']} B")
            else:
                st.info("No media assets linked to this inspection.")


# ==============================================================================
# VIEW 7: IMAGE-ANALYSIS RESULTS
# ==============================================================================
elif selected_view == "7. Image-Analysis Results":
    st.header("AI Vision Anomaly Classification")

    # Display operational model indicators
    gem_status = client.get_gemini_status() if hasattr(client, "get_gemini_status") else {"is_configured": False}
    yolo_status = (
        client.get_yolo_status()
        if hasattr(client, "get_yolo_status")
        else {"is_configured": False, "status": "WEIGHTS_UNAVAILABLE"}
    )

    gem_desc = f"**Google Gemini Vision** (`{gem_status.get('model_name')}` ready)" if gem_status.get("is_configured") else "Gemini Vision: ⚪ *Key not configured (Optional free-tier)*"
    yolo_desc = "**YOLO Localization** (ready)" if yolo_status.get("is_configured") else "YOLO Localization: ⚪ *Weights pending*"

    st.caption(
        f"Active Models: **HistGradientBoosting** (12-class benchmark) | "
        f"{gem_desc} | {yolo_desc}"
    )

    tab_inspect, tab_direct = st.tabs(["Classify Existing Ingested Asset", "Direct Thermal Crop Classification"])

    with tab_inspect:
        inspections = client.get_inspections()
        if not inspections:
            st.info("Create an inspection session first.")
        else:
            insp_opts = {f"#{insp['id']} - {insp['code']}": insp["id"] for insp in inspections}
            sel_insp_id = st.selectbox("Select Inspection Session", list(insp_opts.values()), format_func=lambda x: [k for k, v in insp_opts.items() if v == x][0])

            detail = client.get_inspection_detail(sel_insp_id)
            if not detail or not detail.get("images"):
                st.info("No images uploaded for this inspection. Upload images in View 5 or 6.")
            else:
                img_opts = {f"Asset #{img['id']} ({img['image_type']} - {img['original_filename']})": img["id"] for img in detail["images"]}
                sel_asset_id = st.selectbox("Select Image Asset to Classify", list(img_opts.values()), format_func=lambda x: [k for k, v in img_opts.items() if v == x][0])

                col_img, col_pred = st.columns([1, 1])

                with col_img:
                    img_bytes = client.get_image_file_bytes(sel_insp_id, sel_asset_id)
                    if img_bytes:
                        st.image(Image.open(io.BytesIO(img_bytes)), caption=f"Selected Asset #{sel_asset_id}", use_container_width=True)
                    sel_img_obj = next((im for im in detail["images"] if im["id"] == sel_asset_id), None)
                    if is_synthetic_asset(sel_img_obj):
                        st.warning("⚠️ **SYNTHETIC TEST IMAGE**: Uploaded benchmark crop for software validation. Predictions represent synthetic benchmarks, not real field defects.")

                with col_pred:
                    btn_c1, btn_c2, btn_c3 = st.columns(3)
                    with btn_c1:
                        if st.button("Run ML Classifier", use_container_width=True, key=f"btn_ml_{sel_asset_id}"):
                            with st.spinner("Executing feature extraction and model inference..."):
                                st.session_state[f"ml_pred_{sel_asset_id}"] = client.predict_image_asset(sel_asset_id)
                    with btn_c2:
                        if st.button("✨ Analyze with Gemini", use_container_width=True, key=f"btn_gemini_{sel_asset_id}"):
                            with st.spinner("Querying Gemini multimodal vision (free-tier)..."):
                                st.session_state[f"gemini_pred_{sel_asset_id}"] = client.analyze_gemini_asset(sel_asset_id)
                    with btn_c3:
                        if st.button("🎯 Detect with YOLO", use_container_width=True, key=f"btn_yolo_{sel_asset_id}"):
                            with st.spinner("Running YOLO anomaly localization..."):
                                if hasattr(client, "detect_yolo_asset"):
                                    st.session_state[f"yolo_pred_{sel_asset_id}"] = client.detect_yolo_asset(sel_asset_id)
                                else:
                                    st.session_state[f"yolo_pred_{sel_asset_id}"] = {"status": "UNAVAILABLE", "message": "YOLO client method pending reload."}

                    pred = st.session_state.get(f"ml_pred_{sel_asset_id}")
                    if pred:
                        if pred.get("status") == "MODEL_READY":
                            conf = pred.get("confidence", 0.0)
                            is_low_conf = pred.get("is_low_confidence", False) or conf < 0.50
                            top_cls = pred.get("predicted_class")
                            tier = pred.get("uncertainty_tier", "ACCEPTED_HYPOTHESIS")
                            tax_note = pred.get("taxonomy_note")
                            inp_val = pred.get("input_validation", {})

                            if tier == "AMBIGUOUS_HYPOTHESIS":
                                st.warning(f"⚠️ **Ambiguous AI Hypothesis**: `{top_cls}` (Calibrated Confidence: {conf:.1%})")
                                st.info(f"🔍 {pred.get('message')}")
                            elif is_low_conf:
                                st.warning(f"⚠️ **Low-Confidence AI Hypothesis**: `{top_cls}` (Calibrated Confidence: {conf:.1%})")
                                st.info(f"🔍 {pred.get('message')}")
                            else:
                                st.success(f"**Predicted Class**: `{top_cls}`")
                                st.metric("Calibrated Model Confidence", f"{conf:.1%}")

                            if tax_note:
                                st.info(f"💡 **Benchmark Taxonomy**: {tax_note}")

                            inp_warnings = inp_val.get("warnings", []) if isinstance(inp_val, dict) else []
                            for w in inp_warnings:
                                st.warning(f"⚠️ **Input Geometry / Modality**: {w}")

                            st.caption(f"Model Version: `{pred.get('model_version')}` | Algorithm: `{pred.get('algorithm')}` | Calibrated: `{pred.get('is_calibrated', False)}`")

                            probs = pred.get("probabilities", {})
                            if probs:
                                st.markdown("##### Class Probability Distribution")
                                df_p = pd.DataFrame(list(probs.items()), columns=["Fault Category", "Probability"]).set_index("Fault Category")
                                st.bar_chart(df_p)

                            st.warning(f"**Engineering Disclaimer**: {pred.get('evaluation_disclaimer')}")
                        else:
                            st.error(f"Inference Status: {pred.get('status')} — {pred.get('message')}")

                # Display Gemini Visual Hypothesis below image and ML classifier for comparative analysis
                gemini_res = st.session_state.get(f"gemini_pred_{sel_asset_id}")
                if gemini_res:
                    st.markdown("---")
                    st.subheader("✨ Gemini Multimodal Vision Analysis (Side-by-Side Review)")
                    render_gemini_analysis_card(gemini_res)

                # Display YOLO Localization below image and ML classifier
                yolo_res = st.session_state.get(f"yolo_pred_{sel_asset_id}")
                if yolo_res:
                    st.markdown("---")
                    st.subheader("🎯 YOLO Anomaly Localization & Detection")
                    render_yolo_analysis_card(yolo_res)

    with tab_direct:
        st.subheader("Direct Thermal Crop Upload")
        crop_file = st.file_uploader("Upload 24x40 or arbitrary PV Thermal Crop", type=["jpg", "jpeg", "png"], key="crop_direct")
        if crop_file:
            col_c1, col_c2 = st.columns(2)
            crop_bytes = crop_file.read()
            with col_c1:
                st.image(Image.open(io.BytesIO(crop_bytes)), caption="Uploaded Crop", width=160)
            with col_c2:
                btn_d1, btn_d2, btn_d3 = st.columns(3)
                with btn_d1:
                    if st.button("Classify with ML", use_container_width=True, key=f"btn_ml_crop_{crop_file.name}"):
                        with st.spinner("Extracting features and classifying crop..."):
                            st.session_state[f"ml_crop_{crop_file.name}"] = client.predict_uploaded_image(crop_bytes, crop_file.name)
                with btn_d2:
                    if st.button("✨ Analyze with Gemini", use_container_width=True, key=f"btn_gemini_crop_{crop_file.name}"):
                        with st.spinner("Analyzing crop with Gemini vision..."):
                            st.session_state[f"gemini_crop_{crop_file.name}"] = client.analyze_gemini_upload(crop_bytes, crop_file.name)
                with btn_d3:
                    if st.button("🎯 Detect with YOLO", use_container_width=True, key=f"btn_yolo_crop_{crop_file.name}"):
                        with st.spinner("Detecting anomalies with YOLO..."):
                            if hasattr(client, "detect_yolo_upload"):
                                st.session_state[f"yolo_crop_{crop_file.name}"] = client.detect_yolo_upload(crop_bytes, crop_file.name)
                            else:
                                st.session_state[f"yolo_crop_{crop_file.name}"] = {"status": "UNAVAILABLE", "message": "YOLO client method pending reload."}

                pred = st.session_state.get(f"ml_crop_{crop_file.name}")
                if pred:
                    if pred.get("status") == "MODEL_READY":
                        conf = pred.get("confidence", 0.0)
                        top_cls = pred.get("predicted_class")
                        is_low = pred.get("is_low_confidence", False) or conf < 0.50
                        tax_note = pred.get("taxonomy_note")
                        inp_val = pred.get("input_validation", {})

                        if is_low:
                            st.warning(f"⚠️ **Low-Confidence Hypothesis**: `{top_cls}` ({conf:.1%})")
                            st.info(pred.get("message"))
                        else:
                            st.success(f"**Predicted Class**: `{top_cls}`")
                            st.metric("Calibrated Confidence", f"{conf:.1%}")

                        if tax_note:
                            st.info(f"💡 **Taxonomy Context**: {tax_note}")

                        inp_warnings = inp_val.get("warnings", []) if isinstance(inp_val, dict) else []
                        for w in inp_warnings:
                            st.warning(f"⚠️ {w}")

                        probs = pred.get("probabilities", {})
                        if probs:
                            df_p = pd.DataFrame(list(probs.items()), columns=["Fault Category", "Probability"]).set_index("Fault Category")
                            st.bar_chart(df_p)
                    else:
                        st.error(f"Status: {pred.get('status')} — {pred.get('message')}")

            gemini_crop_res = st.session_state.get(f"gemini_crop_{crop_file.name}")
            if gemini_crop_res:
                st.markdown("---")
                st.subheader("✨ Gemini Multimodal Vision Analysis (Crop Review)")
                render_gemini_analysis_card(gemini_crop_res)

            yolo_crop_res = st.session_state.get(f"yolo_crop_{crop_file.name}")
            if yolo_crop_res:
                st.markdown("---")
                st.subheader("🎯 YOLO Anomaly Localization (Crop Review)")
                render_yolo_analysis_card(yolo_crop_res)

# ==============================================================================
# VIEW 8: DIAGNOSTIC ASSESSMENT
# ==============================================================================
elif selected_view == "8. Diagnostic Assessment":
    st.header("Evidence-Based Diagnostic Reasoning & Engineering Review")
    st.caption("Synthesizes thermal signatures, model predictions, and telemetry into actionable O&M decisions.")

    inspections = client.get_inspections()
    if not inspections:
        st.info("Create an inspection session first.")
    else:
        insp_opts = {f"#{insp['id']} - {insp['code']} ({insp['inspector_name']})": insp["id"] for insp in inspections}
        sel_insp_id = st.selectbox("Select Inspection Session to Diagnose", list(insp_opts.values()), format_func=lambda x: [k for k, v in insp_opts.items() if v == x][0])

        col_act, _ = st.columns([1, 2])
        with col_act:
            if st.button("⚡ Run Multi-Modal AI Diagnosis"):
                with st.spinner("Synthesizing inspection evidence, thermal signatures, and engineering rules..."):
                    ok, res = client.trigger_diagnosis(sel_insp_id)
                if ok:
                    st.success("Diagnostic assessment successfully updated!")
                    st.rerun()
                else:
                    st.error(f"Diagnosis failed: {res}")

        diag = client.get_diagnosis(sel_insp_id)
        if not diag:
            st.info("No diagnostic assessment generated yet. Click 'Run Multi-Modal AI Diagnosis' above.")
        else:
            # Severity Badge
            sev = diag.get("severity", "NORMAL")
            sev_color = {
                "CRITICAL": "🔴",
                "MAJOR": "🟠",
                "MINOR": "🟡",
                "NORMAL": "🟢",
                "INCONCLUSIVE": "⚪",
            }.get(sev, "⚪")

            p_loss_val = diag.get("power_loss_percent")
            p_loss_str = format_power_loss(p_loss_val, fallback="Not quantified (Needs I-V trace)")
            rstatus = diag.get("review_status", "PENDING_REVIEW")
            rstatus_color = "🟢" if rstatus == "CONFIRMED" else ("🔴" if rstatus == "REJECTED" else "🟡")

            insp_det = client.get_inspection_detail(sel_insp_id)
            insp_status = insp_det.get("status", "SCHEDULED") if insp_det else "SCHEDULED"

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Fault Category", diag.get("fault_category"))
            c2.metric("Severity Level", f"{sev_color} {sev}")
            c3.metric("Estimated Yield Loss", p_loss_str)
            c4.metric("Diagnostic Review", f"{rstatus_color} {rstatus}")
            c5.metric("Lifecycle Status", f"🔄 {insp_status}")

            if insp_det and insp_det.get("images"):
                if any(is_synthetic_asset(im) for im in insp_det["images"]):
                    st.info("ℹ️ **Data Provenance**: This inspection contains synthetic/benchmark test imagery uploaded for pipeline validation.")

            if rstatus == "PENDING_REVIEW":
                st.warning("⚠️ **Awaiting Human Engineer Verification**: This assessment reflects an automated AI engineering hypothesis. It has NOT been verified or signed off by a qualified engineer.")

            req_missing = diag.get("required_measurements", [])
            if req_missing:
                st.info(f"📏 **Missing Measurements Required for Physical Confirmation**: {', '.join(req_missing)}")

            st.markdown(f"**Severity Justification**: {diag.get('severity_justification')}")
            if diag.get("model_prediction"):
                conf_val = diag.get("model_confidence")
                conf_str = f"{conf_val:.1%}" if conf_val is not None else "N/A"
                st.caption(f"Corroborating AI Vision Output: `{diag.get('model_prediction')}` (Confidence: {conf_str})")

            # Structured Collapsible Sections
            with st.expander("🔍 Probable Root Causes (Engineering Hypotheses)", expanded=True):
                for rc in diag.get("root_causes", []):
                    st.markdown(f"- {rc}")

            with st.expander("🛠️ Recommended Corrective & Preventive Actions", expanded=True):
                st.markdown("**Corrective Actions:**")
                for ca in diag.get("corrective_actions", []):
                    st.markdown(f"- {ca}")
                st.markdown("**Preventive Measures:**")
                for pa in diag.get("preventive_actions", []):
                    st.markdown(f"- {pa}")

            with st.expander("⚠️ Safety Precautions & Actions to Avoid", expanded=False):
                st.markdown("**Safety Directives:**")
                for sp in diag.get("safety_precautions", []):
                    st.markdown(f"- {sp}")
                st.markdown("**Actions to Avoid:**")
                for aa in diag.get("actions_to_avoid", []):
                    st.markdown(f"- {aa}")

            with st.expander("📏 Required Missing Electrical Measurements", expanded=False):
                for rm in diag.get("required_measurements", []):
                    st.markdown(f"- {rm}")

            st.markdown("---")
            # Human Engineer Review Section
            st.subheader("Human Engineer Review & Sign-Off")
            st.caption("Under IEC 62446-3, only a qualified engineer can certify or confirm a fault diagnosis.")
            with st.form("engineer_review_form"):
                cr1, cr2 = st.columns(2)
                reviewer = cr1.text_input(
                    "Reviewing Engineer Name *",
                    value=diag.get("reviewed_by") or "",
                    placeholder="e.g., J. Doe, PE",
                )

                status_options = ["PENDING_REVIEW", "CONFIRMED", "MODIFIED", "REJECTED"]
                current_status = diag.get("review_status", "PENDING_REVIEW")
                curr_status_idx = status_options.index(current_status) if current_status in status_options else 0
                new_status = cr2.selectbox("Decision Status *", status_options, index=curr_status_idx)

                all_sevs = ["CRITICAL", "MAJOR", "MINOR", "NORMAL", "INCONCLUSIVE"]
                sev_idx = all_sevs.index(sev) if sev in all_sevs else 4
                new_sev = cr1.selectbox("Override Severity (Optional)", all_sevs, index=sev_idx)

                notes = st.text_area(
                    "Engineering Verification Notes",
                    value=diag.get("engineer_notes") or "",
                    placeholder="Enter physical inspection findings, multimeter readings, EL verification notes, or rationale for sign-off...",
                )

                if st.form_submit_button("Submit Engineer Sign-Off"):
                    if new_status in ("CONFIRMED", "MODIFIED") and not reviewer.strip():
                        st.error("Validation Error: Please enter Reviewing Engineer Name to certify or modify this diagnosis.")
                    elif new_status in ("CONFIRMED", "MODIFIED") and not notes.strip():
                        st.error("Validation Error: Please provide Engineering Verification Notes documenting physical evidence or test results.")
                    else:
                        ok, res = client.update_diagnosis_review(diag["id"], new_status, new_sev, reviewer.strip(), notes.strip())
                        if ok:
                            st.success(f"Engineer sign-off recorded! Diagnosis status updated to '{new_status}'.")
                            st.rerun()
                        else:
                            st.error(f"Failed to update review: {res}")

# ==============================================================================
# VIEW 9: TELEMETRY IMPORT & TRENDS
# ==============================================================================
elif selected_view == "9. Telemetry Import & Trends":
    st.header("Electrical Telemetry Ingestion & Time-Series Analytics")
    st.caption("Upload CSV/Excel string and inverter monitoring files to corroborate visual findings with electrical data.")

    plants = client.get_plants()
    if not plants:
        st.warning("Please register a plant first.")
    else:
        plant_options = {p["name"]: p["id"] for p in plants}
        selected_pname = st.selectbox("Select Target Plant", list(plant_options.keys()))
        selected_pid = plant_options[selected_pname]

        tab_import, tab_trends, tab_history = st.tabs(["Import Telemetry (CSV / Excel)", "Telemetry Trends & Curves", "Batch Import History"])

        with tab_import:
            tel_file = st.file_uploader("Select Telemetry File (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])
            if tel_file:
                file_bytes = tel_file.read()
                if st.button("Preview File"):
                    ok, prev = client.preview_telemetry(selected_pid, file_bytes, tel_file.name)
                    if ok:
                        st.success(f"File validated: {prev['valid_rows_count']} valid rows out of {prev['total_rows']} total rows.")
                        st.markdown(f"**Detected Time Range**: `{prev['detected_time_range']['start_time']}` to `{prev['detected_time_range']['end_time']}`")
                        st.json(prev["column_mapping"])
                        if prev.get("validation_warnings"):
                            for w in prev["validation_warnings"]:
                                st.warning(w)
                        st.subheader("Data Preview (First 10 Rows)")
                        st.dataframe(pd.DataFrame(prev["preview_records"]), use_container_width=True)
                    else:
                        st.error(f"Validation failed: {prev}")

                import_notes = st.text_input("Import Notes", placeholder="e.g. Scada export for string inverter 1")
                if st.button("Commit Telemetry to Database"):
                    with st.spinner("Ingesting records into time-series database..."):
                        ok, res = client.commit_telemetry(selected_pid, file_bytes, tel_file.name, import_notes)
                    if ok:
                        st.success(f"Telemetry committed successfully! Batch ID: {res.get('batch', {}).get('id')}")
                        st.rerun()
                    else:
                        st.error(f"Commit failed: {res}")

        with tab_trends:
            hours = st.slider("Time Window (Past Hours)", min_value=1, max_value=168, value=24)
            trends = client.get_telemetry_trends(selected_pid, hours)

            if trends.get("has_data") and trends.get("series"):
                if any(is_synthetic_telemetry(r) for r in trends["series"]):
                    st.warning("⚠️ **SYNTHETIC TEST DATA — NOT REAL PLANT MEASUREMENTS**: These electrical operating values were generated for system testing and do not represent actual plant sensor readings.")
                st.success(f"Loaded {trends['record_count']} time-series data points.")
                df_series = pd.DataFrame(trends["series"])
                df_series["timestamp"] = pd.to_datetime(df_series["timestamp"])
                df_series = df_series.set_index("timestamp")

                c1, c2 = st.columns(2)
                with c1:
                    if "power_kw" in df_series:
                        st.subheader("Power Output (kW)")
                        st.line_chart(df_series[["power_kw"]])
                with c2:
                    if "voltage_v" in df_series:
                        st.subheader("DC Voltage (V)")
                        st.line_chart(df_series[["voltage_v"]])

                c3, c4 = st.columns(2)
                with c3:
                    if "current_a" in df_series:
                        st.subheader("DC Current (A)")
                        st.line_chart(df_series[["current_a"]])
                with c4:
                    if "irradiance_w_m2" in df_series and "module_temp_c" in df_series:
                        st.subheader("Irradiance & Temperature")
                        st.line_chart(df_series[["irradiance_w_m2", "module_temp_c"]])

                st.subheader("Summary Moments")
                st.json(trends.get("summary", {}))
            else:
                st.info("No telemetry records found for this plant in the selected time range. Import data using the tab above.")

        with tab_history:
            batches = client.get_telemetry_batches(selected_pid) if hasattr(client, "get_telemetry_batches") else []
            try:
                # Direct call via client
                import httpx
                with httpx.Client(timeout=5.0) as cl:
                    r = cl.get(f"{client.base_url}/plants/{selected_pid}/telemetry/batches")
                    batches = r.json() if r.status_code == 200 else []
            except Exception:
                batches = []

            if batches:
                st.dataframe(pd.DataFrame(batches)[["id", "filename", "file_type", "records_count", "start_time", "end_time", "status"]], use_container_width=True)
            else:
                st.info("No previous batches imported for this plant.")

# ==============================================================================
# VIEW 10: DIAGNOSTIC CHATBOT
# ==============================================================================
elif selected_view == "10. Diagnostic Chatbot":
    st.header("Contextual Diagnostic Assistant & Engineering Standards RAG")
    st.caption("Ask questions about active inspections, telemetry readings, and technical criteria (IEC 62446-3, IEC 61215).")

    # Select Context
    inspections = client.get_inspections()
    insp_opts = {"General / No Specific Inspection": None}
    for insp in inspections:
        insp_opts[f"Inspection #{insp['id']} ({insp['code']})"] = insp["id"]

    sel_insp_label = st.selectbox("Ground Assistant in Inspection Context", list(insp_opts.keys()))
    sel_insp_id = insp_opts[sel_insp_label]

    # Session Key
    if "chat_session_id" not in st.session_state:
        st.session_state["chat_session_id"] = f"ui_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Suggested Prompts
    st.markdown("##### Quick Technical Questions:")
    cq1, cq2, cq3 = st.columns(3)
    quick_q = None
    if cq1.button("Safety: DC Arc Flash Directives?"):
        quick_q = "What are the safety procedures and hazard precautions for disconnecting high voltage DC solar string cables?"
    if cq2.button("IEC: Thermography Environmental Bounds?"):
        quick_q = "What are the minimum environmental and meteorological conditions required to certify an infrared thermography inspection under IEC 62446-3?"
    if cq3.button("Diagnosis: Missing Electrical Checks?"):
        quick_q = "What additional electrical and I-V curve measurements are required to confirm this fault classification?"

    # Conversation History Display
    history = client.get_chat_history(st.session_state["chat_session_id"])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.caption("Citations: " + " | ".join(msg["citations"]))

    # Chat Input Box
    user_prompt = st.chat_input("Enter your question for the PV diagnostic assistant...")
    prompt_to_send = quick_q or user_prompt

    if prompt_to_send:
        with st.chat_message("user"):
            st.markdown(prompt_to_send)

        with st.spinner("Analyzing plant context and searching engineering standards..."):
            ans = client.send_chat_message(
                query=prompt_to_send,
                session_id=st.session_state["chat_session_id"],
                inspection_id=sel_insp_id,
            )

        resp_text = ans.get("response", "No response generated.")
        is_err = resp_text.startswith("Assistant error") or resp_text.startswith("Network error") or resp_text.startswith("Request timed out")

        with st.chat_message("assistant"):
            if is_err:
                st.error(resp_text)
            else:
                st.markdown(resp_text)
                if ans.get("citations"):
                    st.caption("Citations: " + " | ".join(ans["citations"]))
        if not is_err:
            st.rerun()

# ==============================================================================
# VIEW 11: INSPECTION HISTORY & COMPARE
# ==============================================================================
elif selected_view == "11. Inspection History & Compare":
    st.header("Inspection History & Longitudinal Comparison")
    inspections = client.get_inspections()

    if not inspections:
        st.info("No inspections recorded yet.")
    else:
        tab_list, tab_compare = st.tabs(["All Inspections", "Side-by-Side Comparison"])

        with tab_list:
            df_i = pd.DataFrame(inspections)[["id", "code", "plant_id", "inspector_name", "inspection_date", "status", "ambient_temp_c", "irradiance_w_m2"]]
            st.dataframe(df_i, use_container_width=True)

            with st.expander("Update Inspection Lifecycle Status"):
                c_up1, c_up2, c_up3 = st.columns([2, 2, 1])
                insp_opts_up = {f"#{i['id']} ({i['code']}) — Status: {i['status']}": i for i in inspections}
                sel_label = c_up1.selectbox("Select Inspection Session", list(insp_opts_up.keys()), key="sb_insp_status")
                target_insp = insp_opts_up[sel_label]
                new_status_val = c_up2.selectbox("Target Lifecycle Status", ["SCHEDULED", "IN_PROGRESS", "COMPLETED", "ARCHIVED"], key="sb_new_stat")
                if c_up3.button("Update Status", key="btn_update_status"):
                    ok, res = client.update_inspection(target_insp["id"], {"status": new_status_val})
                    if ok:
                        st.success(f"Inspection #{target_insp['id']} status updated to {new_status_val}!")
                        st.rerun()
                    else:
                        st.error(f"Transition error: {res}")

        with tab_compare:
            st.subheader("Compare Two Inspection Events")
            c1, c2 = st.columns(2)
            insp_map = {f"#{i['id']} ({i['code']})": i["id"] for i in inspections}
            sel_1 = c1.selectbox("Inspection A", list(insp_map.keys()), index=0)
            sel_2 = c2.selectbox("Inspection B", list(insp_map.keys()), index=min(1, len(insp_map)-1))

            id_1, id_2 = insp_map[sel_1], insp_map[sel_2]
            d1 = client.get_inspection_detail(id_1)
            d2 = client.get_inspection_detail(id_2)
            diag1 = client.get_diagnosis(id_1)
            diag2 = client.get_diagnosis(id_2)

            def get_provenance_label(detail):
                if not detail:
                    return "No data"
                imgs = detail.get("images", [])
                if any(is_synthetic_asset(im) for im in imgs):
                    return "⚠️ Contains Synthetic Test Image"
                return f"Field Imagery ({len(imgs)} assets)"

            p_loss_1 = format_power_loss(diag1.get("power_loss_percent")) if diag1 else "N/A"
            p_loss_2 = format_power_loss(diag2.get("power_loss_percent")) if diag2 else "N/A"

            comp_data = {
                "Metric": [
                    "Lifecycle Status",
                    "Inspector",
                    "Timestamp",
                    "Ambient Temp (°C)",
                    "Irradiance (W/m²)",
                    "Data Provenance",
                    "Fault Category",
                    "Severity",
                    "Power Loss Est.",
                    "Review Status",
                ],
                f"Inspection A ({sel_1})": [
                    d1.get("status") if d1 else "N/A",
                    d1.get("inspector_name") if d1 else "N/A",
                    str(d1.get("inspection_date")) if d1 else "N/A",
                    d1.get("ambient_temp_c") if d1 else "N/A",
                    d1.get("irradiance_w_m2") if d1 else "N/A",
                    get_provenance_label(d1),
                    diag1.get("fault_category") if diag1 else "N/A",
                    diag1.get("severity") if diag1 else "N/A",
                    p_loss_1,
                    diag1.get("review_status") if diag1 else "N/A",
                ],
                f"Inspection B ({sel_2})": [
                    d2.get("status") if d2 else "N/A",
                    d2.get("inspector_name") if d2 else "N/A",
                    str(d2.get("inspection_date")) if d2 else "N/A",
                    d2.get("ambient_temp_c") if d2 else "N/A",
                    d2.get("irradiance_w_m2") if d2 else "N/A",
                    get_provenance_label(d2),
                    diag2.get("fault_category") if diag2 else "N/A",
                    diag2.get("severity") if diag2 else "N/A",
                    p_loss_2,
                    diag2.get("review_status") if diag2 else "N/A",
                ],
            }
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

# ==============================================================================
# VIEW 12: BATCH INSPECTION MONITORING
# ==============================================================================
elif selected_view == "12. Batch Inspection Monitoring":
    st.header("Batch Inspection Ingestion & Automated Triage")
    st.caption("Upload multiple thermal module crops simultaneously to execute high-throughput fault triage.")

    plants = client.get_plants()
    if not plants:
        st.warning("Please register a plant first.")
    else:
        plant_options = {p["name"]: p["id"] for p in plants}
        selected_pname = st.selectbox("Select Target Plant", list(plant_options.keys()))
        selected_pid = plant_options[selected_pname]

        with st.form("create_batch_form"):
            b_title = st.text_input("Batch Title *", value=f"Aerial Drone Pass - {datetime.now().strftime('%Y-%m-%d')}")
            batch_files = st.file_uploader("Select Multiple Thermal Module Crops", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            if st.form_submit_button("Run Batch Triage"):
                if not batch_files:
                    st.error("Please upload at least one image file.")
                else:
                    with st.spinner("Creating batch session..."):
                        ok_b, b_res = client.create_batch_inspection(selected_pid, b_title, len(batch_files))

                    if ok_b:
                        batch_id = b_res["id"]
                        file_payload = [(f.name, f.read()) for f in batch_files]
                        with st.spinner(f"Running automated vision inference across {len(batch_files)} images..."):
                            ok_p, p_res = client.process_batch_images(batch_id, file_payload)

                        if ok_p:
                            st.success(f"Batch completed! Processed {p_res['processed_images']} module crops.")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Total Images", p_res["total_images"])
                            c2.metric("Critical/Major Anomalies", p_res["critical_anomalies_count"])
                            c3.metric("Status", p_res["status"])

                            st.subheader("Fault Category Distribution in Batch")
                            st.bar_chart(p_res.get("fault_distribution", {}))

                            st.subheader("Itemized Anomaly Triage Results")
                            res_list = p_res.get("results", [])
                            if res_list:
                                df_triage = pd.DataFrame(res_list)
                                if "power_loss_percent" in df_triage:
                                    df_triage["power_loss_percent"] = df_triage["power_loss_percent"].apply(format_power_loss)
                                st.dataframe(df_triage, use_container_width=True)
                            else:
                                st.dataframe(pd.DataFrame(), use_container_width=True)
                        else:
                            st.error(f"Batch processing error: {p_res}")
                    else:
                        st.error(f"Failed to create batch: {b_res}")

# ==============================================================================
# VIEW 13: PDF ENGINEERING REPORTS
# ==============================================================================
elif selected_view == "13. PDF Engineering Reports":
    st.header("Official PDF Engineering Report Generation")
    st.caption("Generate publication-ready, multi-page inspection dossiers with IEC 62446-3 compliance data.")

    inspections = client.get_inspections()
    if not inspections:
        st.info("No inspections available.")
    else:
        insp_opts = {f"#{insp['id']} - {insp['code']} ({insp['inspector_name']})": insp["id"] for insp in inspections}
        sel_insp_id = st.selectbox("Select Inspection to Compile", list(insp_opts.values()), format_func=lambda x: [k for k, v in insp_opts.items() if v == x][0])

        col_b1, _ = st.columns([1, 2])
        with col_b1:
            if st.button("Compile PDF Engineering Report"):
                with st.spinner("Compiling dossier with ReportLab..."):
                    pdf_bytes = client.download_pdf_report(sel_insp_id)

                if pdf_bytes:
                    st.success("Report successfully generated!")
                    st.download_button(
                        label="⬇️ Download PDF Dossier",
                        data=pdf_bytes,
                        file_name=f"solar_pv_inspection_{sel_insp_id}_report.pdf",
                        mime="application/pdf",
                    )
                else:
                    st.error("Failed to generate PDF report from backend.")

# ==============================================================================
# VIEW 14: MODEL STATUS & METRICS
# ==============================================================================
elif selected_view == "14. Model Status & Metrics":
    st.header("AI Vision Model Status & Empirical Metrics")
    st.caption("Empirically evaluated against the quarantined InfraredSolarModules benchmark.")

    model_status = client.get_vision_model_status()
    st.subheader("Active Model Lifecycle")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", model_status.get("status"))
    c2.metric("Algorithm", model_status.get("algorithm") or "None")
    c3.metric("Version", model_status.get("model_version") or "None")
    c4.metric("Training Samples", model_status.get("training_samples") or 0)

    st.markdown("---")
    st.subheader("Empirical Evaluation Report (Test Partition)")
    eval_data = client.get_vision_evaluation_metrics()

    if eval_data.get("status") == "SUCCESS" and eval_data.get("metrics"):
        m = eval_data["metrics"]
        test_m = m.get("test_metrics", {})

        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("Test Accuracy", f"{test_m.get('overall_accuracy', 0.0):.2%}")
        cm2.metric("Macro-F1 Score", f"{test_m.get('macro_f1', 0.0):.4f}")
        cm3.metric("Weighted-F1", f"{test_m.get('weighted_f1', 0.0):.4f}")
        cm4.metric("Test Sample Count", test_m.get("sample_count", 0))

        st.subheader("Per-Class Precision, Recall, & F1 Scores")
        per_class = test_m.get("per_class_metrics", {})
        if per_class:
            df_pc = pd.DataFrame(per_class).T
            st.dataframe(df_pc, use_container_width=True)

        st.subheader("Confusion Matrix")
        cm = test_m.get("confusion_matrix", [])
        c_names = test_m.get("class_names", [])
        if cm and c_names:
            df_cm = pd.DataFrame(cm, index=c_names, columns=c_names)
            st.dataframe(df_cm, use_container_width=True)
    else:
        st.info("No evaluation report found on disk.")

# ==============================================================================
# VIEW 15: SYSTEM CONFIG & SCADA STATUS
# ==============================================================================
elif selected_view == "15. System Config & SCADA Status":
    st.header("System Configuration & External SCADA Gateway Status")

    st.subheader("Backend Infrastructure")
    st.json(health)

    st.subheader("Industrial SCADA / Modbus Gateway Adapter")
    scada_stat = client.get_scada_status()
    st.json(scada_stat)
    st.info(
        "SCADA Status Notice: Telemetry is currently ingested via CSV and Excel file ingestion. "
        "To enable live polling, configure `SCADA_ENDPOINT_URL` and `SCADA_API_KEY` in `.env`."
    )
