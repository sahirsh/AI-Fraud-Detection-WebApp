import os

import requests
import streamlit as st

API_URL = os.getenv("FRAUD_API_URL", "http://127.0.0.1:5000").rstrip("/")

st.set_page_config(page_title="Job Fraud Detector", layout="centered")
st.title("Job Posting Fraud Detector")
st.caption("Scores a listing via the separate Flask API.")

with st.form("job_form"):
    title = st.text_input("Title")
    company_profile = st.text_area("Company profile", height=80)
    description = st.text_area("Description", height=160)
    requirements = st.text_area("Requirements", height=100)
    benefits = st.text_area("Benefits", height=80)
    submitted = st.form_submit_button("Check posting")

if submitted:
    if not any(
        field.strip()
        for field in (title, company_profile, description, requirements, benefits)
    ):
        st.warning("Enter at least one field before submitting.")
    else:
        payload = {
            "title": title,
            "company_profile": company_profile,
            "description": description,
            "requirements": requirements,
            "benefits": benefits,
        }
        try:
            with st.spinner("Scoring..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=30,
                )
            response.raise_for_status()
            result = response.json()

            probability = float(result.get("fraud_probability", 0))
            risk = result.get("risk_level", "Unknown")
            label = result.get("label")

            st.subheader("Result")
            st.metric("Fraud probability", f"{probability:.1%}")
            st.write(f"**Risk level:** {risk}")
            if label is not None:
                st.write(f"**Label:** {'fraud' if int(label) == 1 else 'legit'}")
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
