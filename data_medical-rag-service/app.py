"""Gradio interface for the Smart Healthcare Diagnostic Hub."""
from __future__ import annotations

import json
from typing import Any, Dict

import gradio as gr

from core.llm_service import generate_patient_report
from core.model_adapters import DLModelAdapter, MLModelAdapter


ml_adapter = MLModelAdapter()
dl_adapter = DLModelAdapter()


def _yes_no(value: float | int) -> str:
    return "Yes" if int(value or 0) == 1 else "No"


def _patient_summary(features: Dict[str, Any], symptoms: str) -> str:
    return (
        "Patient tabular inputs for Diabetes_binary ML model: "
        f"HighBP={_yes_no(features['HighBP'])}, HighChol={_yes_no(features['HighChol'])}, "
        f"BMI={features['BMI']}, Smoker={_yes_no(features['Smoker'])}, Stroke={_yes_no(features['Stroke'])}, "
        f"HeartDiseaseorAttack={_yes_no(features['HeartDiseaseorAttack'])}, "
        f"PhysActivity={_yes_no(features['PhysActivity'])}, "
        f"HvyAlcoholConsump={_yes_no(features['HvyAlcoholConsump'])}, "
        f"NoDocbcCost={_yes_no(features['NoDocbcCost'])}, GenHlth={features['GenHlth']}, "
        f"MentHlth={features['MentHlth']}, PhysHlth={features['PhysHlth']}, "
        f"DiffWalk={_yes_no(features['DiffWalk'])}, Age category={features['Age']}, "
        f"Education={features['Education']}, Income={features['Income']}. "
        f"Symptoms/notes: {symptoms or 'None provided'}."
    )


def run_diagnostic_hub(
    high_bp,
    high_chol,
    bmi,
    smoker,
    stroke,
    heart_disease_or_attack,
    phys_activity,
    heavy_alcohol,
    no_doc_cost,
    gen_hlth,
    ment_hlth,
    phys_hlth,
    diff_walk,
    age,
    education,
    income,
    symptoms,
    xray_image,
):
    features: Dict[str, Any] = {
        "HighBP": high_bp,
        "HighChol": high_chol,
        "BMI": bmi,
        "Smoker": smoker,
        "Stroke": stroke,
        "HeartDiseaseorAttack": heart_disease_or_attack,
        "PhysActivity": phys_activity,
        "HvyAlcoholConsump": heavy_alcohol,
        "NoDocbcCost": no_doc_cost,
        "GenHlth": gen_hlth,
        "MentHlth": ment_hlth,
        "PhysHlth": phys_hlth,
        "DiffWalk": diff_walk,
        "Age": age,
        "Education": education,
        "Income": income,
    }

    ml_result = ml_adapter.predict(features).to_dict()
    dl_result = dl_adapter.predict(xray_image).to_dict()

    patient_summary = _patient_summary(features, symptoms)

    llm_output = generate_patient_report(
        patient_summary=patient_summary,
        ml_result=ml_result,
        dl_result=dl_result,
    )

    # 🔥 مهم جدًا: عشان نشوف لو فيه مشكلة
    print("\n========== LLM REPORT ==========")
    print(llm_output["report"])
    print("================================\n")

    model_outputs = {
        "ml_result": ml_result,
        "dl_result": dl_result,
        "rag_query": llm_output["rag_query"],
        "evidence": llm_output["evidence"],
    }

    return llm_output["report"], json.dumps(model_outputs, ensure_ascii=False, indent=2)


with gr.Blocks(title="Smart Healthcare Diagnostic Hub") as demo:
    gr.Markdown("# Smart Healthcare Diagnostic Hub")
    gr.Markdown(
        "النظام بيستخدم ML لتوقع خطر السكري + DL لتحليل X-ray + RAG + Gemini لعمل تقرير طبي متكامل."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## ML inputs (Diabetes Risk)")

            high_bp = gr.Radio([0, 1], value=1, label="HighBP (0=No, 1=Yes)")
            high_chol = gr.Radio([0, 1], value=1, label="HighChol (0=No, 1=Yes)")
            bmi = gr.Number(label="BMI", value=28.0)
            smoker = gr.Radio([0, 1], value=0, label="Smoker (0=No, 1=Yes)")
            stroke = gr.Radio([0, 1], value=0, label="Stroke (0=No, 1=Yes)")
            heart_disease_or_attack = gr.Radio([0, 1], value=0, label="HeartDiseaseorAttack (0=No, 1=Yes)")
            phys_activity = gr.Radio([0, 1], value=1, label="PhysActivity (0=No, 1=Yes)")
            heavy_alcohol = gr.Radio([0, 1], value=0, label="HvyAlcoholConsump (0=No, 1=Yes)")
            no_doc_cost = gr.Radio([0, 1], value=0, label="NoDocbcCost (0=No, 1=Yes)")

            gen_hlth = gr.Slider(1, 5, value=3, step=1, label="GenHlth (1=Excellent, 5=Poor)")
            ment_hlth = gr.Slider(0, 30, value=0, step=1, label="MentHlth days")
            phys_hlth = gr.Slider(0, 30, value=0, step=1, label="PhysHlth days")

            diff_walk = gr.Radio([0, 1], value=0, label="DiffWalk (0=No, 1=Yes)")
            age = gr.Slider(1, 13, value=9, step=1, label="Age category (1-13)")
            education = gr.Slider(1, 6, value=5, step=1, label="Education (1-6)")
            income = gr.Slider(1, 8, value=6, step=1, label="Income (1-8)")

            symptoms = gr.Textbox(
                label="Symptoms / Notes",
                lines=4,
                placeholder="مثال: fatigue, thirst, cough, fever...",
            )

            gr.Markdown("## DL input - Chest X-ray")
            xray_image = gr.Image(label="Chest X-ray Image", type="pil")

            submit = gr.Button("Generate AI Report", variant="primary")

        with gr.Column():
            gr.Markdown("## Final LLM Report")

            # ✅ تم التعديل هنا
            report = gr.Textbox(
                label="Gemini Report",
                lines=22,
            )

            debug_json = gr.Code(
                label="Raw Model / RAG Outputs",
                language="json",
            )

    submit.click(
        fn=run_diagnostic_hub,
        inputs=[
            high_bp,
            high_chol,
            bmi,
            smoker,
            stroke,
            heart_disease_or_attack,
            phys_activity,
            heavy_alcohol,
            no_doc_cost,
            gen_hlth,
            ment_hlth,
            phys_hlth,
            diff_walk,
            age,
            education,
            income,
            symptoms,
            xray_image,
        ],
        outputs=[report, debug_json],
    )


if __name__ == "__main__":
    demo.launch()