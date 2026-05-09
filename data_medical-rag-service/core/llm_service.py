"""Gemini-powered explanation layer for the Smart Healthcare Diagnostic Hub."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from core.config import settings
from core.rag_engine import get_context_debug


DISCLAIMER = (
    "This AI report is for educational/project support only and is not a medical diagnosis. "
    "A qualified clinician must review any healthcare decision."
)


SYSTEM_INSTRUCTIONS = """
You are the LLM explanation layer in a Smart Healthcare Diagnostic Hub for a university/project demo.
The project has two model outputs:
1. ML tabular model trained from the notebook to predict Diabetes_binary / Diabetes Risk.
2. DL DenseNet121 chest X-ray model trained from the notebook to predict Normal X-ray vs Pneumonia.

Your job is to explain why the models may have produced their predictions by connecting:
- the model labels and probabilities,
- the patient tabular inputs and symptoms,
- the retrieved RAG medical evidence.

Safety and grounding rules:
1. Do not claim the patient definitely has diabetes or pneumonia.
2. Use cautious language: "the model may have predicted this because...", "this may be consistent with...".
3. Do not invent symptoms, medications, doses, lab values, sources, or diagnoses.
4. If ML/DL status is not "ok", clearly say that component is not connected/skipped/failed.
5. Every medical explanation or recommendation should cite provided evidence IDs like [Evidence 1].
6. If evidence is weak or not directly relevant, say that explicitly.
7. Mention urgent clinical review for red flags such as severe chest pain, severe shortness of breath, confusion, blue lips, fainting, or symptoms of hyperglycemic crisis.
8. Write in clear Arabic by default. Keep medical terms understandable.
9. End with the required disclaimer.
""".strip()


REPORT_TEMPLATE = """
اكتب تقريرًا تفسيريًا بالعربية يجاوب على سؤال: لماذا قد يكون الموديل أعطى هذه النتيجة؟

استخدم الهيكل التالي:

## 1) ملخص النتيجة
- اذكر نتيجة ML الخاصة بخطر السكري Diabetes Risk إن كانت متاحة.
- اذكر نتيجة DL الخاصة بصورة الأشعة Normal/Pneumonia إن كانت متاحة.
- وضح أن هذه توقعات AI وليست تشخيصًا.

## 2) لماذا قد يكون ML توقع Diabetes Risk؟
- اربط النتيجة بالمدخلات الرقمية المتاحة مثل BMI, HighBP, HighChol, GenHlth, PhysActivity, Age category, HeartDiseaseorAttack إن وُجدت.
- لا تذكر عوامل غير موجودة في Input JSON.
- لو ML غير متصل أو فشل، اشرح أنه لا يمكن تفسير نتيجة ML حقيقية.

## 3) لماذا قد يكون DL توقع Pneumonia أو Normal X-ray؟
- فسّر label و probability لصورة الأشعة بحذر.
- لو لم يتم رفع صورة أو DL غير متصل، اذكر ذلك بوضوح.

## 4) الأدلة الطبية المستخدمة من RAG
- لخّص الأدلة المرتبطة بالسكري أو الالتهاب الرئوي حسب الحالة.
- استخدم citations مثل [Evidence 1] بجانب أي معلومة طبية.

## 5) ماذا يعني ذلك للمستخدم؟
- أعطِ خطوات آمنة عامة: مراجعة طبيب، متابعة فحوصات، تقييم الأعراض.
- ممنوع وصف أدوية أو جرعات.
- اربط النصائح بالأدلة عند الإمكان.

## 6) علامات خطر تستدعي مراجعة عاجلة
- قائمة قصيرة بالعلامات الخطرة المناسبة للسكري/التنفس/الأشعة.

## 7) حدود التفسير
- وضح أن التفسير يعتمد على جودة البيانات، طريقة تدريب الموديلات، والـRAG evidence.

## 8) تنبيه طبي
- أضف التنبيه الطبي المطلوب بالنص.
""".strip()


class GeminiLLMClient:
    def __init__(self) -> None:
        self.api_key = settings.gemini_api_key
        self.model_name = settings.gemini_model_name

    def _generate_with_google_genai(self, prompt: str) -> str:
        from google import genai

        client = genai.Client(api_key=self.api_key)

        response = client.models.generate_content(
            model=self.model_name,
            contents=SYSTEM_INSTRUCTIONS + "\n\n" + prompt,
        )

        text = getattr(response, "text", None)

        if text:
            return text

        return f"Gemini returned no text. Raw response:\n{response}"

    def generate(self, prompt: str) -> str:
        if not self.api_key or self.api_key.strip() == "" or self.api_key == "your_gemini_api_key_here":
            return (
                "Gemini API key is missing. Create a .env file and set GEMINI_API_KEY.\n\n"
                + DISCLAIMER
            )

        try:
            text = self._generate_with_google_genai(prompt).strip()

            if text:
                return text

            return "Gemini returned an empty response.\n\n" + DISCLAIMER

        except Exception as exc:
            return f"Gemini generation failed:\n{exc}\n\n{DISCLAIMER}"


def build_rag_query(
    patient_summary: str,
    ml_result: Dict[str, Any],
    dl_result: Dict[str, Any],
) -> str:
    terms: List[str] = [patient_summary]

    ml_label = str(ml_result.get("label", ""))
    dl_label = str(dl_result.get("label", ""))

    combined = f"{patient_summary} {ml_label} {dl_label}".lower()

    if ml_result.get("status") == "ok" or "diabetes" in combined or "risk" in combined:
        terms.append(
            "diabetes risk factors BMI high blood pressure high cholesterol physical activity general health clinical explanation"
        )

    if dl_result.get("status") == "ok" or "pneumonia" in combined or "x-ray" in combined:
        terms.append(
            "pneumonia chest x-ray symptoms evaluation respiratory infection urgent care"
        )

    return " ".join(t for t in terms if t).strip() or "diabetes pneumonia AI prediction explanation"


def retrieve_evidence(
    query: str,
    top_k: int | None = None,
) -> Tuple[str, List[str]]:
    info = get_context_debug(query, top_k=top_k or settings.rag_top_k)

    docs = [
        doc
        for doc, _score in info.get("reranked", [])[: top_k or settings.rag_top_k]
    ]

    evidence_lines: List[str] = []

    for i, doc in enumerate(docs, start=1):
        clean = " ".join(str(doc).split())
        evidence_lines.append(f"[Evidence {i}] {clean}")

    return "\n\n".join(evidence_lines), evidence_lines


def build_prompt(
    patient_summary: str,
    ml_result: Dict[str, Any],
    dl_result: Dict[str, Any],
    evidence_context: str,
) -> str:
    payload = {
        "patient_summary": patient_summary,
        "ml_model_output": ml_result,
        "dl_model_output": dl_result,
        "retrieved_evidence": evidence_context,
    }

    return f"""
{REPORT_TEMPLATE}

Input JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def generate_patient_report(
    patient_summary: str,
    ml_result: Dict[str, Any],
    dl_result: Dict[str, Any],
) -> Dict[str, Any]:
    query = build_rag_query(patient_summary, ml_result, dl_result)
    evidence_context, evidence_lines = retrieve_evidence(query)

    prompt = build_prompt(
        patient_summary=patient_summary,
        ml_result=ml_result,
        dl_result=dl_result,
        evidence_context=evidence_context,
    )

    report = GeminiLLMClient().generate(prompt)

    if DISCLAIMER.lower() not in report.lower():
        report = report.rstrip() + "\n\n" + DISCLAIMER

    return {
        "rag_query": query,
        "evidence": evidence_lines,
        "report": report,
    }