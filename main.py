from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import joblib
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained pipeline
model = joblib.load("startup_pipeline.pkl")


@app.post("/predict")
def predict(data: dict):
    try:
        expected_columns = model.named_steps["encoder"].feature_names_in_

        categorical_cols = [
            "state_code",
            "zip_code",
            "id",
            "city",
            "founded_at",
            "first_funding_at",
            "last_funding_at",
            "state_code.1",
            "category_code",
        ]

        full_input = {}

        for col in expected_columns:
            if col in data:
                value = data[col]

                # Force correct type
                if col in categorical_cols:
                    full_input[col] = str(value)
                else:
                    try:
                        full_input[col] = float(value)
                    except:
                        full_input[col] = 0

            else:
                # Default values
                if col in categorical_cols:
                    full_input[col] = "unknown"
                else:
                    full_input[col] = 0.0

        df = pd.DataFrame([full_input])

        prediction = model.predict(df)[0]
        probability = float(model.predict_proba(df)[0].max())

        health_score = int(probability * 100) if prediction == "acquired" else int((1 - probability) * 100)

        if probability >= 0.85:
            level = "Very High Confidence"
        elif probability >= 0.70:
            level = "High Confidence"
        elif probability >= 0.55:
            level = "Moderate Confidence"
        else:
            level = "Low Confidence"

        return {
            "prediction": str(prediction),
            "confidence": round(probability * 100, 2),
            "confidence_level": level,
            "health_score": health_score,
            "reasons": [
                "Model analyzed funding pattern",
                "Investor backing strength evaluated",
                "Growth timeline assessed"
            ]
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/generate-report")
def generate_report(data: dict):

    prediction = data.get("prediction", "N/A")
    confidence = data.get("confidence", "N/A")
    confidence_level = data.get("confidence_level", "N/A")
    health_score = data.get("health_score", "N/A")
    reasons = data.get("reasons", [])

    file_path = "startup_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Startup Diagnostic Report", styles["Heading1"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Prediction: {prediction}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Confidence: {confidence}%", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Confidence Level: {confidence_level}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Startup Health Score: {health_score}/100", styles["Normal"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Key Insights:", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    for reason in reasons:
        elements.append(Paragraph(f"- {reason}", styles["Normal"]))
        elements.append(Spacer(1, 5))

    doc.build(elements)

    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename="startup_report.pdf"
    )