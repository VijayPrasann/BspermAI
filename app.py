import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_migrate import Migrate

from config import Config
from models import db, User, PatientDetails, SampleUpload, AnalysisResult


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # DB + Migrations + CORS
    db.init_app(app)
    Migrate(app, db)
    CORS(app)

    # Ensure upload folder exists
    os.makedirs(app.config.get("UPLOAD_FOLDER", "uploads"), exist_ok=True)

    # -------------------------
    # OPTIONAL: SKIP MODEL LOAD DURING MIGRATION
    # In PowerShell:
    #   $env:SKIP_MODEL_LOAD="1"; flask db upgrade
    # -------------------------
    SKIP_MODEL_LOAD = os.environ.get("SKIP_MODEL_LOAD", "0") == "1"

    # -------------------------
    # LOAD AI MODEL (SMIDS)
    # Put these files next to app.py:
    #   smids_morphology_model.keras
    #   labels.json
    # -------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "smids_morphology_model.keras")
    LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

    morph_model = None
    morph_labels = None

    if not SKIP_MODEL_LOAD:
        try:
            if os.path.exists(MODEL_PATH):
                morph_model = tf.keras.models.load_model(MODEL_PATH)
                print("✅ Morphology model loaded:", MODEL_PATH)

            if os.path.exists(LABELS_PATH):
                with open(LABELS_PATH, "r") as f:
                    morph_labels = json.load(f)  # list like ["Abnormal_Sperm","Non-Sperm","Normal_Sperm"]
                print("✅ Labels loaded:", LABELS_PATH)

        except Exception as e:
            print("⚠️ Model load failed:", e)
            morph_model = None
            morph_labels = None
    else:
        print("⏭️ SKIP_MODEL_LOAD=1 → Model load skipped (safe for migrations).")

    # -------------------------
    # HELPERS
    # -------------------------
    def error_response(message, errors=None, code=400):
        payload = {"status": "error", "message": message}
        if errors is not None:
            payload["errors"] = errors
        return jsonify(payload), code

    def parse_date(date_str: str):
        return datetime.strptime(date_str, "%Y-%m-%d").date()

    def allowed_image(filename: str) -> bool:
        ext = os.path.splitext(filename.lower())[1]
        return ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    def allowed_video(filename: str) -> bool:
        ext = os.path.splitext(filename.lower())[1]
        return ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]

    def unique_filename(upload_dir: str, filename: str) -> str:
        name, ext = os.path.splitext(filename)
        candidate = filename
        i = 1
        while os.path.exists(os.path.join(upload_dir, candidate)):
            candidate = f"{name}_{i}{ext}"
            i += 1
        return candidate

    def predict_morphology(image_path: str):
        """
        Returns:
          predicted_class, confidence, probs_map
        """
        if morph_model is None or not morph_labels:
            return None

        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(img).astype("float32") / 255.0
        x = np.expand_dims(x, axis=0)

        probs = morph_model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        pred_class = morph_labels[idx]
        conf = float(probs[idx])
        probs_map = {morph_labels[i]: float(probs[i]) for i in range(len(morph_labels))}
        return pred_class, conf, probs_map

    # -------------------------
    # HEALTH CHECK
    # -------------------------
    @app.get("/api/health")
    def health():
        return jsonify({"status": "success", "message": "Flask backend running"}), 200

    # -------------------------
    # SIGNUP
    # -------------------------
    @app.post("/api/signup")
    def signup():
        try:
            data = request.get_json(silent=True) or {}

            username = (data.get("username") or "").strip()
            email = (data.get("email") or "").strip()
            password = data.get("password") or ""
            confirm_password = data.get("confirm_password") or ""

            if not username or not email or not password or not confirm_password:
                return error_response("All fields required")

            if password != confirm_password:
                return error_response("Passwords do not match")

            if User.query.filter_by(email=email).first():
                return error_response("Email already registered", code=409)

            user = User(username=username, email=email, password=password)
            db.session.add(user)
            db.session.commit()

            return jsonify({"status": "success", "message": "Account created"}), 201

        except Exception as e:
            db.session.rollback()
            return error_response("Internal server error", {"details": str(e)}, 500)

    # -------------------------
    # LOGIN
    # -------------------------
    @app.post("/api/login")
    def login():
        try:
            data = request.get_json(silent=True) or {}
            email = (data.get("email") or "").strip()
            password = data.get("password") or ""

            if not email or not password:
                return error_response("Email and password required")

            user = User.query.filter_by(email=email).first()
            if not user:
                return error_response("No account found", code=404)

            if user.password != password:
                return error_response("Invalid password", code=401)

            return jsonify({
                "status": "success",
                "message": "Login successful",
                "data": {"id": user.id, "username": user.username, "email": user.email}
            }), 200

        except Exception as e:
            return error_response("Internal server error", {"details": str(e)}, 500)

    # -------------------------
    # PATIENT DETAILS - CREATE
    # -------------------------
    @app.post("/api/patient-details")
    def create_patient():
        try:
            data = request.get_json(silent=True) or {}

            required = [
                "patient_name", "visit_date", "age",
                "height_cm", "weight_kg",
                "occupation", "exercise_frequency",
            ]
            missing = [k for k in required if data.get(k) in (None, "", [])]
            if missing:
                return error_response("Missing required fields", {"missing": missing}, 400)

            visit_date = parse_date(str(data["visit_date"]).strip())
            age = int(data["age"])
            height_cm = float(data["height_cm"])
            weight_kg = float(data["weight_kg"])

            row = PatientDetails(
                patient_name=data["patient_name"],
                visit_date=visit_date,
                age=age,
                height_cm=height_cm,
                weight_kg=weight_kg,
                occupation=data["occupation"],
                exercise_frequency=data["exercise_frequency"],
            )

            db.session.add(row)
            db.session.commit()

            return jsonify({
                "status": "success",
                "message": "Patient details saved successfully",
                "data": row.to_dict(),
            }), 201

        except Exception as e:
            db.session.rollback()
            return error_response("Internal server error", {"details": str(e)}, 500)

    # -------------------------
    # UPLOAD SAMPLE + SAVE RESULT
    # -------------------------
    @app.post("/api/upload-sample")
    def upload_sample():
        try:
            # ✅ accept both images & images[] keys
            images = request.files.getlist("images") or request.files.getlist("images[]")
            videos = request.files.getlist("videos") or request.files.getlist("videos[]")

            # patient_id can be sent from Android (optional)
            patient_id = request.form.get("patient_id")

            if not images and not videos:
                return error_response("No files uploaded", code=400)

            upload_dir = app.config.get("UPLOAD_FOLDER", "uploads")
            os.makedirs(upload_dir, exist_ok=True)

            saved_files = []
            saved_image_paths = []

            # Save images
            for f in images:
                original = f.filename or "image.jpg"
                if not allowed_image(original):
                    return error_response(
                        "Unsupported image format",
                        {"allowed": [".jpg", ".jpeg", ".png", ".bmp", ".webp"], "got": original},
                        400
                    )
                filename = secure_filename(original)
                filename = unique_filename(upload_dir, filename)
                path = os.path.join(upload_dir, filename)
                f.save(path)

                db.session.add(SampleUpload(user_id=None, sample_type="image", file_path=path))
                saved_files.append({"type": "image", "file": filename})
                saved_image_paths.append(path)

            # Save videos
            for f in videos:
                original = f.filename or "video.mp4"
                if not allowed_video(original):
                    return error_response(
                        "Unsupported video format",
                        {"allowed": [".mp4", ".mov", ".avi", ".mkv", ".webm"], "got": original},
                        400
                    )
                filename = secure_filename(original)
                filename = unique_filename(upload_dir, filename)
                path = os.path.join(upload_dir, filename)
                f.save(path)

                db.session.add(SampleUpload(user_id=None, sample_type="video", file_path=path))
                saved_files.append({"type": "video", "file": filename})

            # -------------------------
            # MORPHOLOGY MODEL (use first image)
            # -------------------------
            morphology_percent = 0
            morphology_ai = None

            if saved_image_paths:
                pred = predict_morphology(saved_image_paths[0])
                if pred:
                    pred_class, conf, probs_map = pred
                    morphology_ai = {
                        "predicted_class": pred_class,
                        "confidence": conf,
                        "probs": probs_map
                    }
                    # ✅ simple mapping for % (demo)
                    if pred_class == "Normal_Sperm":
                        morphology_percent = 60
                    elif pred_class == "Abnormal_Sperm":
                        morphology_percent = 30
                    else:
                        morphology_percent = 10

            # Dummy values (replace later with real AI)
            concentration = 45
            motility = 62
            dfi = 18

            # -------------------------
            # SAVE INTO analysis_results TABLE
            # -------------------------
            patient_row = None
            if patient_id:
                patient_row = PatientDetails.query.get(int(patient_id))

            # if patient not passed, use last patient record (optional)
            if patient_row is None:
                patient_row = PatientDetails.query.order_by(PatientDetails.id.desc()).first()

            if patient_row is None:
                return error_response("No patient found. Save patient details first.", code=400)

            result_row = AnalysisResult(
                patient_id=patient_row.id,

                patient_name=patient_row.patient_name,
                visit_date=patient_row.visit_date,
                age=patient_row.age,
                height_cm=patient_row.height_cm,
                weight_kg=patient_row.weight_kg,
                occupation=patient_row.occupation,
                exercise_frequency=patient_row.exercise_frequency,

                concentration_million_ml=float(concentration),
                motility_percent=int(motility),
                morphology_percent=int(morphology_percent),
                dfi_percent=int(dfi),
            )

            db.session.add(result_row)
            db.session.commit()

            return jsonify({
                "status": "success",
                "message": "Upload successful",
                "uploaded": saved_files,

                "patient": patient_row.to_dict(),
                "result": result_row.to_dict(),
                "morphology_ai": morphology_ai
            }), 200

        except Exception as e:
            db.session.rollback()
            return error_response("Internal server error", {"details": str(e)}, 500)

    # -------------------------
    # ✅ HISTORY API (FIX for 404/500)
    # -------------------------
    @app.get("/api/history")
    def history():
        try:
            rows = AnalysisResult.query.order_by(AnalysisResult.id.desc()).all()
            return jsonify({
                "status": "success",
                "count": len(rows),
                "data": [r.to_dict() for r in rows]
            }), 200
        except Exception as e:
            return error_response("Internal server error", {"details": str(e)}, 500)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)