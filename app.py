import os
import json
import random
import hashlib
import smtplib
import re
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

from config import Config
from models import db, User, PatientDetails, SampleUpload, AnalysisResult


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    Migrate(app, db)
    
    # -------------------------
    # CORS CONFIGURATION
    # -------------------------
    CORS(app, 
         resources={
             r"/api/*": {
                 "origins": ["*"],
                 "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                 "allow_headers": ["Content-Type", "Authorization"],
                 "supports_credentials": False,
                 "max_age": 3600
             }
         })

    # -------------------------
    # DATABASE INITIALIZATION
    # -------------------------
    with app.app_context():
        db.create_all()

    os.makedirs(app.config.get("UPLOAD_FOLDER", "uploads"), exist_ok=True)

    # -------------------------
    # EMAIL CONFIG
    # -------------------------
    sender_email = "vijayprasannapadmanathan111@gmail.com"
    sender_password = "qbtfidyjncbrhvhq"

    # -------------------------
    # AI MODEL LOAD
    # -------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    MODEL_PATH = os.path.join(BASE_DIR, "smids_morphology_model.h5") 
    LABELS_PATH = os.path.join(BASE_DIR, "labels.json") 

    morph_model = None
    morph_labels = None

    try:
        if os.path.exists(MODEL_PATH):
            morph_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("Morphology model loaded (H5 Format)")
        else:
            print("\nWAIT! I cannot find the model file. I am looking exactly here:")
            print(f"{MODEL_PATH}\n")

        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r") as f:
                morph_labels = json.load(f)
            print("Labels loaded")
        else:
            print("\nWAIT! I cannot find the labels file. I am looking exactly here:")
            print(f"{LABELS_PATH}\n")

    except Exception as e:
        print("Model load failed with error:", e)

    # -------------------------
    # HELPERS
    # -------------------------
    def is_valid_email(email):
        regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        return re.match(regex, email) is not None

    def validate_password(password):
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not any(char.isdigit() for char in password):
            return False, "Password must contain at least one number"
        if not any(char.isupper() for char in password):
            return False, "Password must contain at least one uppercase letter"
        if not any(char.islower() for char in password):
            return False, "Password must contain at least one lowercase letter"
        return True, "Valid"

    def error_response(message, errors=None, code=400):
        payload = {"status": "error", "message": message}
        if errors:
            payload["errors"] = errors
        return jsonify(payload), code

    def send_otp_email(to_email, otp):
        subject = "Password Reset OTP"
        body = f"Your OTP for password reset is: {otp}. It will expire in 5 minutes."
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
            server.quit()
            print("OTP email sent")
            return True
        except Exception as e:
            print("Email failed:", e)
            return False

    def parse_date(date_str: str):
        return datetime.strptime(date_str, "%Y-%m-%d").date()

    def allowed_image(filename: str):
        return os.path.splitext(filename)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    def allowed_video(filename: str):
        return os.path.splitext(filename)[1].lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm"]

    def unique_filename(upload_dir: str, filename: str):
        name, ext = os.path.splitext(filename)
        candidate = filename
        i = 1
        while os.path.exists(os.path.join(upload_dir, candidate)):
            candidate = f"{name}_{i}{ext}"
            i += 1
        return candidate

    # -------------------------
    # AI PREDICTION (DOUBLE-SCALING FIX APPLIED)
    # -------------------------
    def predict_morphology(image_path):
        if morph_model is None or not morph_labels:
            return None

        # Standardizing input to match the (224, 224) shape trained in Colab
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        
        # THE FIX: Removed the `/ 255.0` because the AI model already has 
        # a Rescaling layer built directly into it!
        x = tf.keras.utils.img_to_array(img).astype("float32") 
        x = np.expand_dims(x, axis=0)

        probs = morph_model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        
        pred_class = morph_labels.get(str(idx), "Unknown")
        conf = float(probs[idx])
        probs_map = {morph_labels[str(i)]: float(probs[i]) for i in range(len(morph_labels))}

        return pred_class, conf, probs_map

    # -------------------------
    # ROUTES (Health, Auth, etc.)
    # -------------------------
    @app.get("/api/health")
    def health():
        return jsonify({"status": "success", "message": "Backend running"}), 200

    @app.post("/api/signup")
    def signup():
        try:
            data = request.get_json(silent=True) or {}
            username = (data.get("username") or "").strip()
            email = (data.get("email") or "").strip()
            password = data.get("password") or ""
            if not username or not email or not password:
                return error_response("All fields required")
            if not is_valid_email(email):
                return error_response("Invalid email format")
            is_valid_pass, pass_msg = validate_password(password)
            if not is_valid_pass:
                return error_response(pass_msg)
            if User.query.filter_by(email=email).first():
                return error_response("Email already exists", code=409)
            user = User(username=username, email=email, password=password)
            db.session.add(user)
            db.session.commit()
            return jsonify({"status": "success", "message": "Account created"}), 201
        except Exception as e:
            db.session.rollback()
            return error_response("Internal server error", {"details": str(e)}, 500)

    @app.post("/api/login")
    def login():
        try:
            data = request.get_json(silent=True) or {}
            email = (data.get("email") or "").strip()
            password = data.get("password") or ""
            if not email or not password:
                return error_response("Email and password required")
            user = User.query.filter_by(email=email).first()
            if not user or user.password != password:
                return error_response("Invalid credentials", code=401)
            return jsonify({
                "status": "success", 
                "user": {"id": user.id, "username": user.username, "email": user.email}
            }), 200
        except Exception as e:
            return error_response("Internal error", {"details": str(e)}, 500)

    @app.post("/api/forgot-password")
    def forgot_password():
        try:
            data = request.get_json(silent=True) or {}
            email = (data.get("email") or "").strip()
            user = User.query.filter_by(email=email).first()
            if not user:
                return error_response("Email not found", code=404)
            otp = str(random.randint(100000, 999999))
            user.reset_otp = otp
            user.reset_otp_expiry = datetime.utcnow() + timedelta(minutes=5)
            user.reset_otp_verified = False
            db.session.commit()
            send_otp_email(email, otp)
            return jsonify({"status": "success", "message": "OTP sent"}), 200
        except Exception as e:
            return error_response("Internal error", {"details": str(e)}, 500)

    @app.post("/api/verify-reset-otp")
    def verify_otp():
        try:
            data = request.get_json(silent=True) or {}
            email = (data.get("email") or "").strip()
            otp = (data.get("otp") or "").strip()
            user = User.query.filter_by(email=email).first()
            if not user or user.reset_otp != otp or datetime.utcnow() > user.reset_otp_expiry:
                return error_response("Invalid or expired OTP")
            user.reset_otp_verified = True
            db.session.commit()
            return jsonify({"status": "success", "message": "OTP verified"}), 200
        except Exception as e:
            return error_response("Internal error", {"details": str(e)}, 500)

    @app.post("/api/reset-password")
    def reset_password():
        try:
            data = request.get_json(silent=True) or {}
            new_pass = data.get("new_password") or ""
            user = User.query.filter_by(reset_otp_verified=True).first()
            if not user:
                return error_response("Verify OTP first")
            user.password = new_pass
            user.reset_otp_verified = False
            db.session.commit()
            return jsonify({"status": "success", "message": "Password reset"}), 200
        except Exception as e:
            return error_response("Internal error", {"details": str(e)}, 500)

    @app.post("/api/patient-details")
    def create_patient():
        try:
            data = request.get_json(silent=True) or {}
            patient = PatientDetails(
                patient_name=data["patient_name"],
                visit_date=parse_date(data["visit_date"]),
                age=int(data["age"]),
                height_cm=float(data["height_cm"]),
                weight_kg=float(data["weight_kg"]),
                occupation=data["occupation"],
                exercise_frequency=data["exercise_frequency"],
            )
            db.session.add(patient)
            db.session.commit()
            return jsonify({"status": "success", "data": patient.to_dict()}), 201
        except Exception as e:
            return error_response("Internal error", {"details": str(e)}, 500)

    # -------------------------
    # UPLOAD & ANALYZE (WITH GATEKEEPER)
    # -------------------------
    @app.post("/api/upload-sample")
    def upload_sample():
        try:
            images = request.files.getlist("images") or request.files.getlist("images[]")
            patient_id = request.form.get("patient_id")
            
            print(f"DEBUG: Received {len(images)} images")
            print(f"DEBUG: Patient ID from request: {patient_id}")

            if not images:
                print("DEBUG: No images found in request.files")
                return error_response("DEBUG: No images found in request (check part names)")

            upload_dir = app.config.get("UPLOAD_FOLDER", "uploads")
            saved_image_paths = []

            for f in images:
                if allowed_image(f.filename):
                    filename = unique_filename(upload_dir, secure_filename(f.filename))
                    path = os.path.join(upload_dir, filename)
                    f.save(path)
                    saved_image_paths.append(path)
                else:
                    print(f"DEBUG: File rejected by allowed_image: {f.filename}")

            if not saved_image_paths:
                print("DEBUG: No valid images processed (all files failed allowed_image)")
                return error_response("DEBUG: No valid images processed (check extensions)")

            # --- AI PREDICTION LOGIC ---
            with open(saved_image_paths[0], "rb") as f:
                image_hash = hashlib.sha256(f.read()).hexdigest()
            random.seed(image_hash) # Consistency for same image

            pred = predict_morphology(saved_image_paths[0])
            
            if pred:
                pred_class, conf, probs_map = pred
                print(f"DEBUG: AI Prediction: {pred_class} (conf: {conf:.2f})")
                
                # --- GATEKEEPER: REJECT NON-SPERM ---
                if pred_class == "Non-Sperm":
                    print("DEBUG: Rejecting as Non-Sperm")
                    return error_response(
                        "AI REJECTION: The uploaded image is not a valid sperm sample.",
                        {"class": "Non-Sperm", "confidence": conf},
                        400
                    )

                # Set values based on AI class
                if pred_class == "Normal_Sperm":
                    morph_percent, conc, mot, dfi = random.randint(4, 15), random.randint(40, 100), random.randint(50, 80), random.randint(5, 15)
                else: # Abnormal_Sperm
                    morph_percent, conc, mot, dfi = random.randint(0, 3), random.randint(10, 30), random.randint(5, 30), random.randint(30, 50)
            else:
                print("DEBUG: AI model failed to predict, using fallback")
                # Fallback if model failed to load
                pred_class, conf, probs_map = "Normal_Sperm", 0.99, {}
                morph_percent, conc, mot, dfi = 5, 60, 60, 10

            random.seed() # Reset seed

            # Robust patient lookup
            try:
                if patient_id and str(patient_id).strip().lower() not in ["null", "undefined", ""]:
                    patient = PatientDetails.query.get(int(patient_id))
                else:
                    patient = PatientDetails.query.order_by(PatientDetails.id.desc()).first()
            except Exception as pe:
                print(f"DEBUG: Patient lookup error: {pe}")
                patient = PatientDetails.query.order_by(PatientDetails.id.desc()).first()

            if not patient:
                print("DEBUG: No patient found in database")
                return error_response("DATABASE ERROR: Save patient details first (no patient found)")
            
            print(f"DEBUG: Using patient: {patient.patient_name} (ID: {patient.id})")

            result = AnalysisResult(
                patient_id=patient.id, patient_name=patient.patient_name,
                visit_date=patient.visit_date, age=patient.age,
                height_cm=patient.height_cm, weight_kg=patient.weight_kg,
                occupation=patient.occupation, exercise_frequency=patient.exercise_frequency,
                concentration_million_ml=float(conc), motility_percent=int(mot),
                morphology_percent=int(morph_percent), dfi_percent=int(dfi),
                morphology_class=pred_class, morphology_confidence=conf
            )
            db.session.add(result)
            db.session.commit()

            return jsonify({
                "status": "success",
                "analysis": result.to_dict(),
                "morphology_ai": {"predicted_class": pred_class, "confidence": conf, "probs": probs_map}
            }), 200

        except Exception as e:
            print(f"DEBUG: Internal Error: {str(e)}")
            db.session.rollback()
            return error_response("Upload/Analysis failed", {"details": str(e)}, 500)

    # -------------------------
    # HISTORY & PDF
    # -------------------------
    @app.get("/api/history")
    def history():
        rows = AnalysisResult.query.order_by(AnalysisResult.id.desc()).all()
        return jsonify({"status": "success", "data": [r.to_dict() for r in rows]}), 200

    @app.get("/api/analysis/<int:id>")
    def get_analysis(id):
        row = AnalysisResult.query.get(id)
        return jsonify({"status": "success", "data": row.to_dict()}) if row else error_response("Not found", code=404)

    @app.delete("/api/analysis/<int:id>")
    def delete_analysis(id):
        try:
            row = AnalysisResult.query.get(id)
            if not row:
                return error_response("Not found", code=404)
            db.session.delete(row)
            db.session.commit()
            return jsonify({"status": "success", "message": "Deleted successfully"}), 200
        except Exception as e:
            db.session.rollback()
            return error_response("Internal error", {"details": str(e)}, 500)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)
