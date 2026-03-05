from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class SampleUpload(db.Model):
    __tablename__ = "sample_uploads"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True)

    sample_type = db.Column(db.String(20), nullable=False)  # image/video
    file_path = db.Column(db.String(255), nullable=False)

    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class PatientDetails(db.Model):
    __tablename__ = "patient_details"

    id = db.Column(db.Integer, primary_key=True)

    patient_name = db.Column(db.String(120), nullable=False)
    visit_date = db.Column(db.Date, nullable=False)

    age = db.Column(db.Integer, nullable=False)
    height_cm = db.Column(db.Float, nullable=False)
    weight_kg = db.Column(db.Float, nullable=False)

    occupation = db.Column(db.String(120), nullable=False)
    exercise_frequency = db.Column(db.String(120), nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "patient_name": self.patient_name,
            "visit_date": self.visit_date.isoformat(),
            "age": self.age,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "occupation": self.occupation,
            "exercise_frequency": self.exercise_frequency,
            "created_at": self.created_at.isoformat(),
        }


class AnalysisResult(db.Model):
    """
    ✅ history table
    stores snapshot of patient details + AI results
    """
    __tablename__ = "analysis_results"

    id = db.Column(db.Integer, primary_key=True)

    # link to patient row (optional)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient_details.id"), nullable=True)
    patient = db.relationship("PatientDetails", backref="results", lazy=True)

    # patient snapshot
    patient_name = db.Column(db.String(120), nullable=False)
    visit_date = db.Column(db.Date, nullable=False)

    age = db.Column(db.Integer, nullable=False)
    height_cm = db.Column(db.Float, nullable=False)
    weight_kg = db.Column(db.Float, nullable=False)

    occupation = db.Column(db.String(120), nullable=False)
    exercise_frequency = db.Column(db.String(120), nullable=False)

    # AI outputs
    concentration_million_ml = db.Column(db.Float, nullable=False, default=0.0)
    motility_percent = db.Column(db.Integer, nullable=False, default=0)
    morphology_percent = db.Column(db.Integer, nullable=False, default=0)
    dfi_percent = db.Column(db.Integer, nullable=False, default=0)

    # AI morphology class output (from SMIDS model)
    morphology_class = db.Column(db.String(50), nullable=True)
    morphology_confidence = db.Column(db.Float, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient_id,

            "patient_name": self.patient_name,
            "visit_date": self.visit_date.isoformat(),
            "age": self.age,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "occupation": self.occupation,
            "exercise_frequency": self.exercise_frequency,

            "concentration_million_ml": self.concentration_million_ml,
            "motility_percent": self.motility_percent,
            "morphology_percent": self.morphology_percent,
            "dfi_percent": self.dfi_percent,

            "morphology_class": self.morphology_class,
            "morphology_confidence": self.morphology_confidence,

            "created_at": self.created_at.isoformat(),
        }