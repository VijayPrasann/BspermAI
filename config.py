class Config:
    # Secret key
    SECRET_KEY = "super-secret-key-change-this"

    # MySQL Database Configuration
    DB_HOST = "localhost"
    DB_USER = "root"
    DB_PASSWORD = "root"
    DB_NAME = "project"

    # SQLAlchemy Connection String
    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:root@localhost/project"

    

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload Settings
    UPLOAD_FOLDER = "uploads"
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB

