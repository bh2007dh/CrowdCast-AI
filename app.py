"""
CrowdCast AI — Location-Aware Intelligent Crowd & Traffic Prediction System.
Main Flask Application Server.
"""
import os
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template

from config import Config
from database.db import get_db_connection, init_db
from models.congestion_engine import congestion_engine
from models.crowd_predictor import crowd_predictor
from routes.admin_routes import admin_bp
from routes.event_routes import event_bp
from routes.location_routes import location_bp
from routes.prediction_routes import prediction_bp
from routes.report_routes import report_bp
from routes.traffic_routes import traffic_bp
from services.traffic_service import traffic_service


def create_app(test_config: dict = None) -> Flask:
    """Application factory for CrowdCast AI."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    # Initialize SQLite database and baseline seed records
    db_path = app.config.get("DATABASE_PATH", Config.DATABASE_PATH)
    init_db(db_path)

    # Register REST API Blueprints
    app.register_blueprint(location_bp)
    app.register_blueprint(traffic_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(event_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(admin_bp)

    # Context processor so every template knows provider status & data labels
    @app.context_processor
    def inject_global_status():
        return {
            "provider_status": traffic_service.get_provider_status(),
            "weather_configured": bool(Config.OPENWEATHER_API_KEY),
            "fusion_weights": congestion_engine.weights,
        }

    # Health check endpoint
    @app.route("/api/health", methods=["GET"])
    def api_health():
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r["name"] for r in cursor.fetchall()]
        conn.close()
        return (
            jsonify(
                {
                    "status": "ok",
                    "application": "CrowdCast AI",
                    "version": "1.0.0",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "database_connected": True,
                    "database_tables": tables,
                    "ml_model_loaded": crowd_predictor.bundle is not None,
                    "ml_dataset_type": "SYNTHETIC_DEMONSTRATION_DATASET",
                    "traffic_provider": traffic_service.get_provider_status(),
                    "weather_provider_configured": bool(Config.OPENWEATHER_API_KEY),
                    "fusion_weights": congestion_engine.weights,
                }
            ),
            200,
        )

    # Website Pages (Section 9 & 10)
    @app.route("/")
    def home_page():
        return render_template("index.html", active_page="home")

    @app.route("/dashboard")
    def dashboard_page():
        return render_template("dashboard.html", active_page="dashboard")

    @app.route("/traffic")
    def traffic_page():
        return render_template("traffic.html", active_page="traffic")

    @app.route("/prediction")
    def prediction_page():
        return render_template("prediction.html", active_page="prediction")

    @app.route("/events")
    def events_page():
        return render_template("events.html", active_page="events")

    @app.route("/congestion")
    def congestion_page():
        return render_template("congestion.html", active_page="congestion")

    @app.route("/reports")
    def reports_page():
        return render_template("reports.html", active_page="reports")

    @app.route("/admin")
    def admin_page():
        return render_template("admin.html", active_page="admin")

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", Config.PORT))
    host = os.getenv("HOST", Config.HOST)
    print(f"Starting CrowdCast AI server at http://{host}:{port}")
    app.run(host=host, port=port, debug=Config.DEBUG, use_reloader=False)


