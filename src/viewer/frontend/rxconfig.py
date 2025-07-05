import reflex as rx

config = rx.Config(
    app_name="traces_viewer",
    frontend_port=3200,
    backend_port=8200,  # Run Reflex backend on 8200
    api_url="http://localhost:8200",  # Our FastAPI backend
)
