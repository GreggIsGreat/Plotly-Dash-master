import traceback

try:
    from final_dashboard import server as app
except Exception:
    from flask import Flask
    err = traceback.format_exc()
    app = Flask(__name__)

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def show_error(path):
        return f"<pre>{err}</pre>", 500