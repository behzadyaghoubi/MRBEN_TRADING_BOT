import json
import os

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    signals_df = (
        pd.read_csv("final_signal.csv") if os.path.exists("final_signal.csv") else pd.DataFrame()
    )
    final_signal = None
    if os.path.exists("latest_signal.json"):
        with open("latest_signal.json") as f:
            final_signal = json.load(f)
    return render_template(
        "index.html",
        signals=signals_df.tail(50).to_dict(orient='records'),
        final_signal=final_signal,
    )


@app.route('/api/latest')
def api_latest():
    if os.path.exists("latest_signal.json"):
        with open("latest_signal.json") as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"status": "no signal"})


@app.route('/execute', methods=['POST'])
def execute():
    signal_type = request.form.get("signal_type")
    if signal_type:
        os.system("python live_trader.py")
    return redirect("/")


@app.route('/report')
def report():
    if os.path.exists("weekly_report.csv"):
        report_df = pd.read_csv("weekly_report.csv")
    else:
        report_df = pd.DataFrame()
    return render_template("report.html", report=report_df.to_dict(orient='records'))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
