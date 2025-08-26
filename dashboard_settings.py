import json
import os
import subprocess

from flask import Flask, flash, redirect, render_template_string, request, url_for

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(PROJECT_PATH, 'config.json')
TRADER_SCRIPT = os.path.join(PROJECT_PATH, 'live_trader_clean.py')

app = Flask(__name__)
app.secret_key = 'mrben_secret'
trading_thread = None
trading_process = None

# --- HTML Template ---
TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MR BEN Advanced Settings Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { background: #f8f9fa; }
        .container { max-width: 800px; margin-top: 40px; }
        .status-dot { height: 12px; width: 12px; border-radius: 50%; display: inline-block; }
        .dot-on { background: #28a745; }
        .dot-off { background: #dc3545; }
    </style>
</head>
<body>
<div class="container">
    <h2 class="mb-4">MR BEN Advanced Settings Dashboard</h2>
    <form method="post" action="/save">
        <h4>Trading</h4>
        <div class="row">
            {% for key, value in config['trading'].items() %}
            <div class="col-md-6 mb-2">
                <label class="form-label">{{ key }}</label>
                <input type="text" class="form-control" name="trading.{{key}}" value="{{ value }}">
            </div>
            {% endfor %}
        </div>
        <h4>Models</h4>
        <div class="row">
            {% for key, value in config['models'].items() %}
            <div class="col-md-4 mb-2">
                <label class="form-label">{{ key }}</label>
                {% if value is boolean %}
                <select class="form-select" name="models.{{key}}">
                    <option value="true" {% if value %}selected{% endif %}>True</option>
                    <option value="false" {% if not value %}selected{% endif %}>False</option>
                </select>
                {% else %}
                <input type="text" class="form-control" name="models.{{key}}" value="{{ value }}">
                {% endif %}
            </div>
            {% endfor %}
        </div>
        <h4>MT5</h4>
        <div class="row">
            {% for key, value in config['mt5'].items() %}
            <div class="col-md-6 mb-2">
                <label class="form-label">{{ key }}</label>
                <input type="text" class="form-control" name="mt5.{{key}}" value="{{ value }}">
            </div>
            {% endfor %}
        </div>
        <h4>Logging</h4>
        <div class="row">
            {% for key, value in config['logging'].items() %}
            <div class="col-md-6 mb-2">
                <label class="form-label">{{ key }}</label>
                <input type="text" class="form-control" name="logging.{{key}}" value="{{ value }}">
            </div>
            {% endfor %}
        </div>
        <h4>Notifications</h4>
        <div class="row">
            {% for key, value in config['notifications'].items() %}
            <div class="col-md-6 mb-2">
                <label class="form-label">{{ key }}</label>
                {% if value is boolean %}
                <select class="form-select" name="notifications.{{key}}">
                    <option value="true" {% if value %}selected{% endif %}>True</option>
                    <option value="false" {% if not value %}selected{% endif %}>False</option>
                </select>
                {% else %}
                <input type="text" class="form-control" name="notifications.{{key}}" value="{{ value }}">
                {% endif %}
            </div>
            {% endfor %}
        </div>
        <button type="submit" class="btn btn-primary mt-3">Save</button>
    </form>
    <hr>
    <h4>Status</h4>
    <ul>
        <li>RL Model: <span class="status-dot {% if status['rl'] %}dot-on{% else %}dot-off{% endif %}"></span> {{ 'Active' if status['rl'] else 'Inactive' }}</li>
        <li>LSTM Model: <span class="status-dot {% if status['lstm'] %}dot-on{% else %}dot-off{% endif %}"></span> {{ 'Active' if status['lstm'] else 'Inactive' }}</li>
        <li>Technical: <span class="status-dot {% if status['technical'] %}dot-on{% else %}dot-off{% endif %}"></span> {{ 'Active' if status['technical'] else 'Inactive' }}</li>
        <li>MT5 Connection: <span class="status-dot {% if status['mt5'] %}dot-on{% else %}dot-off{% endif %}"></span> {{ 'Connected' if status['mt5'] else 'Disconnected' }}</li>
        <li>Trading: <span class="status-dot {% if status['trading'] %}dot-on{% else %}dot-off{% endif %}"></span> {{ 'Running' if status['trading'] else 'Stopped' }}</li>
    </ul>
    <form method="post" action="/start">
        <button type="submit" class="btn btn-success" {% if status['trading'] %}disabled{% endif %}>Start Trading</button>
    </form>
    <form method="post" action="/stop" class="mt-2">
        <button type="submit" class="btn btn-danger" {% if not status['trading'] %}disabled{% endif %}>Stop Trading</button>
    </form>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="alert alert-info mt-3">
          {% for message in messages %}{{ message }}<br>{% endfor %}
        </div>
      {% endif %}
    {% endwith %}
</div>
</body>
</html>
'''


# --- Helper Functions ---
def load_config():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def check_model(path):
    return os.path.exists(path)


def check_mt5_connection():
    # Dummy check for now (could be improved)
    return True


def is_trading_running():
    global trading_process
    return trading_process is not None and trading_process.poll() is None


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def dashboard():
    config = load_config()
    status = {
        'rl': check_model(os.path.join(PROJECT_PATH, 'models', 'mrben_rl_model.pth')),
        'lstm': check_model(os.path.join(PROJECT_PATH, 'models', 'mrben_lstm_model.h5')),
        'technical': config['models'].get('use_technical', True),
        'mt5': check_mt5_connection(),
        'trading': is_trading_running(),
    }
    return render_template_string(TEMPLATE, config=config, status=status)


@app.route('/save', methods=['POST'])
def save():
    config = load_config()
    for key in request.form:
        section, param = key.split('.', 1)
        value = request.form[key]
        # Convert booleans
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except:
                pass
        config[section][param] = value
    save_config(config)
    flash('Settings saved successfully!')
    return redirect(url_for('dashboard'))


@app.route('/start', methods=['POST'])
def start_trading():
    global trading_process
    if not is_trading_running():
        trading_process = subprocess.Popen(['python', TRADER_SCRIPT], cwd=PROJECT_PATH)
        flash('Trading started!')
    else:
        flash('Trading is already running.')
    return redirect(url_for('dashboard'))


@app.route('/stop', methods=['POST'])
def stop_trading():
    global trading_process
    if is_trading_running():
        trading_process.terminate()
        trading_process = None
        flash('Trading stopped!')
    else:
        flash('Trading is not running.')
    return redirect(url_for('dashboard'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
