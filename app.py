import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the model once
model = joblib.load("grail_model.pkl")
columns = ["QuizAvg", "Attendance", "Missed", "Overall"]

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Risk Detector</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .risk-label {
            font-size: 1.5rem;
            font-weight: bold;
        }
    </style>
</head>
<body class="bg-light">
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-md-6">
                <div class="card shadow">
                    <div class="card-body">
                        <h2 class="mb-4 text-center">Student Risk Detector</h2>
                        <form id="riskForm" autocomplete="off">
                            <div class="mb-3">
                                <label for="quizavg" class="form-label">Quiz Average</label>
                                <input type="number" class="form-control" id="quizavg" name="QuizAvg" min="0" max="100" required>
                            </div>
                            <div class="mb-3">
                                <label for="attendance" class="form-label">Attendance</label>
                                <input type="number" class="form-control" id="attendance" name="Attendance" min="0" max="100" required>
                            </div>
                            <div class="mb-3">
                                <label for="missed" class="form-label">Missed Classes</label>
                                <input type="number" class="form-control" id="missed" name="Missed" min="0" required>
                            </div>
                            <div class="mb-3">
                                <label for="overall" class="form-label">Overall</label>
                                <input type="number" class="form-control" id="overall" name="Overall" min="0" max="100" required>
                            </div>
                        </form>
                        <div class="mt-4 text-center">
                            <span id="riskLabel" class="risk-label text-secondary">Enter data to see risk...</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        const form = document.getElementById('riskForm');
        const riskLabel = document.getElementById('riskLabel');
        const inputs = form.querySelectorAll('input');

        function getFormData() {
            return {
                QuizAvg: form.quizavg.value,
                Attendance: form.attendance.value,
                Missed: form.missed.value,
                Overall: form.overall.value
            };
        }

        function allFieldsFilled(data) {
            return Object.values(data).every(v => v !== '' && !isNaN(v));
        }

        function updateRisk() {
            const data = getFormData();
            if (!allFieldsFilled(data)) {
                riskLabel.textContent = 'Enter data to see risk...';
                riskLabel.className = 'risk-label text-secondary';
                return;
            }
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                riskLabel.textContent = result.label;
                riskLabel.className = 'risk-label ' + (result.label === 'At Risk' ? 'text-danger' : 'text-success');
            })
            .catch(() => {
                riskLabel.textContent = 'Error detecting risk';
                riskLabel.className = 'risk-label text-warning';
            });
        }

        inputs.forEach(input => {
            input.addEventListener('input', updateRisk);
        });
    </script>
</body>
</html>
'''

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        values = [[
            float(data.get("QuizAvg", 0)),
            float(data.get("Attendance", 0)),
            float(data.get("Missed", 0)),
            float(data.get("Overall", 0))
        ]]
        df = pd.DataFrame(values, columns=columns)
        pred = model.predict(df)[0]
        label = "At Risk" if pred == 1 else "Not At Risk"
        return jsonify({"label": label})
    except Exception as e:
        return jsonify({"label": "Error"}), 400

if __name__ == "__main__":
    app.run(debug=True) 