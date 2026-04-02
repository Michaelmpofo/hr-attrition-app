from flask import Flask, request, jsonify, render_template_string
import pickle, json, numpy as np, os

app = Flask(__name__)

BASE = os.path.dirname(__file__)
model   = pickle.load(open(os.path.join(BASE,'model.pkl'),'rb'))
le_dept = pickle.load(open(os.path.join(BASE,'le_dept.pkl'),'rb'))
le_sal  = pickle.load(open(os.path.join(BASE,'le_sal.pkl'),'rb'))
meta    = json.load(open(os.path.join(BASE,'meta.json')))

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>HR Attrition Predictor</title>
<style>
  :root{--primary:#4f46e5;--danger:#ef4444;--success:#22c55e;--bg:#f8fafc;--card:#fff;--border:#e2e8f0;--text:#1e293b;--muted:#64748b}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
  header{background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);padding:2rem;text-align:center;color:#fff;box-shadow:0 4px 20px rgba(79,70,229,.3)}
  header h1{font-size:2rem;font-weight:800;letter-spacing:-.5px}
  header p{margin-top:.4rem;opacity:.85;font-size:1rem}
  .badge{display:inline-block;background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.3);border-radius:999px;padding:.25rem .8rem;font-size:.8rem;margin-top:.6rem}
  .container{max-width:800px;margin:2rem auto;padding:0 1.2rem}
  .card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:1.8rem;margin-bottom:1.5rem;box-shadow:0 1px 6px rgba(0,0,0,.06)}
  .card h2{font-size:1.1rem;color:var(--primary);margin-bottom:1.2rem;font-weight:700;display:flex;align-items:center;gap:.5rem}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
  @media(max-width:540px){.grid{grid-template-columns:1fr}}
  label{display:block;font-size:.82rem;font-weight:600;color:var(--muted);margin-bottom:.35rem;text-transform:uppercase;letter-spacing:.5px}
  input,select{width:100%;padding:.6rem .85rem;border:1.5px solid var(--border);border-radius:8px;font-size:.95rem;color:var(--text);background:#fff;transition:border .2s}
  input:focus,select:focus{outline:none;border-color:var(--primary);box-shadow:0 0 0 3px rgba(79,70,229,.12)}
  .range-wrap{position:relative}
  .range-wrap input[type=range]{padding:0;border:none;background:none;cursor:pointer}
  .range-val{position:absolute;right:0;top:0;font-size:.85rem;font-weight:700;color:var(--primary)}
  button{width:100%;padding:.85rem;background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;border:none;border-radius:10px;font-size:1rem;font-weight:700;cursor:pointer;transition:transform .15s,box-shadow .15s;letter-spacing:.3px}
  button:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(79,70,229,.35)}
  button:active{transform:translateY(0)}
  #result{display:none;border-radius:14px;padding:1.8rem;text-align:center;animation:fadeIn .4s ease}
  @keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
  .result-stay{background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:2px solid #86efac}
  .result-leave{background:linear-gradient(135deg,#fff1f2,#ffe4e6);border:2px solid #fca5a5}
  .result-icon{font-size:3.5rem}
  .result-label{font-size:1.6rem;font-weight:800;margin:.5rem 0}
  .result-prob{font-size:1rem;color:var(--muted);margin-bottom:1rem}
  .prob-bar{height:10px;border-radius:999px;background:#e2e8f0;overflow:hidden;margin-top:.4rem}
  .prob-fill{height:100%;border-radius:999px;transition:width .8s ease}
  .tips{text-align:left;margin-top:1rem;padding:1rem;background:rgba(255,255,255,.7);border-radius:10px;font-size:.88rem}
  .tips li{margin:.3rem 0;margin-left:1.2rem}
  footer{text-align:center;padding:2rem;color:var(--muted);font-size:.82rem}
</style>
</head>
<body>
<header>
  <h1>🏢 HR Attrition Predictor</h1>
  <p>Predict the likelihood of an employee leaving the company</p>
  <span class="badge">✨ Random Forest · Accuracy: {{ accuracy }}%</span>
</header>

<div class="container">
  <div class="card">
    <h2>📋 Employee Information</h2>
    <div class="grid">
      <div>
        <label>Department</label>
        <select id="dept">
          {% for d in departments %}<option value="{{ d }}">{{ d.title() }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label>Salary Level</label>
        <select id="salary">
          <option value="low">Low</option>
          <option value="medium" selected>Medium</option>
          <option value="high">High</option>
        </select>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>📊 Performance Metrics</h2>
    <div class="grid">
      <div>
        <label>Satisfaction Level</label>
        <div class="range-wrap">
          <span class="range-val" id="sat-val">0.50</span>
          <input type="range" id="satisfaction" min="0.01" max="1" step="0.01" value="0.5"
            oninput="document.getElementById('sat-val').textContent=parseFloat(this.value).toFixed(2)">
        </div>
      </div>
      <div>
        <label>Last Evaluation Score</label>
        <div class="range-wrap">
          <span class="range-val" id="eval-val">0.70</span>
          <input type="range" id="evaluation" min="0.01" max="1" step="0.01" value="0.7"
            oninput="document.getElementById('eval-val').textContent=parseFloat(this.value).toFixed(2)">
        </div>
      </div>
      <div>
        <label>Number of Projects</label>
        <input type="number" id="projects" min="1" max="10" value="4">
      </div>
      <div>
        <label>Avg Monthly Hours</label>
        <input type="number" id="hours" min="80" max="310" value="200">
      </div>
      <div>
        <label>Years at Company</label>
        <input type="number" id="years" min="1" max="20" value="3">
      </div>
    </div>
  </div>

  <div class="card">
    <h2>🔍 Additional Factors</h2>
    <div class="grid">
      <div>
        <label>Work Accident?</label>
        <select id="accident">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>
      <div>
        <label>Promoted in Last 5 Years?</label>
        <select id="promotion">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>
    </div>
  </div>

  <button onclick="predict()">🔮 Predict Attrition Risk</button>

  <div id="result" style="margin-top:1.5rem"></div>
</div>

<footer>HR Attrition Predictor · Powered by Random Forest · {{ accuracy }}% Accuracy</footer>

<script>
async function predict(){
  const btn = document.querySelector('button');
  btn.textContent = '⏳ Analysing...';
  btn.disabled = true;

  const payload = {
    satisfaction_level: parseFloat(document.getElementById('satisfaction').value),
    last_evaluation: parseFloat(document.getElementById('evaluation').value),
    number_project: parseInt(document.getElementById('projects').value),
    average_montly_hours: parseInt(document.getElementById('hours').value),
    time_spend_company: parseInt(document.getElementById('years').value),
    Work_accident: parseInt(document.getElementById('accident').value),
    promotion_last_5years: parseInt(document.getElementById('promotion').value),
    Department: document.getElementById('dept').value,
    salary: document.getElementById('salary').value
  };

  const res = await fetch('/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data = await res.json();
  btn.textContent = '🔮 Predict Attrition Risk';
  btn.disabled = false;

  const div = document.getElementById('result');
  div.style.display = 'block';
  const willLeave = data.prediction === 1;
  const prob = (data.probability * 100).toFixed(1);
  const stayProb = (100 - prob).toFixed(1);

  let tips = '';
  if(willLeave){
    tips = `<div class="tips"><strong>⚠️ Retention Suggestions:</strong><ul>
      ${parseFloat(document.getElementById('satisfaction').value) < 0.5 ? '<li>Address satisfaction issues — consider 1:1 meetings</li>' : ''}
      ${parseInt(document.getElementById('projects').value) > 5 ? '<li>Reduce project overload</li>' : ''}
      ${parseInt(document.getElementById('hours').value) > 230 ? '<li>Monitor work-life balance — reduce overtime</li>' : ''}
      ${document.getElementById('promotion').value === '0' ? '<li>Review promotion eligibility</li>' : ''}
      ${document.getElementById('salary').value === 'low' ? '<li>Consider salary review or benefits upgrade</li>' : ''}
      <li>Schedule a career development conversation</li>
    </ul></div>`;
  }

  div.className = willLeave ? 'result-leave' : 'result-stay';
  div.innerHTML = `
    <div class="result-icon">${willLeave ? '🚨' : '✅'}</div>
    <div class="result-label" style="color:${willLeave?'#dc2626':'#16a34a'}">${willLeave ? 'High Risk of Leaving' : 'Likely to Stay'}</div>
    <div class="result-prob">Attrition probability: <strong>${prob}%</strong></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem;max-width:340px;margin:0 auto;font-size:.85rem">
      <div>🔴 Leave: ${prob}%
        <div class="prob-bar"><div class="prob-fill" style="width:${prob}%;background:#ef4444"></div></div>
      </div>
      <div>🟢 Stay: ${stayProb}%
        <div class="prob-bar"><div class="prob-fill" style="width:${stayProb}%;background:#22c55e"></div></div>
      </div>
    </div>
    ${tips}
  `;
  div.scrollIntoView({behavior:'smooth',block:'nearest'});
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML,
        departments=meta['departments'],
        accuracy=meta['accuracy'])

@app.route('/predict', methods=['POST'])
def predict():
    d = request.get_json()
    dept_enc = le_dept.transform([d['Department']])[0]
    sal_enc  = le_sal.transform([d['salary']])[0]
    import pandas as pd
    features = pd.DataFrame([[
        d['satisfaction_level'], d['last_evaluation'],
        d['number_project'], d['average_montly_hours'],
        d['time_spend_company'], d['Work_accident'],
        d['promotion_last_5years'], dept_enc, sal_enc
    ]], columns=meta['features'])
    pred  = int(model.predict(features)[0])
    proba = float(model.predict_proba(features)[0][1])
    return jsonify({'prediction': pred, 'probability': round(proba,4)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)), debug=False)
