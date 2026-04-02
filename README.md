# HR Employee Attrition Predictor

A machine learning web application that predicts whether an employee is likely to leave a company.

## Model
- **Algorithm:** Random Forest Classifier  
- **Dataset:** HR_comma_sep.csv (14,999 employees)  
- **Accuracy:** 98.83%  
- **Target:** `left` (0 = Stayed, 1 = Left)

## Features Used
- Satisfaction Level
- Last Evaluation Score
- Number of Projects
- Average Monthly Hours
- Time Spent at Company
- Work Accident (Yes/No)
- Promotion in Last 5 Years (Yes/No)
- Department
- Salary Level

## Local Setup
```bash
pip install -r requirements.txt
python app.py
```
Visit: http://localhost:5000

## Deploy to Render
1. Push all files to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Click Deploy!
