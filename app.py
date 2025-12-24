import joblib 
import json
import pandas as pd


def load_model():
    model = joblib.load('credit_score_model.pkl')
    return model


def predict_credit_score(features_dict, feature_columns
                        ,model, default_value=0):
    row = {col: features_dict.get(col, default_value) for col in feature_columns}
    X = pd.DataFrame([row], columns=feature_columns)
    return float(model.predict(X)[0])


def load_feature_columns():
    with open('feature_columns.json', 'r') as f:
        columns = json.load(f)
    return columns

if __name__ == "__main__":
    model = load_model()
    feature_columns = load_feature_columns()
    features = {
        "INCOME": 33269,
        "SAVINGS": 0,
        "DEBT": 532304,
        "R_SAVINGS_INCOME": 0.0,
        "R_DEBT_INCOME": 16.0,
        "R_DEBT_SAVINGS": 1.2,
        "T_CLOTHING_12": 1889,
        "T_CLOTHING_6": 945,
        "R_CLOTHING": 0.5003,
        "R_CLOTHING_INCOME": 0.0568,
        "R_CLOTHING_SAVINGS": 0.0,
        "R_CLOTHING_DEBT": 0.0035,
        "T_EDUCATION_12": 0,
        "T_EDUCATION_6": 0,
        "R_EDUCATION": 0.5044,
        "R_EDUCATION_INCOME": 0.0,
        "R_EDUCATION_SAVINGS": 0.1404,
        "R_EDUCATION_DEBT": 0.0,
        "T_ENTERTAINMENT_12": 3068,
        "T_ENTERTAINMENT_6": 1554,
        "R_ENTERTAINMENT": 0.5065,
        "R_ENTERTAINMENT_INCOME": 0.0922,
        "R_ENTERTAINMENT_SAVINGS": 0.0,
        "R_ENTERTAINMENT_DEBT": 0.0058,
        "T_FINES_12": 0,
        "T_FINES_6": 0,
        "R_FINES": 1.3634,
        "R_FINES_INCOME": 0.0,
        "R_FINES_SAVINGS": 0.0026,
        "R_FINES_DEBT": 0.0,
        "T_GAMBLING_12": 1313,
        "T_GAMBLING_6": 605,
        "R_GAMBLING": 0.4608,
        "R_GAMBLING_INCOME": 0.0395,
        "R_GAMBLING_SAVINGS": 0.0,
        "R_GAMBLING_DEBT": 0.0025,
        "T_GROCERIES_12": 4849,
        "T_GROCERIES_6": 2753,
        "R_GROCERIES": 0.5677,
        "R_GROCERIES_INCOME": 0.1458,
        "R_GROCERIES_SAVINGS": 0.0,
        "R_GROCERIES_DEBT": 0.0091,
        "T_HEALTH_12": 320,
        "T_HEALTH_6": 0,
        "R_HEALTH": 0.3314,
        "R_HEALTH_INCOME": 0.0096,
        "R_HEALTH_SAVINGS": 0.0055,
        "R_HEALTH_DEBT": 0.0006,
        "T_HOUSING_12": 3006,
        "T_HOUSING_6": 1521,
        "R_HOUSING": 0.506,
        "R_HOUSING_INCOME": 0.0904,
        "R_HOUSING_SAVINGS": 0.0,
        "R_HOUSING_DEBT": 0.0056,
        "T_TAX_12": 0,
        "T_TAX_6": 0,
        "R_TAX": 0.5322,
        "R_TAX_INCOME": 0.0,
        "R_TAX_SAVINGS": 0.0276,
        "R_TAX_DEBT": 0.0,
        "T_TRAVEL_12": 17893,
        "T_TRAVEL_6": 11439,
        "R_TRAVEL": 0.6393,
        "R_TRAVEL_INCOME": 0.5378,
        "R_TRAVEL_SAVINGS": 0.0,
        "R_TRAVEL_DEBT": 0.0336,
        "T_UTILITIES_12": 931,
        "T_UTILITIES_6": 469,
        "R_UTILITIES": 0.5038,
        "R_UTILITIES_INCOME": 0.028,
        "R_UTILITIES_SAVINGS": 0.0655,
        "R_UTILITIES_DEBT": 0.0017,
        "T_EXPENDITURE_12": 33269,
        "T_EXPENDITURE_6": 19286,
        "R_EXPENDITURE": 0.5797,
        "R_EXPENDITURE_INCOME": 1.0,
        "R_EXPENDITURE_SAVINGS": 0.0,
        "R_EXPENDITURE_DEBT": 0.0625,
        "CAT_GAMBLING": 2,
        "CAT_DEBT": 1,
        "CAT_CREDIT_CARD": 0,
        "CAT_MORTGAGE": 0,
        "CAT_SAVINGS_ACCOUNT": 0,
        "CAT_DEPENDENTS": 0
    }
    predicted_score = predict_credit_score(features, feature_columns, model)
    print(f"Predicted Credit Score: {predicted_score}")