import pandas as pd


def prepare_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage et variables metier simples pour la prediction du churn."""
    data = df.copy()

    data["complaint_type"] = data["complaint_type"].fillna("Aucune Plainte")

    tenure_safe = data["tenure_months"].clip(lower=1)
    logins_safe = data["monthly_logins"].clip(lower=1)

    data["has_complaint"] = (data["complaint_type"] != "Aucune Plainte").astype(int)
    data["payment_risk"] = (
        (data["payment_failures"] > 0) | (data["price_increase_last_3m"] == "Yes")
    ).astype(int)
    data["low_satisfaction"] = (
        (data["nps_score"] < 0)
        | (data["csat_score"] <= 2)
        | (data["survey_response"] == "Unsatisfied")
    ).astype(int)
    data["inactive_customer"] = (
        (data["monthly_logins"] <= 5) | (data["last_login_days_ago"] >= 20)
    ).astype(int)
    data["monthly_contract"] = (data["contract_type"] == "Monthly").astype(int)

    data["tickets_per_tenure"] = data["support_tickets"] / tenure_safe
    data["revenue_per_month"] = data["total_revenue"] / tenure_safe
    data["fee_per_login"] = data["monthly_fee"] / logins_safe
    data["support_pressure"] = data["support_tickets"] * data["avg_resolution_time"]
    data["engagement_score"] = (
        data["monthly_logins"] * 0.30
        + data["weekly_active_days"] * 2.0
        + data["features_used"] * 1.5
        + data["usage_growth_rate"] * 10
        - data["last_login_days_ago"] * 0.40
    )
    data["satisfaction_score"] = (
        data["csat_score"] * 10
        + data["nps_score"] * 0.20
        - data["escalations"] * 5
        - data["has_complaint"] * 8
    )

    return data
