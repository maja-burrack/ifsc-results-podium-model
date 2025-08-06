import shap

def _get_clean_feature_names(model):
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    feature_names = [name.replace("remainder__", "").replace("cat__", "").replace("ath__", "") for name in feature_names]
    return feature_names

def compute_shap(model, X):
    xgb_model = model.named_steps['classifier']
    X_processed = model.named_steps['preprocessor'].transform(X)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_processed.to_pandas())
    
    shap_values.feature_names = _get_clean_feature_names(model)
    
    return explainer, shap_values