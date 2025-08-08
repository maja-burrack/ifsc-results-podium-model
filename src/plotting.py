import matplotlib.pyplot as plt
import numpy as np

import shap
from .explainability import compute_shap, _get_clean_feature_names

def plot_feature_importances(model, top_n=20, save_fig=False):
    importances = model.named_steps['classifier'].feature_importances_

    feature_names = _get_clean_feature_names(model)

    # Sort feature importances descending
    indices = np.argsort(importances)[::-1]

    # Limit to top n features
    top_n = min(top_n, len(indices))
    top_indices = indices[:top_n]
    top_features = np.array(feature_names)[top_indices]
    top_importances = importances[top_indices]

    plt.figure(figsize=(9, 0.3*top_n))
    plt.title(f"Top {top_n} Feature Importances")

    # Reverse order so highest importance is at top
    plt.barh(range(top_n), top_importances[::-1], align='center')
    plt.yticks(range(top_n), top_features[::-1])
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('plots/feature_importances.png')
        
    plt.show()

def plot_shap(model, X, top_n=20, shap_values=None, save_path=None):
    if not shap_values:
        _, shap_values = compute_shap(model, X)
    
    X_processed = model.named_steps['preprocessor'].transform(X)
    feature_names = _get_clean_feature_names(model)
    shap_values.feature_names = _get_clean_feature_names(model)
    
    fig_size = (12, 0.4*top_n)
    shap.summary_plot(
        shap_values, 
        # X_processed.to_pandas(), 
        feature_names=feature_names,
        max_display=top_n,
        plot_size=fig_size,
        show=False
    )
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_shap_waterfall(shap_values=None, index=0, top_n=10, model=None, X=None, save_path=None):
    if not shap_values:
        _, shap_values = compute_shap(model, X)
    
    shap_values.feature_names = _get_clean_feature_names(model)
    
    shap.plots.waterfall(shap_values[index], max_display=top_n, show=False)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.show()
        
def plot_shap_bar(shap_values, save_path=False):
    shap.plots.bar(shap_values, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show() 