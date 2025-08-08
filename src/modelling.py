import xgboost as xgb

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

def fit_pipeline(data, categorical_features, numerical_features, target, params):
    X = data.select(categorical_features+numerical_features)
    y = data.select(target)
    
    pipeline = build_pipeline(categorical_features)
    
    pipeline.set_params(**params)
    
    model = pipeline.fit(X, y)
    
    return model

def fit_pipeline_with_tuning(data, categorical_features, numerical_features, target, param_dist):
    X = data.select(categorical_features+numerical_features)
    y = data.select(target)
    
    pipeline = build_pipeline(categorical_features)
    tuning = hyperparameter_tuning(pipeline, param_dist)
    
    tuning.fit(X, y)
    
    return tuning.best_estimator_, tuning.best_score_, tuning.best_params_

def build_pipeline(categorical_features):
    encoder = OneHotEncoder(
        drop='first', 
        handle_unknown='infrequent_if_exist',
        sparse_output=False,
    )

    athlete_encoder = OneHotEncoder(
        drop='first',
        handle_unknown='infrequent_if_exist',
        sparse_output=False
    )
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', encoder, [c for c in categorical_features if c != 'athlete_id']),
            ('ath', athlete_encoder, ['athlete_id'])
        ],
        remainder='passthrough',  # keeps numerical columns like 'weight'
    )

    preprocessor.set_output(transform="polars")

    clf = xgb.XGBClassifier(
        eval_metric = 'aucpr',
        scale_pos_weight = 23.6, # sum(neg) / sum(pos)
        seed=42
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    return pipeline

def hyperparameter_tuning(pipeline, param_dist):
    tscv = TimeSeriesSplit(n_splits=3)

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=500,
        scoring='average_precision',
        cv=tscv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    return random_search