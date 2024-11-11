"""
General settings and configuration for the ESG analyzer
"""

# Model Settings
MODEL_SETTINGS = {
    'bert_model': 'bert-base-uncased',
    'max_length': 512,
    'batch_size': 32
}

# XGBoost Parameters
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'gamma': 0,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': 42
}

# Random Forest Parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1
}

# Visualization Settings
VIZ_SETTINGS = {
    'figure_size': (10, 6),
    'style': 'default',
    'grid': True,
    'auto_layout': True,
    'color_scheme': {
        'Environmental': '#2ecc71',
        'Social': '#3498db',
        'Governance': '#9b59b6'
    }
}

# File Settings
FILE_SETTINGS = {
    'max_file_size': 200,  # MB
    'allowed_extensions': ['pdf'],
    'encoding': 'utf-8'
}

# Logging Settings
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        }
    }
}