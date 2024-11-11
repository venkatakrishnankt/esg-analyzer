"""
Constants used across the ESG analyzer application
"""
from .enums import ScoreType

# ESG Categories and Factors
ESG_FACTORS = {
    'Environmental': [
        'climate change',
        'carbon emissions',
        'greenhouse gas',
        'renewable energy',
        'energy efficiency',
        'waste management',
        'water usage',
        'biodiversity',
        'pollution',
        'deforestation',
        'recycling',
        'sustainable development'
    ],
    'Social': [
        'diversity and inclusion',
        'human rights',
        'labor practices',
        'community relations',
        'data privacy',
        'employee volunteering participation rate',
        'employee volunteering days',
        'health and safety',
        'modern slavery',
        'ethical marketing'
    ],
    'Governance': [
        'gender diversity',
        'shareholder rights',
        'business ethics',
        'anti-corruption',
        'transparency',
        'lobbying activities',
        'political contributions',
        'whistleblower protection',
        'tax strategy'
    ]
}

# Stop Words
CUSTOM_STOP_WORDS = {
    'group', 'reporting', 'annual report', 'disclosure', 'report', 
    'reports', 'meeting', 'summary', 'company', 'companies', 
    'industries', 'industry', 'organization', 'organizations', 
    'procedures', 'topics', 'year', 'statement', 'task', 
    'category', 'metrics', 'pages', 'page', 'business', 
    'corporation', 'corp', 'inc', 'ltd', 'quarter', 
    'annual', 'quarterly', 'fiscal', 'financial', 
    'please', 'may', 'might', 'could', 'would', 'also'
}


SCORING_THRESHOLDS = {
    'minimum_score': 5,
    'score_weights': {
        ScoreType.EXPLICIT.value: 0.5,
        ScoreType.SEMANTIC.value: 0.2,
        ScoreType.METRIC.value: 0.3
    }
}

# Window Sizes
WINDOW_SIZES = {
    'context': 150,
    'extended_context': 300,
    'metric_context': 50
}