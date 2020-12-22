TRAINING_FILENAME = '../data/training_data.csv'
TESTING_FILENAME = '../data/testing_data.csv'
SCORING_FUNC = 'average_precision'
HYPER_PARAM_N_ITER = 10
TARGET_COLUMN = 'valid_precip'

PREDICTOR_COLUMNS = ['zh', 'max_zh', 'min_zh', 'mean_zh', 'std_zh',
                     'zdr', 'max_zdr', 'min_zdr', 'mean_zdr', 'std_zdr',
                     'rhv', 'max_rhv', 'min_rhv', 'mean_rhv', 'std_rhv',
                     'phidp', 'max_phidp', 'min_phidp', 'mean_phidp', 'std_phidp',
                     'rh_2m_at_radar', 'zh_range', 'zdr_range', 'rhv_range', 'phidp_range']

RF_FILENAME = '../data/RF_predictions.csv'
LR_FILENAME = '../data/LR_predictions.csv'

RF_MODEL_PATH = '../models/RandomForestModel.pkl'
RF_ISO_MODEL_PATH = '../models/RandomForest_IsotonicModel.pkl'

LR_MODEL_PATH = '../models/LogisticRegressionModel.pkl'
LR_ISO_MODEL_PATH = '../models/LogisticRegression_IsotonicModel.pkl'
