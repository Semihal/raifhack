import argparse
import datetime as dt
import logging.config
import pandas as pd
from traceback import format_exc

from raif_hack.model import BenchmarkModel
from raif_hack.settings import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES,\
    CATEGORICAL_STE_FEATURES, TARGET, ADDITIONAL_CATEGORICAL_FEATURES
from raif_hack.utils import PriceTypeEnum
from raif_hack.metrics import metrics_stat
from raif_hack.features import prepare_categorical

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для обучения модели
     
     Примеры:
        1) с poetry - poetry run python3 train.py --train_data /path/to/train/data --model_path /path/to/model
        2) без poetry - python3 train.py --train_data /path/to/train/data --model_path /path/to/model
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--train_data", "-d", type=str, dest="d", required=True, help="Путь до обучающего датасета")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True, help="Куда сохранить обученную ML модель")

    return parser.parse_args()


if __name__ == "__main__":
    FEATURES = NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES + ADDITIONAL_CATEGORICAL_FEATURES
    VAL_START_DATE = dt.date(2020, 7, 15)
    try:
        logger.info('START train.py')
        args = vars(parse_args())
        logger.info('Load train df')
        train_df = pd.read_csv(args['d'], parse_dates=['date'])
        logger.info(f'Input shape: {train_df.shape}')
        train_df = prepare_categorical(train_df)

        train_mask = train_df['date'].dt.date < VAL_START_DATE
        x_train, y_train = train_df.loc[train_mask, FEATURES], train_df.loc[train_mask, TARGET]
        x_val, y_val = train_df.loc[~train_mask, FEATURES], train_df.loc[~train_mask, TARGET]

        logger.info(f'X_offer {x_train.shape}  y_offer {y_train.shape}\tX_manual {x_val.shape} y_manual {y_val.shape}')
        model = BenchmarkModel(
            numerical_features=NUM_FEATURES,
            ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
            ste_categorical_features=CATEGORICAL_STE_FEATURES,
            additional_categorical_features=ADDITIONAL_CATEGORICAL_FEATURES,
            model_params=MODEL_PARAMS)

        logger.info('Fit model')
        model.fit(x_train, y_train, x_val, y_val)
        logger.info('Save model')
        model.save(args['mp'])

        x_manual_for_metrics = x_val[x_val.price_type == PriceTypeEnum.MANUAL_PRICE]
        y_manual_for_metrics = y_val[x_val.price_type == PriceTypeEnum.MANUAL_PRICE]
        predictions_manual = model.predict(x_manual_for_metrics)
        metrics = metrics_stat(y_manual_for_metrics.values, predictions_manual)
        logger.info(f'Metrics stat for training data with manual prices: {metrics}')
    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise(e)
    logger.info('END train.py')