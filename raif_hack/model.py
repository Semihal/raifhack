import typing
import pickle
import pandas as pd
import numpy as np
import logging

from lightgbm import LGBMRegressor

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)


class BenchmarkModel():
    """
    Модель представляет из себя sklearn pipeline. Пошаговый алгоритм:
      1) в качестве обучения выбираются все данные с price_type=0
      1) все фичи делятся на три типа (numerical_features, ohe_categorical_features, ste_categorical_features):
          1.1) numerical_features - применяется StandardScaler
          1.2) ohe_categorical_featires - кодируются через one hot encoding
          1.3) ste_categorical_features - кодируются через SmoothedTargetEncoder
      2) после этого все полученные фичи конкатенируются в одно пространство фичей и подаются на вход модели Lightgbm
      3) делаем предикт на данных с price_type=1, считаем среднее отклонение реальных значений от предикта. Вычитаем это отклонение на финальном шаге (чтобы сместить отклонение к 0)

    :param numerical_features: list, список численных признаков из датафрейма
    :param ohe_categorical_features: list, список категориальных признаков для one hot encoding
    :param ste_categorical_features, list, список категориальных признаков для smoothed target encoding.
                                     Можно кодировать сразу несколько полей (например объединять категориальные признаки)
    :
    """

    def __init__(self, numerical_features: typing.List[str],
                 ohe_categorical_features: typing.List[str],
                 ste_categorical_features: typing.List[typing.Union[str, typing.List[str]]],
                 additional_categorical_features: typing.List[str],
                 model_params: typing.Dict[str, typing.Union[str, int, float]]):
        self.num_features = numerical_features
        self.ohe_cat_features = ohe_categorical_features
        self.ste_cat_features = ste_categorical_features
        self.additional_categorical_features = additional_categorical_features

        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder()
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        self.model = LGBMRegressor(**model_params)

        self._is_fitted = False
        self.corr_coef = pd.DataFrame([], columns=['city', 'deviation'])

    def _fit_preprocessing(self, X, y):
        self.scaler.fit(X[self.num_features])
        self.one_hot_encoder.fit(X[self.ohe_cat_features])
        self.ordinal_encoder.fit(X[self.ste_cat_features], y)

    def osm_mask_with_infrastructure(self, df, prefix, values, dist_list):
        query = ' or '.join([
            f'`{prefix}_{dist}` >= {min_building}'
            for dist, min_building in zip(dist_list, values)])
        true_indexes = df.query(query).index
        mask = df.index.isin(true_indexes).astype('int')
        return mask

    def _transform_preprocessing(self, X, y=None):
        X = X.copy()
        X['total_square'] = np.log1p(X['total_square'])
        X['osm_city_closest_dist'] = np.log1p(X['osm_city_closest_dist'])

        # feature engineering
        X['cat_is_moscow'] = (X['region'] == 'Москва').astype(int)
        X['cat_is_moscow_oblast'] = (X['region'] == 'Московская область').astype(int)
        X['cat_is_other_region'] = (~X['region'].isin(['Москва', 'Московская область'])).astype(int)
        # self.ste_cat_features.remove('region')

        # osm features
        dist_list = ['0.001', '0.005', '0.0075', '0.01']
        values = [2, 5, 7, 10]
        # engineering
        osm_prefix = 'osm_shops_points_in'
        X['cat_many_shops'] = self.osm_mask_with_infrastructure(X, osm_prefix, values, dist_list)
        osm_prefix = 'osm_leisure_points_in'
        X['cat_many_entertainment'] = self.osm_mask_with_infrastructure(X, osm_prefix, [1, 2, 3], dist_list[1:])
        osm_prefix = 'osm_historic_points_in'
        X['cat_many_history'] = self.osm_mask_with_infrastructure(X, osm_prefix, [1, 2, 3], dist_list[1:])
        osm_prefix = 'osm_transport_stop_points_in'
        X['cat_many_transports'] = self.osm_mask_with_infrastructure(X, osm_prefix, [1, 2, 3], dist_list[1:])
        osm_prefix = 'osm_culture_points_in'
        X['cat_many_culture_points'] = self.osm_mask_with_infrastructure(X, osm_prefix, [1, 2, 2], dist_list[1:])
        osm_prefix = 'osm_amenity_points_in'
        X['cat_many_amenity_points'] = self.osm_mask_with_infrastructure(X, osm_prefix, [1, 1, 1], dist_list[1:])

        new_columns = [
            'cat_is_moscow',
            'cat_is_moscow_oblast',
            'cat_is_other_region',
            'cat_many_shops',
            'cat_many_entertainment',
            'cat_many_history',
            'cat_many_transports',
            'cat_many_culture_points',
            'cat_many_amenity_points',
            'price_type'
        ]
        new_features = X[new_columns].rename(columns={'price_type': 'cat_price_type'}).reset_index(drop=True)
        # применение стндартизациии
        scaler_features = self.scaler.transform(X[self.num_features])
        scaler_features = pd.DataFrame(scaler_features, columns=[f'num_{col}' for col in self.num_features])
        # применение ordinal_encoder
        ste_columns = [f'cat_{col}' for col in self.ste_cat_features]
        ste_encoding = self.ordinal_encoder.transform(X[self.ste_cat_features])
        ste_encoding = pd.DataFrame(ste_encoding, columns=ste_columns)
        # применение one_hot_encoder
        ohe_columns = [
            f'cat_{prefix}_{value}'
            for prefix, values in zip(self.ohe_cat_features, self.one_hot_encoder.categories_)
            for value in values
        ]
        ohe_encoding = self.one_hot_encoder.transform(X[self.ohe_cat_features])
        ohe_encoding = pd.DataFrame.sparse.from_spmatrix(ohe_encoding, columns=ohe_columns)
        # result of transformers
        transformed = pd.concat([new_features, scaler_features, ste_encoding, ohe_encoding], axis=1)
        return transformed, y

    def _find_corr_coefficient(self, X_manual: pd.DataFrame, y_manual: pd.Series):
        """Вычисление корректирующего коэффициента

        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        features = X_manual.copy()
        features = features[features['price_type'] == 1]
        features['predict'] = self.predict(features)
        features['deviation'] = (y_manual - features['predict']) / features['predict']
        self.corr_coef = features[['city', 'deviation']].groupby('city')['deviation'].median().reset_index()

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series,
            x_val: pd.DataFrame, y_val: pd.Series):
        """Обучение модели.
        ML модель обучается на данных по предложениям на рынке (цены из объявления)
        Затем вычисляется среднее отклонение между руяными оценками и предиктами для корректировки стоимости

        :param X_offer: pd.DataFrame с объявлениями
        :param y_offer: pd.Series - цена предложения (в объявлениях)
        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        logger.info('Fit lightgbm')
        self._fit_preprocessing(x_train, y_train)
        x_train, y_train = self._transform_preprocessing(x_train, y_train)
        x_val_transformed, y_val_transformed = self._transform_preprocessing(x_val, y_val)
        self.model.fit(
            x_train, y_train,
            eval_set=[
                (x_train, y_train),
                (x_val_transformed, y_val_transformed)
            ],
            eval_names=['train', 'val'],
            early_stopping_rounds=100,
            feature_name=x_train.columns.tolist(),
            categorical_feature=[col for col in x_train.columns if col.startswith('cat_')]
        )
        logger.info('Find corr coefficient')
        self.__is_fitted = True
        self._find_corr_coefficient(x_val, y_val)

    def predict(self, X: pd.DataFrame) -> np.array:
        """Предсказание модели Предсказываем преобразованный таргет, затем конвертируем в обычную цену через обратное
        преобразование.

        :param X: pd.DataFrame
        :return: np.array, предсказания (цены на коммерческую недвижимость)
        """
        if self.__is_fitted:
            features, _ = self._transform_preprocessing(X)
            X.loc[:, 'predictions'] = self.model.predict(features)
            pred_with_corr_coef = X[['city', 'predictions']].merge(self.corr_coef, on=['city'], how='left')
            median_deviation = pred_with_corr_coef['deviation'].median()
            median_deviation = 0 if np.isnan(median_deviation) else median_deviation
            pred_with_corr_coef['deviation'] = pred_with_corr_coef['deviation'].fillna(median_deviation)
            corrected_price = pred_with_corr_coef['predictions'] * (1 + pred_with_corr_coef['deviation'])
            return corrected_price.values
        else:
            raise NotFittedError(
                "This {} instance is not fitted yet! Call 'fit' with appropriate arguments before predict".format(
                    type(self).__name__
                )
            )

    def save(self, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        :return: Модель
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
