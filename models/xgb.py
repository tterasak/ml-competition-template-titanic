import xgboost as xgb
import logging

from logs.logger import log_evaluation


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, xgb_params):

    # データセットを生成する
    xgb_train = xgb.Dataset(X_train, y_train)
    xgb_eval = xgb.Dataset(X_valid, y_valid, reference=xgb_train)

    logging.debug(xgb_params)

    # ロガーの作成
    logger = logging.getLogger('main')
    callbacks = [log_evaluation(logger, period=30)]

    # 上記のパラメータでモデルを学習する
    model = xgb.train(
        xgb_params, xgb_train,
        # モデルの評価用データを渡す
        valid_sets=xgb_eval,
        # 最大で 1000 ラウンドまで学習する
        num_boost_round=1000,
        # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
        early_stopping_rounds=10,
        # ログ
        callbacks=callbacks
    )

    # テストデータを予測する
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    return y_pred, model