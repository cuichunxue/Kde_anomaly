"""
KDE異常検知システム - デモスクリプト

簡単に試せるサンプルコード集
"""

import numpy as np
import pandas as pd
from kde_anomaly_detector import KDEAnomalyDetector, EasyKDEDetector, quick_detect


def demo_simple():
    """最もシンプルな使い方"""
    print("\n" + "="*70)
    print("デモ1: 最もシンプルな使い方")
    print("="*70 + "\n")

    # サンプルデータ作成
    np.random.seed(42)
    X_train = np.random.randn(1000, 2)  # 正常データ
    X_test = np.vstack([
        np.random.randn(90, 2),          # 正常データ
        np.random.randn(10, 2) * 5 + 8   # 異常データ
    ])

    print("データ準備:")
    print(f"  訓練データ: {X_train.shape[0]}サンプル")
    print(f"  テストデータ: {X_test.shape[0]}サンプル")
    print()

    # 3行で異常検知
    print("コード例:")
    print("  labels, scores = quick_detect(X_train, X_test)")
    print()

    labels, scores = quick_detect(X_train, X_test)

    # 結果表示
    print("結果:")
    print(f"  正常: {np.sum(labels == 1)}個")
    print(f"  異常: {np.sum(labels == -1)}個")
    print(f"  異常度スコア範囲: {scores.min():.2f} ~ {scores.max():.2f}")


def demo_with_dataframe():
    """DataFrameを使った使い方"""
    print("\n" + "="*70)
    print("デモ2: DataFrameを使った使い方")
    print("="*70 + "\n")

    # サンプルデータ作成（センサーデータを模擬）
    np.random.seed(42)
    n_normal = 500
    n_anomaly = 50

    # 正常データ
    temp_normal = np.random.randn(n_normal) * 5 + 25  # 温度
    pressure_normal = np.random.randn(n_normal) * 0.5 + 1.0  # 圧力
    vibration_normal = np.random.randn(n_normal) * 0.1 + 0.5  # 振動

    df_train = pd.DataFrame({
        'temperature': temp_normal,
        'pressure': pressure_normal,
        'vibration': vibration_normal
    })

    # テストデータ（正常 + 異常）
    temp_test = np.concatenate([
        np.random.randn(n_normal) * 5 + 25,
        np.random.randn(n_anomaly) * 10 + 50  # 異常な温度
    ])
    pressure_test = np.concatenate([
        np.random.randn(n_normal) * 0.5 + 1.0,
        np.random.randn(n_anomaly) * 2 + 3.0  # 異常な圧力
    ])
    vibration_test = np.concatenate([
        np.random.randn(n_normal) * 0.1 + 0.5,
        np.random.randn(n_anomaly) * 1 + 2.0  # 異常な振動
    ])

    df_test = pd.DataFrame({
        'temperature': temp_test,
        'pressure': pressure_test,
        'vibration': vibration_test
    })

    print("データ準備:")
    print(f"  訓練データ: {len(df_train)}サンプル x {len(df_train.columns)}特徴量")
    print(f"  テストデータ: {len(df_test)}サンプル")
    print(f"  特徴量: {', '.join(df_train.columns)}")
    print()

    # モデル作成と学習
    print("モデル学習中...")
    detector = EasyKDEDetector(contamination=0.1)
    detector.fit(df_train.values, feature_names=df_train.columns.tolist())
    print()

    # 予測
    print("予測実行...")
    labels = detector.predict(df_test.values)
    scores = detector.detector.anomaly_score(
        detector._scale(df_test.values, fit=False)
    )

    # 結果をDataFrameに追加
    df_result = df_test.copy()
    df_result['anomaly_label'] = labels
    df_result['anomaly_score'] = scores
    df_result['is_anomaly'] = (labels == -1)

    print()
    print("結果:")
    print(f"  総サンプル数: {len(df_result)}")
    print(f"  正常: {np.sum(labels == 1)}個 ({np.sum(labels == 1)/len(labels)*100:.1f}%)")
    print(f"  異常: {np.sum(labels == -1)}個 ({np.sum(labels == -1)/len(labels)*100:.1f}%)")
    print()

    # 上位の異常を表示
    print("最も異常度の高い5サンプル:")
    top_5 = df_result.nlargest(5, 'anomaly_score')
    for idx, row in top_5.iterrows():
        print(f"  #{idx}: temp={row['temperature']:.1f}, "
              f"pressure={row['pressure']:.2f}, "
              f"vibration={row['vibration']:.2f}, "
              f"score={row['anomaly_score']:.2f}")


def demo_csv_workflow():
    """CSVファイルを使った実務的なワークフロー"""
    print("\n" + "="*70)
    print("デモ3: CSVファイルを使った実務的なワークフロー")
    print("="*70 + "\n")

    # サンプルCSVデータ作成
    np.random.seed(42)

    # 訓練データ
    df_train = pd.DataFrame({
        'sensor1': np.random.randn(1000) * 2 + 10,
        'sensor2': np.random.randn(1000) * 1 + 5,
        'sensor3': np.random.randn(1000) * 0.5 + 3,
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H')
    })
    train_file = '/tmp/sensor_train.csv'
    df_train.to_csv(train_file, index=False)
    print(f"訓練データを保存: {train_file}")

    # テストデータ
    normal_data = np.random.randn(180, 3) * np.array([2, 1, 0.5]) + np.array([10, 5, 3])
    anomaly_data = np.random.randn(20, 3) * np.array([5, 3, 2]) + np.array([20, 15, 10])
    test_data = np.vstack([normal_data, anomaly_data])

    df_test = pd.DataFrame({
        'sensor1': test_data[:, 0],
        'sensor2': test_data[:, 1],
        'sensor3': test_data[:, 2],
        'timestamp': pd.date_range('2024-02-01', periods=200, freq='1H')
    })
    test_file = '/tmp/sensor_test.csv'
    df_test.to_csv(test_file, index=False)
    print(f"テストデータを保存: {test_file}")
    print()

    # モデル作成と学習
    print("=" * 70)
    print("ステップ1: モデル学習")
    print("=" * 70)
    detector = EasyKDEDetector(contamination=0.1, auto_scale=True)
    detector.fit_csv(train_file, features=['sensor1', 'sensor2', 'sensor3'])
    print()

    # モデル概要表示
    detector.summary()
    print()

    # 予測と結果保存
    print("=" * 70)
    print("ステップ2: 異常検知実行")
    print("=" * 70)
    result_file = '/tmp/sensor_result.csv'
    df_result = detector.predict_csv(test_file, output=result_file)
    print()

    # 異常データの詳細表示
    print("=" * 70)
    print("ステップ3: 異常データの分析")
    print("=" * 70)
    anomalies = df_result[df_result['is_anomaly'] == True]
    print(f"\n検出された異常データ: {len(anomalies)}件")
    print("\n異常度の高い順に上位5件:")
    print(anomalies.nlargest(5, 'anomaly_score')[
        ['timestamp', 'sensor1', 'sensor2', 'sensor3', 'anomaly_score']
    ].to_string(index=False))

    print(f"\n\n保存されたファイル:")
    print(f"  - {train_file}")
    print(f"  - {test_file}")
    print(f"  - {result_file}")


def demo_custom_parameters():
    """パラメータをカスタマイズした使い方"""
    print("\n" + "="*70)
    print("デモ4: パラメータをカスタマイズした使い方")
    print("="*70 + "\n")

    # データ準備
    np.random.seed(42)
    X_train = np.random.randn(500, 2)
    X_test = np.vstack([
        np.random.randn(90, 2),
        np.random.randn(10, 2) * 4 + 6
    ])

    # 異なるパラメータで比較
    configs = [
        {'bandwidth': 'scott', 'contamination': 0.1, 'name': 'デフォルト設定'},
        {'bandwidth': 'silverman', 'contamination': 0.1, 'name': 'Silverman帯域幅'},
        {'bandwidth': 0.5, 'contamination': 0.1, 'name': '固定帯域幅(0.5)'},
        {'bandwidth': 'scott', 'contamination': 0.2, 'name': '高汚染度(20%)'},
    ]

    print("異なるパラメータで比較:\n")

    for config in configs:
        detector = KDEAnomalyDetector(
            bandwidth=config['bandwidth'],
            contamination=config['contamination']
        )
        detector.fit(X_train)
        labels = detector.predict(X_test)
        n_anomalies = np.sum(labels == -1)

        print(f"{config['name']}:")
        print(f"  帯域幅: {detector.kde_model.bandwidth:.4f}")
        print(f"  閾値: {detector.threshold_:.4f}")
        print(f"  検出異常数: {n_anomalies}個 ({n_anomalies/len(X_test)*100:.1f}%)")
        print()


def demo_realtime_monitoring():
    """リアルタイム監視のシミュレーション"""
    print("\n" + "="*70)
    print("デモ5: リアルタイム監視シミュレーション")
    print("="*70 + "\n")

    # 訓練データで学習
    np.random.seed(42)
    X_train = np.random.randn(1000, 3) * np.array([5, 2, 1]) + np.array([20, 10, 5])

    detector = EasyKDEDetector(contamination=0.05)
    detector.fit(X_train, feature_names=['temperature', 'pressure', 'flow'])

    print("監視システム起動")
    print(f"学習データ: {X_train.shape[0]}サンプル")
    print(f"監視する特徴量: temperature, pressure, flow")
    print("-" * 70)

    # リアルタイムデータのシミュレーション
    print("\nリアルタイムデータ監視中...\n")

    for i in range(20):
        # ランダムにデータを生成（時々異常を含む）
        if i in [5, 12, 17]:  # 異常データ
            sample = np.random.randn(3) * np.array([10, 5, 3]) + np.array([40, 20, 15])
            is_true_anomaly = True
        else:  # 正常データ
            sample = np.random.randn(3) * np.array([5, 2, 1]) + np.array([20, 10, 5])
            is_true_anomaly = False

        # 予測
        label = detector.predict(sample.reshape(1, -1))[0]
        score = detector.detector.anomaly_score(
            detector._scale(sample.reshape(1, -1), fit=False)
        )[0]

        # 結果表示
        status = "異常" if label == -1 else "正常"
        alert = " [!!! 警告 !!!]" if label == -1 else ""
        true_label = " (実際: 異常)" if is_true_anomaly else ""

        print(f"時刻 {i:02d}: temp={sample[0]:6.2f}, "
              f"pres={sample[1]:6.2f}, "
              f"flow={sample[2]:6.2f} "
              f"=> {status} (スコア={score:.2f}){alert}{true_label}")


def main():
    """全デモを実行"""
    print("\n" + "="*70)
    print("KDE異常検知システム - デモンストレーション")
    print("="*70)

    demo_simple()
    demo_with_dataframe()
    demo_csv_workflow()
    demo_custom_parameters()
    demo_realtime_monitoring()

    print("\n" + "="*70)
    print("全デモが完了しました！")
    print("="*70)


if __name__ == "__main__":
    main()
