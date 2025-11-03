"""
Plotly可視化機能のデモ

時系列データの異常度推移を対話的に可視化する例
"""

import numpy as np
import pandas as pd
from kde_anomaly_detector import EasyKDEDetector
from plotly_visualizer import (
    plot_anomaly_score_timeline,
    plot_anomaly_score_with_features,
    plot_anomaly_heatmap,
    plot_anomaly_distribution,
    create_anomaly_dashboard
)


def generate_sensor_data_with_anomalies(n_normal=500, n_anomaly=50):
    """
    異常を含むセンサーデータを生成

    パラメータ:
    -----------
    n_normal : int
        正常データのサンプル数
    n_anomaly : int
        異常データのサンプル数

    戻り値:
    -------
    df : DataFrame
        センサーデータ
    """
    np.random.seed(42)

    # 正常データの生成
    timestamps_normal = pd.date_range('2024-01-01', periods=n_normal, freq='1H')
    temp_normal = 25 + 5 * np.sin(np.linspace(0, 4*np.pi, n_normal)) + np.random.randn(n_normal) * 2
    pressure_normal = 1.0 + 0.1 * np.sin(np.linspace(0, 4*np.pi, n_normal)) + np.random.randn(n_normal) * 0.05
    vibration_normal = 0.5 + 0.1 * np.sin(np.linspace(0, 4*np.pi, n_normal)) + np.random.randn(n_normal) * 0.05

    df_normal = pd.DataFrame({
        'timestamp': timestamps_normal,
        'temperature': temp_normal,
        'pressure': pressure_normal,
        'vibration': vibration_normal,
        'true_label': 1  # 正常
    })

    # 異常データの生成（ランダムな位置に挿入）
    anomaly_indices = np.random.choice(n_normal, n_anomaly, replace=False)
    anomaly_indices = np.sort(anomaly_indices)

    for idx in anomaly_indices:
        # 異常パターン1: 温度異常（スパイク）
        if np.random.rand() < 0.4:
            df_normal.loc[idx, 'temperature'] += np.random.uniform(15, 25)

        # 異常パターン2: 圧力異常（急上昇）
        if np.random.rand() < 0.3:
            df_normal.loc[idx, 'pressure'] += np.random.uniform(0.5, 1.0)

        # 異常パターン3: 振動異常（急増）
        if np.random.rand() < 0.3:
            df_normal.loc[idx, 'vibration'] += np.random.uniform(0.5, 1.5)

        # 異常パターン4: 複合異常
        if np.random.rand() < 0.2:
            df_normal.loc[idx, 'temperature'] += np.random.uniform(10, 15)
            df_normal.loc[idx, 'pressure'] += np.random.uniform(0.3, 0.7)

        df_normal.loc[idx, 'true_label'] = -1  # 異常

    return df_normal


def demo_basic_timeline():
    """基本的な時系列プロット"""
    print("\n" + "="*70)
    print("デモ1: 基本的な異常度推移プロット")
    print("="*70)

    # データ生成
    df = generate_sensor_data_with_anomalies(n_normal=300, n_anomaly=30)

    # モデル学習（正常データのみ）
    df_train = df[df['true_label'] == 1].head(200)
    features = ['temperature', 'pressure', 'vibration']

    detector = EasyKDEDetector(contamination=0.1)
    detector.fit(df_train[features].values, feature_names=features)

    # 予測
    labels = detector.predict(df[features].values)
    scores = detector.detector.anomaly_score(
        detector._scale(df[features].values, fit=False)
    )

    print(f"\nデータ情報:")
    print(f"  総サンプル数: {len(df)}")
    print(f"  真の異常: {np.sum(df['true_label'] == -1)}個")
    print(f"  検出異常: {np.sum(labels == -1)}個")
    print(f"  正解率: {np.mean(labels == df['true_label'].values):.2%}")

    # プロット
    fig = plot_anomaly_score_timeline(
        timestamps=df['timestamp'],
        scores=scores,
        labels=labels,
        threshold=detector.detector.threshold_,
        title="センサーデータの異常度推移",
        output_file='output_timeline.html',
        show=False
    )

    print(f"\n✓ 基本的な時系列プロットが完成しました")


def demo_with_features():
    """特徴量を含む詳細プロット"""
    print("\n" + "="*70)
    print("デモ2: 特徴量を含む詳細プロット")
    print("="*70)

    # データ生成
    df = generate_sensor_data_with_anomalies(n_normal=200, n_anomaly=20)

    # モデル学習
    df_train = df[df['true_label'] == 1].head(150)
    features = ['temperature', 'pressure', 'vibration']

    detector = EasyKDEDetector(contamination=0.1)
    detector.fit(df_train[features].values, feature_names=features)

    # 予測
    labels = detector.predict(df[features].values)
    scores = detector.detector.anomaly_score(
        detector._scale(df[features].values, fit=False)
    )

    print(f"\nデータ情報:")
    print(f"  特徴量: {', '.join(features)}")
    print(f"  検出異常: {np.sum(labels == -1)}個")

    # プロット
    fig = plot_anomaly_score_with_features(
        timestamps=df['timestamp'],
        scores=scores,
        features_df=df[features],
        labels=labels,
        threshold=detector.detector.threshold_,
        title="異常度と各特徴量の推移",
        output_file='output_with_features.html',
        show=False
    )

    print(f"\n✓ 特徴量を含む詳細プロットが完成しました")


def demo_heatmap():
    """ヒートマップ可視化"""
    print("\n" + "="*70)
    print("デモ3: ヒートマップ可視化")
    print("="*70)

    # データ生成
    df = generate_sensor_data_with_anomalies(n_normal=150, n_anomaly=15)

    # モデル学習
    df_train = df[df['true_label'] == 1].head(100)
    features = ['temperature', 'pressure', 'vibration']

    detector = EasyKDEDetector(contamination=0.1)
    detector.fit(df_train[features].values, feature_names=features)

    # 予測
    scores = detector.detector.anomaly_score(
        detector._scale(df[features].values, fit=False)
    )

    print(f"\nデータ情報:")
    print(f"  サンプル数: {len(df)}")
    print(f"  特徴量数: {len(features)}")

    # プロット
    fig = plot_anomaly_heatmap(
        timestamps=df['timestamp'],
        features_df=df[features],
        scores=scores,
        title="特徴量と異常度のヒートマップ",
        output_file='output_heatmap.html',
        show=False
    )

    print(f"\n✓ ヒートマップが完成しました")


def demo_distribution():
    """異常度分布の可視化"""
    print("\n" + "="*70)
    print("デモ4: 異常度分布の可視化")
    print("="*70)

    # データ生成
    df = generate_sensor_data_with_anomalies(n_normal=400, n_anomaly=40)

    # モデル学習
    df_train = df[df['true_label'] == 1].head(300)
    features = ['temperature', 'pressure', 'vibration']

    detector = EasyKDEDetector(contamination=0.1)
    detector.fit(df_train[features].values, feature_names=features)

    # 予測
    labels = detector.predict(df[features].values)
    scores = detector.detector.anomaly_score(
        detector._scale(df[features].values, fit=False)
    )

    print(f"\nデータ情報:")
    print(f"  正常データのスコア: 平均={scores[labels==1].mean():.2f}, "
          f"標準偏差={scores[labels==1].std():.2f}")
    print(f"  異常データのスコア: 平均={scores[labels==-1].mean():.2f}, "
          f"標準偏差={scores[labels==-1].std():.2f}")

    # プロット
    fig = plot_anomaly_distribution(
        scores=scores,
        labels=labels,
        threshold=detector.detector.threshold_,
        title="異常度スコアの分布",
        output_file='output_distribution.html',
        show=False
    )

    print(f"\n✓ 異常度分布プロットが完成しました")


def demo_dashboard():
    """包括的なダッシュボード作成"""
    print("\n" + "="*70)
    print("デモ5: 包括的な異常検知ダッシュボード")
    print("="*70)

    # データ生成
    df = generate_sensor_data_with_anomalies(n_normal=400, n_anomaly=40)

    # モデル学習
    df_train = df[df['true_label'] == 1].head(300)
    features = ['temperature', 'pressure', 'vibration']

    detector = EasyKDEDetector(contamination=0.1)
    detector.fit(df_train[features].values, feature_names=features)

    print(f"\nモデル情報:")
    print(f"  帯域幅: {detector.detector.kde_model.bandwidth:.4f}")
    print(f"  閾値: {detector.detector.threshold_:.4f}")

    # 予測
    labels = detector.predict(df[features].values)
    scores = detector.detector.anomaly_score(
        detector._scale(df[features].values, fit=False)
    )

    # ダッシュボード作成
    create_anomaly_dashboard(
        timestamps=df['timestamp'],
        scores=scores,
        features_df=df[features],
        labels=labels,
        threshold=detector.detector.threshold_,
        output_file='anomaly_dashboard.html'
    )


def demo_realtime_simulation():
    """リアルタイム監視のシミュレーション"""
    print("\n" + "="*70)
    print("デモ6: リアルタイム監視シミュレーション")
    print("="*70)

    # 訓練データで学習
    np.random.seed(42)
    df_train = pd.DataFrame({
        'temperature': np.random.randn(500) * 5 + 25,
        'pressure': np.random.randn(500) * 0.5 + 1.0,
        'vibration': np.random.randn(500) * 0.1 + 0.5
    })

    features = ['temperature', 'pressure', 'vibration']
    detector = EasyKDEDetector(contamination=0.05)
    detector.fit(df_train[features].values, feature_names=features)

    print(f"\n監視システム起動")
    print(f"  モデル準備完了")
    print(f"  閾値: {detector.detector.threshold_:.4f}")

    # リアルタイムデータのシミュレーション
    n_points = 100
    timestamps = pd.date_range('2024-01-01', periods=n_points, freq='1min')
    all_scores = []
    all_labels = []

    print(f"\nリアルタイムデータ収集中...")

    for i in range(n_points):
        # データ生成（時々異常を含む）
        if i in [20, 35, 50, 72, 88]:  # 異常データ
            sample = np.array([
                25 + np.random.uniform(15, 25),
                1.0 + np.random.uniform(0.5, 1.0),
                0.5 + np.random.uniform(0.5, 1.0)
            ])
        else:  # 正常データ
            sample = np.array([
                25 + np.random.randn() * 5,
                1.0 + np.random.randn() * 0.5,
                0.5 + np.random.randn() * 0.1
            ])

        # 予測
        label = detector.predict(sample.reshape(1, -1))[0]
        score = detector.detector.anomaly_score(
            detector._scale(sample.reshape(1, -1), fit=False)
        )[0]

        all_scores.append(score)
        all_labels.append(label)

    # 結果をDataFrameに
    df_result = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': [25 + np.random.randn() * 5 for _ in range(n_points)],
        'pressure': [1.0 + np.random.randn() * 0.5 for _ in range(n_points)],
        'vibration': [0.5 + np.random.randn() * 0.1 for _ in range(n_points)]
    })

    print(f"\n検出結果:")
    print(f"  総測定回数: {n_points}")
    print(f"  異常検出: {np.sum(np.array(all_labels) == -1)}回")

    # 可視化
    plot_anomaly_score_timeline(
        timestamps=timestamps,
        scores=np.array(all_scores),
        labels=np.array(all_labels),
        threshold=detector.detector.threshold_,
        title="リアルタイム監視 - 異常度推移",
        output_file='output_realtime.html',
        show=False
    )

    print(f"\n✓ リアルタイム監視結果が保存されました")


def main():
    """全デモを実行"""
    print("\n" + "="*70)
    print("Plotly可視化デモ - 異常度の推移を対話的に可視化")
    print("="*70)

    # 各デモを実行
    demo_basic_timeline()
    demo_with_features()
    demo_heatmap()
    demo_distribution()
    demo_dashboard()
    demo_realtime_simulation()

    print("\n" + "="*70)
    print("全てのデモが完了しました！")
    print("="*70)
    print("\n作成されたHTMLファイル:")
    print("  1. output_timeline.html - 基本的な異常度推移")
    print("  2. output_with_features.html - 特徴量を含む詳細プロット")
    print("  3. output_heatmap.html - ヒートマップ可視化")
    print("  4. output_distribution.html - 異常度分布")
    print("  5. anomaly_dashboard.html - 包括的ダッシュボード")
    print("  6. output_realtime.html - リアルタイム監視シミュレーション")
    print("\nこれらのファイルをブラウザで開いて対話的に操作できます")
    print("  - ズーム: ドラッグで範囲選択")
    print("  - パン: ダブルクリックでリセット")
    print("  - ホバー: データポイントにカーソルを合わせて詳細表示")
    print("="*70)


if __name__ == "__main__":
    main()
