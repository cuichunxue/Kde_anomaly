"""
簡単な動作確認スクリプト
"""

import numpy as np
from kde_anomaly_detector import quick_detect, KDEAnomalyDetector, EasyKDEDetector

def test_basic():
    """基本動作テスト"""
    print("=" * 60)
    print("テスト1: 基本動作確認")
    print("=" * 60)

    # データ生成
    np.random.seed(42)
    X_train = np.random.randn(500, 2)
    X_test = np.vstack([
        np.random.randn(90, 2),
        np.random.randn(10, 2) * 5 + 8
    ])

    # quick_detect
    labels, scores = quick_detect(X_train, X_test, contamination=0.1)

    # 結果確認
    n_anomalies = np.sum(labels == -1)
    print(f"✓ quick_detect 成功")
    print(f"  - 総サンプル数: {len(labels)}")
    print(f"  - 検出異常数: {n_anomalies}")
    print(f"  - 異常率: {n_anomalies/len(labels)*100:.1f}%")
    print()

    return True


def test_kde_anomaly_detector():
    """KDEAnomalyDetectorテスト"""
    print("=" * 60)
    print("テスト2: KDEAnomalyDetector")
    print("=" * 60)

    np.random.seed(42)
    X_train = np.random.randn(300, 2)
    X_test = np.random.randn(50, 2)

    # モデル作成・学習
    detector = KDEAnomalyDetector(contamination=0.1)
    detector.fit(X_train)

    # 予測
    labels = detector.predict(X_test)
    scores = detector.anomaly_score(X_test)
    labels2, scores2 = detector.predict_with_score(X_test)

    print(f"✓ KDEAnomalyDetector 成功")
    print(f"  - 帯域幅: {detector.kde_model.bandwidth:.4f}")
    print(f"  - 閾値: {detector.threshold_:.4f}")
    print(f"  - 検出異常数: {np.sum(labels == -1)}")
    print(f"  - スコア範囲: {scores.min():.2f} ~ {scores.max():.2f}")
    print()

    # 検証
    assert np.array_equal(labels, labels2), "predict と predict_with_score の結果が一致しません"
    assert np.array_equal(scores, scores2), "anomaly_score と predict_with_score の結果が一致しません"

    return True


def test_easy_kde_detector():
    """EasyKDEDetectorテスト"""
    print("=" * 60)
    print("テスト3: EasyKDEDetector")
    print("=" * 60)

    np.random.seed(42)
    X_train = np.random.randn(200, 3) * 2 + 5
    X_test = np.random.randn(30, 3) * 2 + 5

    # モデル作成・学習
    detector = EasyKDEDetector(contamination=0.1, auto_scale=True)
    detector.fit(X_train, feature_names=['f1', 'f2', 'f3'])

    # 予測
    labels = detector.predict(X_test)

    # top anomalies
    top_indices, top_scores = detector.get_top_anomalies(X_test, n=3)

    print(f"✓ EasyKDEDetector 成功")
    print(f"  - 特徴量: {detector.feature_names}")
    print(f"  - 検出異常数: {np.sum(labels == -1)}")
    print(f"  - Top 3 異常インデックス: {top_indices}")
    print()

    return True


def test_1d_data():
    """1次元データのテスト"""
    print("=" * 60)
    print("テスト4: 1次元データ")
    print("=" * 60)

    np.random.seed(42)
    X_train = np.random.randn(500)
    X_test = np.concatenate([
        np.random.randn(45),
        np.random.randn(5) * 3 + 10
    ])

    detector = KDEAnomalyDetector(contamination=0.1)
    detector.fit(X_train)
    labels = detector.predict(X_test)

    print(f"✓ 1次元データ処理 成功")
    print(f"  - 検出異常数: {np.sum(labels == -1)}")
    print()

    return True


def test_different_kernels():
    """異なるカーネルのテスト"""
    print("=" * 60)
    print("テスト5: 異なるカーネル関数")
    print("=" * 60)

    np.random.seed(42)
    X_train = np.random.randn(200, 2)
    X_test = np.random.randn(30, 2)

    kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential']

    for kernel in kernels:
        detector = KDEAnomalyDetector(kernel=kernel, contamination=0.1)
        detector.fit(X_train)
        labels = detector.predict(X_test)
        n_anomalies = np.sum(labels == -1)

        print(f"  ✓ {kernel:15s}: {n_anomalies}個の異常を検出")

    print()
    return True


def test_different_bandwidths():
    """異なる帯域幅のテスト"""
    print("=" * 60)
    print("テスト6: 異なる帯域幅選択方法")
    print("=" * 60)

    np.random.seed(42)
    X_train = np.random.randn(200, 2)
    X_test = np.random.randn(30, 2)

    bandwidths = ['scott', 'silverman', 0.5, 1.0]

    for bw in bandwidths:
        detector = KDEAnomalyDetector(bandwidth=bw, contamination=0.1)
        detector.fit(X_train)
        labels = detector.predict(X_test)
        n_anomalies = np.sum(labels == -1)
        actual_bw = detector.kde_model.bandwidth

        print(f"  ✓ {str(bw):15s}: 実際の帯域幅={actual_bw:.4f}, 異常={n_anomalies}個")

    print()
    return True


def main():
    """全テスト実行"""
    print("\n" + "=" * 60)
    print("KDE異常検知システム - 動作確認テスト")
    print("=" * 60 + "\n")

    tests = [
        test_basic,
        test_kde_anomaly_detector,
        test_easy_kde_detector,
        test_1d_data,
        test_different_kernels,
        test_different_bandwidths,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} 失敗: {str(e)}\n")
            failed += 1

    print("=" * 60)
    print("テスト結果")
    print("=" * 60)
    print(f"成功: {passed}/{len(tests)}")
    print(f"失敗: {failed}/{len(tests)}")

    if failed == 0:
        print("\n全てのテストが成功しました！✓")
    else:
        print(f"\n{failed}個のテストが失敗しました。")

    print("=" * 60)


if __name__ == "__main__":
    main()
