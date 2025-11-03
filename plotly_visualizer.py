"""
KDE異常検知システム - Plotly可視化モジュール

時系列データの異常度推移を対話的に可視化
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from kde_anomaly_detector import KDEAnomalyDetector, EasyKDEDetector


def plot_anomaly_score_timeline(
    timestamps,
    scores,
    labels=None,
    threshold=None,
    title="異常度スコアの推移",
    output_file=None,
    show=True
):
    """
    異常度スコアの時系列プロット

    パラメータ:
    -----------
    timestamps : array-like
        タイムスタンプ（datetime または文字列）
    scores : array-like
        異常度スコア
    labels : array-like, optional
        異常判定ラベル（1: 正常, -1: 異常）
    threshold : float, optional
        異常判定の閾値
    title : str, default="異常度スコアの推移"
        グラフタイトル
    output_file : str, optional
        HTMLファイルとして保存するパス
    show : bool, default=True
        ブラウザで表示するか

    戻り値:
    -------
    fig : plotly.graph_objects.Figure
        Plotlyのfigureオブジェクト

    使用例:
    -------
    >>> fig = plot_anomaly_score_timeline(
    ...     timestamps=df['timestamp'],
    ...     scores=anomaly_scores,
    ...     labels=labels,
    ...     threshold=detector.threshold_,
    ...     output_file='anomaly_timeline.html'
    ... )
    """
    # データ準備
    timestamps = pd.to_datetime(timestamps)
    scores = np.asarray(scores)

    # 色分け用のデータ
    if labels is not None:
        labels = np.asarray(labels)
        colors = ['red' if l == -1 else 'blue' for l in labels]
        normal_mask = labels == 1
        anomaly_mask = labels == -1
    else:
        colors = 'blue'
        normal_mask = np.ones(len(scores), dtype=bool)
        anomaly_mask = np.zeros(len(scores), dtype=bool)

    # Figure作成
    fig = go.Figure()

    # 正常データ
    if np.any(normal_mask):
        fig.add_trace(go.Scatter(
            x=timestamps[normal_mask],
            y=scores[normal_mask],
            mode='lines+markers',
            name='正常',
            line=dict(color='steelblue', width=2),
            marker=dict(size=6, color='steelblue'),
            hovertemplate='<b>時刻</b>: %{x}<br>' +
                          '<b>異常度</b>: %{y:.2f}<br>' +
                          '<b>状態</b>: 正常<extra></extra>'
        ))

    # 異常データ
    if np.any(anomaly_mask):
        fig.add_trace(go.Scatter(
            x=timestamps[anomaly_mask],
            y=scores[anomaly_mask],
            mode='markers',
            name='異常',
            marker=dict(
                size=12,
                color='red',
                symbol='x',
                line=dict(width=2, color='darkred')
            ),
            hovertemplate='<b>時刻</b>: %{x}<br>' +
                          '<b>異常度</b>: %{y:.2f}<br>' +
                          '<b>状態</b>: 異常<extra></extra>'
        ))

    # 閾値ライン
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"閾値 = {threshold:.2f}",
            annotation_position="right"
        )

    # レイアウト設定
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family='Arial, sans-serif')
        ),
        xaxis_title="時刻",
        yaxis_title="異常度スコア",
        hovermode='closest',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )

    # 保存
    if output_file:
        fig.write_html(output_file)
        print(f"グラフを保存: {output_file}")

    # 表示
    if show:
        fig.show()

    return fig


def plot_anomaly_score_with_features(
    timestamps,
    scores,
    features_df,
    labels=None,
    threshold=None,
    title="異常度スコアと特徴量の推移",
    output_file=None,
    show=True
):
    """
    異常度スコアと特徴量を同時にプロット

    パラメータ:
    -----------
    timestamps : array-like
        タイムスタンプ
    scores : array-like
        異常度スコア
    features_df : DataFrame
        特徴量のDataFrame
    labels : array-like, optional
        異常判定ラベル
    threshold : float, optional
        異常判定の閾値
    title : str
        グラフタイトル
    output_file : str, optional
        HTMLファイルとして保存するパス
    show : bool, default=True
        ブラウザで表示するか

    戻り値:
    -------
    fig : plotly.graph_objects.Figure

    使用例:
    -------
    >>> fig = plot_anomaly_score_with_features(
    ...     timestamps=df['timestamp'],
    ...     scores=scores,
    ...     features_df=df[['temp', 'pressure', 'vibration']],
    ...     labels=labels,
    ...     threshold=detector.threshold_
    ... )
    """
    timestamps = pd.to_datetime(timestamps)
    scores = np.asarray(scores)

    # 色分け
    if labels is not None:
        labels = np.asarray(labels)
        colors = ['red' if l == -1 else 'blue' for l in labels]
    else:
        colors = 'blue'

    # サブプロット作成
    n_features = len(features_df.columns)
    fig = make_subplots(
        rows=n_features + 1,
        cols=1,
        subplot_titles=['異常度スコア'] + list(features_df.columns),
        vertical_spacing=0.05,
        row_heights=[1.5] + [1] * n_features
    )

    # 異常度スコアのプロット
    if labels is not None:
        normal_mask = labels == 1
        anomaly_mask = labels == -1

        # 正常データ
        if np.any(normal_mask):
            fig.add_trace(
                go.Scatter(
                    x=timestamps[normal_mask],
                    y=scores[normal_mask],
                    mode='lines+markers',
                    name='正常',
                    line=dict(color='steelblue', width=2),
                    marker=dict(size=4),
                    showlegend=True
                ),
                row=1, col=1
            )

        # 異常データ
        if np.any(anomaly_mask):
            fig.add_trace(
                go.Scatter(
                    x=timestamps[anomaly_mask],
                    y=scores[anomaly_mask],
                    mode='markers',
                    name='異常',
                    marker=dict(size=10, color='red', symbol='x'),
                    showlegend=True
                ),
                row=1, col=1
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=scores,
                mode='lines+markers',
                name='異常度',
                line=dict(color='steelblue', width=2)
            ),
            row=1, col=1
        )

    # 閾値ライン
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="orange",
            row=1, col=1
        )

    # 各特徴量のプロット
    feature_colors = px.colors.qualitative.Plotly
    for i, col in enumerate(features_df.columns):
        values = features_df[col].values

        if labels is not None:
            # 正常データ
            if np.any(normal_mask):
                fig.add_trace(
                    go.Scatter(
                        x=timestamps[normal_mask],
                        y=values[normal_mask],
                        mode='lines',
                        name=col,
                        line=dict(color=feature_colors[i % len(feature_colors)], width=1.5),
                        showlegend=False
                    ),
                    row=i+2, col=1
                )

            # 異常箇所をハイライト
            if np.any(anomaly_mask):
                fig.add_trace(
                    go.Scatter(
                        x=timestamps[anomaly_mask],
                        y=values[anomaly_mask],
                        mode='markers',
                        marker=dict(size=8, color='red', symbol='x'),
                        showlegend=False
                    ),
                    row=i+2, col=1
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines',
                    name=col,
                    line=dict(color=feature_colors[i % len(feature_colors)], width=1.5)
                ),
                row=i+2, col=1
            )

    # レイアウト設定
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20)
        ),
        height=300 * (n_features + 1),
        hovermode='x unified',
        template='plotly_white',
        showlegend=True
    )

    # X軸の設定（最下行のみラベル表示）
    for i in range(n_features + 1):
        if i < n_features:
            fig.update_xaxes(showticklabels=False, row=i+1, col=1)
        else:
            fig.update_xaxes(title_text="時刻", row=i+1, col=1)

    # 保存
    if output_file:
        fig.write_html(output_file)
        print(f"グラフを保存: {output_file}")

    # 表示
    if show:
        fig.show()

    return fig


def plot_anomaly_heatmap(
    timestamps,
    features_df,
    scores,
    title="異常度ヒートマップ",
    output_file=None,
    show=True
):
    """
    時系列での異常度をヒートマップで可視化

    パラメータ:
    -----------
    timestamps : array-like
        タイムスタンプ
    features_df : DataFrame
        特徴量のDataFrame
    scores : array-like
        異常度スコア
    title : str
        グラフタイトル
    output_file : str, optional
        HTMLファイルとして保存するパス
    show : bool, default=True
        ブラウザで表示するか

    戻り値:
    -------
    fig : plotly.graph_objects.Figure
    """
    timestamps = pd.to_datetime(timestamps)
    scores = np.asarray(scores)

    # データ準備（特徴量を標準化）
    features_normalized = (features_df - features_df.mean()) / features_df.std()

    # スコアを正規化して色の強度に使用
    scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

    fig = go.Figure()

    # ヒートマップ
    fig.add_trace(go.Heatmap(
        z=features_normalized.T.values,
        x=timestamps,
        y=features_df.columns,
        colorscale='RdYlBu_r',
        hovertemplate='<b>時刻</b>: %{x}<br>' +
                      '<b>特徴量</b>: %{y}<br>' +
                      '<b>正規化値</b>: %{z:.2f}<extra></extra>'
    ))

    # 異常度を示すマーカーを上部に追加
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[features_df.columns[-1]] * len(timestamps),
        mode='markers',
        marker=dict(
            size=scores_normalized * 20 + 5,
            color=scores_normalized,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(
                title="異常度",
                y=1.1,
                len=0.3
            )
        ),
        showlegend=False,
        text=[f'{s:.2f}' for s in scores],
        hovertemplate='<b>時刻</b>: %{x}<br>' +
                      '<b>異常度</b>: %{text}<extra></extra>'
    ))

    # レイアウト
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="時刻",
        yaxis_title="特徴量",
        height=400,
        template='plotly_white'
    )

    # 保存
    if output_file:
        fig.write_html(output_file)
        print(f"グラフを保存: {output_file}")

    # 表示
    if show:
        fig.show()

    return fig


def plot_anomaly_distribution(
    scores,
    labels=None,
    threshold=None,
    title="異常度スコアの分布",
    output_file=None,
    show=True
):
    """
    異常度スコアのヒストグラム

    パラメータ:
    -----------
    scores : array-like
        異常度スコア
    labels : array-like, optional
        異常判定ラベル
    threshold : float, optional
        異常判定の閾値
    title : str
        グラフタイトル
    output_file : str, optional
        HTMLファイルとして保存するパス
    show : bool, default=True
        ブラウザで表示するか

    戻り値:
    -------
    fig : plotly.graph_objects.Figure
    """
    scores = np.asarray(scores)

    fig = go.Figure()

    if labels is not None:
        labels = np.asarray(labels)
        normal_scores = scores[labels == 1]
        anomaly_scores = scores[labels == -1]

        # 正常データのヒストグラム
        fig.add_trace(go.Histogram(
            x=normal_scores,
            name='正常',
            marker_color='steelblue',
            opacity=0.7,
            nbinsx=50
        ))

        # 異常データのヒストグラム
        fig.add_trace(go.Histogram(
            x=anomaly_scores,
            name='異常',
            marker_color='red',
            opacity=0.7,
            nbinsx=50
        ))
    else:
        fig.add_trace(go.Histogram(
            x=scores,
            marker_color='steelblue',
            opacity=0.7,
            nbinsx=50
        ))

    # 閾値ライン
    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="orange",
            line_width=3,
            annotation_text=f"閾値 = {threshold:.2f}",
            annotation_position="top"
        )

    # レイアウト
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="異常度スコア",
        yaxis_title="頻度",
        barmode='overlay',
        template='plotly_white',
        height=400,
        showlegend=True
    )

    # 保存
    if output_file:
        fig.write_html(output_file)
        print(f"グラフを保存: {output_file}")

    # 表示
    if show:
        fig.show()

    return fig


def create_anomaly_dashboard(
    timestamps,
    scores,
    features_df,
    labels=None,
    threshold=None,
    output_file='anomaly_dashboard.html'
):
    """
    包括的な異常検知ダッシュボードを作成

    パラメータ:
    -----------
    timestamps : array-like
        タイムスタンプ
    scores : array-like
        異常度スコア
    features_df : DataFrame
        特徴量のDataFrame
    labels : array-like, optional
        異常判定ラベル
    threshold : float, optional
        異常判定の閾値
    output_file : str, default='anomaly_dashboard.html'
        HTMLファイルとして保存するパス

    戻り値:
    -------
    None (HTMLファイルを作成)

    使用例:
    -------
    >>> create_anomaly_dashboard(
    ...     timestamps=df['timestamp'],
    ...     scores=scores,
    ...     features_df=df[['temp', 'pressure', 'vibration']],
    ...     labels=labels,
    ...     threshold=detector.threshold_
    ... )
    """
    print("=" * 70)
    print("異常検知ダッシュボードを作成中...")
    print("=" * 70)

    # 統計情報
    if labels is not None:
        n_anomalies = np.sum(labels == -1)
        anomaly_rate = n_anomalies / len(labels) * 100
        print(f"\n統計情報:")
        print(f"  総サンプル数: {len(labels)}")
        print(f"  異常検出数: {n_anomalies} ({anomaly_rate:.1f}%)")
        print(f"  正常: {np.sum(labels == 1)} ({100-anomaly_rate:.1f}%)")

    print(f"  異常度スコア範囲: {scores.min():.2f} ~ {scores.max():.2f}")
    if threshold is not None:
        print(f"  閾値: {threshold:.2f}")

    # サブプロット作成
    n_features = len(features_df.columns)
    fig = make_subplots(
        rows=2 + n_features,
        cols=2,
        subplot_titles=[
            '異常度スコアの推移', '異常度スコアの分布',
            *[f'{col}の推移' for col in features_df.columns],
            '', ''  # 最後の行用
        ],
        specs=[
            [{"colspan": 2}, None],  # 1行目: 異常度推移（全幅）
            [{"type": "histogram"}, {"type": "scatter"}],  # 2行目: ヒストグラムと散布図
            *[[{"type": "scatter"}, {"type": "scatter"}] for _ in range(n_features)]
        ],
        vertical_spacing=0.08,
        row_heights=[2] + [1] * (1 + n_features)
    )

    timestamps = pd.to_datetime(timestamps)
    scores = np.asarray(scores)

    if labels is not None:
        normal_mask = labels == 1
        anomaly_mask = labels == -1
    else:
        normal_mask = np.ones(len(scores), dtype=bool)
        anomaly_mask = np.zeros(len(scores), dtype=bool)

    # 1. 異常度スコアの推移（1行目、全幅）
    if np.any(normal_mask):
        fig.add_trace(
            go.Scatter(
                x=timestamps[normal_mask],
                y=scores[normal_mask],
                mode='lines+markers',
                name='正常',
                line=dict(color='steelblue', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )

    if np.any(anomaly_mask):
        fig.add_trace(
            go.Scatter(
                x=timestamps[anomaly_mask],
                y=scores[anomaly_mask],
                mode='markers',
                name='異常',
                marker=dict(size=10, color='red', symbol='x'),
            ),
            row=1, col=1
        )

    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color="orange", row=1, col=1)

    # 2. 異常度スコアの分布（2行目左）
    if labels is not None:
        fig.add_trace(
            go.Histogram(
                x=scores[normal_mask],
                name='正常',
                marker_color='steelblue',
                opacity=0.7,
                nbinsx=30,
                showlegend=False
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(
                x=scores[anomaly_mask],
                name='異常',
                marker_color='red',
                opacity=0.7,
                nbinsx=30,
                showlegend=False
            ),
            row=2, col=1
        )
    else:
        fig.add_trace(
            go.Histogram(x=scores, marker_color='steelblue', nbinsx=30),
            row=2, col=1
        )

    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color="orange", row=2, col=1)

    # 3. 特徴量の散布図（2行目右）: 最初の2つの特徴量
    if len(features_df.columns) >= 2:
        col1, col2 = features_df.columns[0], features_df.columns[1]
        if np.any(normal_mask):
            fig.add_trace(
                go.Scatter(
                    x=features_df[col1].values[normal_mask],
                    y=features_df[col2].values[normal_mask],
                    mode='markers',
                    name='正常',
                    marker=dict(size=6, color='steelblue', opacity=0.6),
                    showlegend=False
                ),
                row=2, col=2
            )
        if np.any(anomaly_mask):
            fig.add_trace(
                go.Scatter(
                    x=features_df[col1].values[anomaly_mask],
                    y=features_df[col2].values[anomaly_mask],
                    mode='markers',
                    name='異常',
                    marker=dict(size=10, color='red', symbol='x'),
                    showlegend=False
                ),
                row=2, col=2
            )
        fig.update_xaxes(title_text=col1, row=2, col=2)
        fig.update_yaxes(title_text=col2, row=2, col=2)

    # 4. 各特徴量の時系列推移（3行目以降）
    feature_colors = px.colors.qualitative.Plotly
    for i, col in enumerate(features_df.columns):
        values = features_df[col].values
        row_idx = 3 + i

        # 左列: 時系列推移
        if np.any(normal_mask):
            fig.add_trace(
                go.Scatter(
                    x=timestamps[normal_mask],
                    y=values[normal_mask],
                    mode='lines',
                    line=dict(color=feature_colors[i % len(feature_colors)], width=2),
                    showlegend=False
                ),
                row=row_idx, col=1
            )

        if np.any(anomaly_mask):
            fig.add_trace(
                go.Scatter(
                    x=timestamps[anomaly_mask],
                    y=values[anomaly_mask],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='x'),
                    showlegend=False
                ),
                row=row_idx, col=1
            )

        fig.update_yaxes(title_text=col, row=row_idx, col=1)

        # 右列: 異常度との相関
        if np.any(normal_mask):
            fig.add_trace(
                go.Scatter(
                    x=values[normal_mask],
                    y=scores[normal_mask],
                    mode='markers',
                    marker=dict(size=4, color='steelblue', opacity=0.5),
                    showlegend=False
                ),
                row=row_idx, col=2
            )
        if np.any(anomaly_mask):
            fig.add_trace(
                go.Scatter(
                    x=values[anomaly_mask],
                    y=scores[anomaly_mask],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='x'),
                    showlegend=False
                ),
                row=row_idx, col=2
            )

        fig.update_xaxes(title_text=col, row=row_idx, col=2)
        fig.update_yaxes(title_text="異常度", row=row_idx, col=2)

    # レイアウト設定
    fig.update_layout(
        title=dict(
            text="異常検知ダッシュボード",
            font=dict(size=24)
        ),
        height=400 + 300 * n_features,
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        barmode='overlay'
    )

    # 保存
    fig.write_html(output_file)
    print(f"\n✓ ダッシュボードを保存: {output_file}")
    print("  ブラウザで開いて対話的に操作できます")
    print("=" * 70)
