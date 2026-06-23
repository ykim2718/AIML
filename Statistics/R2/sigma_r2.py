"""
(copyLeft) yRocket, 2026.4.8 - 9
Sigma Score vs R² Chart
- generate_2d_gaussian 기반 데이터 생성
- sigma score (CV = std/mean) 0.1 ~ 4.0 범위에서 R² 변화 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys


def generate_2d_gaussian(
    n: int = 256,
    mu: float = 50.0,
    sigma: float = 10.0,
    shape: float = 1.0,
    offset: float = 0.0,
    angle_deg: float = 45.0,
    random_seed: int = 42,
    shape_label: str = '',
    dispersion_label: str = '',
    offset_label: str = '',
    angle_label: str = '',
) -> pd.DataFrame:
    """
    가우시안 분포 기반 2D 데이터 생성.

    Parameters
    ----------
    n : int
        샘플 수 (default 256)
    mu : float
        1:1 라인 위의 데이터 중심 위치
    sigma : float
        기본 분포 폭 (minor axis)
    shape : float
        종횡비 (1=원형, >1=타원형, major/minor ratio)
    offset : float
        1:1 라인에서 수직 방향 이탈 거리
    angle_deg : float
        주축 각도 (45°=1:1 라인 위, 135°=1:1 라인 직각)
    random_seed : int
        재현성을 위한 시드
    shape_label, dispersion_label, offset_label, angle_label : str
        MultiIndex column 레이블

    Returns
    -------
    pd.DataFrame
        MultiIndex columns = (shape, dispersion, offset, angle, var)
    """
    rng = np.random.default_rng(random_seed)
    z = rng.standard_normal((n, 2))

    z[:, 0] *= sigma * shape   # major axis (0° 방향)
    z[:, 1] *= sigma           # minor axis (90° 방향)

    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]])
    data = z @ R.T

    perp = np.array([-1.0, 1.0]) / np.sqrt(2)
    center = np.array([mu, mu]) + offset * perp

    data[:, 0] += center[0]
    data[:, 1] += center[1]

    col_idx = pd.MultiIndex.from_tuples(
        [(shape_label, dispersion_label, offset_label, angle_label, 'x'),
         (shape_label, dispersion_label, offset_label, angle_label, 'y')],
        names=['shape', 'dispersion', 'offset', 'angle', 'var']
    )
    return pd.DataFrame(data, columns=col_idx)


def calc_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    r_val, _ = stats.pearsonr(x, y)
    r2 = r_val ** 2

    return { 'R2': r2 }


def show_charts(n_points: int = 20) -> pd.DataFrame:
    """
    Sigma score (CV = std(x)/mean(x)) vs R² 차트.

    sigma score 0.1 ~ 4.0 범위에서 R² 변화를 시각화.
    shape = 0.9 + sigma_score → sigma_score 작을수록 원형(R²≈0),
    클수록 타원형(R²↑).

    Parameters
    ----------
    n_points : int
        x축 데이터 포인트 수 (default 20)

    Returns
    -------
    pd.Series
        index = sigma_score_x, values = R2
    """
    mu = 50.0
    n = 256
    angle_deg = 45.0
    offset = 0.0
    random_seed = 42

    target_sigma_scores = np.linspace(0.1, 4.0, n_points)

    ss_x_list = []
    ss_y_list = []
    r2_list = []
    inset_data = {}  # left / center / right scatter 데이터 저장
    mid_idx = n_points // 2

    for idx, target in enumerate(target_sigma_scores):
        # shape = 0.9 + target → target=0.1 → shape=1(R²≈0), target=4 → shape=4.9(R²↑)
        shape = 0.9 + target

        # sigma 역산: sigma_score_x = std(x)/mean(x) = target
        # std(x) = sigma * sqrt((shape²+1)/2), mean(x) = mu
        # → sigma = target * mu / sqrt((shape²+1)/2)
        sigma = target * mu / np.sqrt((shape ** 2 + 1) / 2)

        df = generate_2d_gaussian(
            n=n, mu=mu, sigma=sigma, shape=shape,
            offset=offset, angle_deg=angle_deg, random_seed=random_seed,
        )
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values

        # 데이터로부터 sigma score 계산
        ss_x = np.std(x, ddof=1) / np.mean(x)
        ss_y = np.std(y, ddof=1) / np.mean(y)

        metrics = calc_metrics(x, y)

        ss_x_list.append(ss_x)
        ss_y_list.append(ss_y)
        r2_list.append(metrics['R2'])

        # left / center / right 인덱스 기록
        if idx in (0, mid_idx, n_points - 1):
            if idx == 0:
                label = 'left'
            elif idx == mid_idx:
                label = 'center'
            else:
                label = 'right'
            inset_data[label] = {
                'ss_x': ss_x, 'r2': metrics['R2'],
                'idx': idx, 'target': target,
            }

    # inset용 데이터: shape=1.0 고정으로 별도 생성
    for label, d in inset_data.items():
        inset_shape = 1.0
        # sigma_score_x = std(x)/mean(x), shape=1 → std(x)=sigma → sigma = target*mu
        inset_sigma = d['target'] * mu / np.sqrt((inset_shape ** 2 + 1) / 2)
        df_inset = generate_2d_gaussian(
            n=n, mu=mu, sigma=inset_sigma, shape=inset_shape,
            offset=offset, angle_deg=angle_deg, random_seed=random_seed,
        )
        d['x'] = df_inset.iloc[:, 0].values
        d['y'] = df_inset.iloc[:, 1].values

    # pd.Series: index = sigma_score_x, values = R²
    calculated = pd.Series(
        r2_list, index=pd.Index(ss_x_list, name='Sigma Score (x)'),
        name='R2',
    )

    # --- Chart ---
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.subplots_adjust(top=0.9, bottom=0.1)
    fig.suptitle('Sigma Score vs R²',
                 fontsize=14, fontweight='bold', y=0.97)

    ax.plot(ss_x_list, r2_list, 'o-', color='steelblue', markersize=5,
            linewidth=1.5)

    # sigma_score_x, sigma_score_y 텍스트 (매 4번째 포인트)
    step = max(1, n_points // 5)
    for i in range(0, n_points, step):
        ax.annotate(
            f'σx={ss_x_list[i]:.2f}\nσy={ss_y_list[i]:.2f}',
            (ss_x_list[i], r2_list[i]),
            textcoords='offset points', xytext=(5, 0), ha='left', va='center',
            fontsize=6, color='gray',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7,
                      edgecolor='none'),
        )

    ax.set_xlabel('Sigma Score (x) = std(x) / mean(x)', fontsize=11)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_ylim(-0.05, 1.00)
    ax.grid(True, alpha=0.3)

    # --- Inset scatter charts (left / center / right) ---
    # 모든 inset에 동일 축 스케일 적용
    global_lo = min(d['x'].min() for d in inset_data.values())
    global_lo = min(global_lo, min(d['y'].min() for d in inset_data.values()))
    global_hi = max(d['x'].max() for d in inset_data.values())
    global_hi = max(global_hi, max(d['y'].max() for d in inset_data.values()))
    margin = (global_hi - global_lo) * 0.05
    inset_lo, inset_hi = global_lo - margin, global_hi + margin

    # shape=1.0 통일 → 정사각형 inset
    inset_size = 0.25
    inset_cfgs = [
        # (key,  [x, y, w, h] in axes fraction)
        ('left',   [0.4, 0.1, inset_size, inset_size]),
        ('center', [0.1, 0.7, inset_size, inset_size]),
        ('right',  [0.8, 0.4, inset_size, inset_size]),
    ]
    for key, rect in inset_cfgs:
        d = inset_data[key]
        ax_in = ax.inset_axes(rect)
        ax_in.scatter(d['x'], d['y'], s=5, alpha=0.7, c='#F54927', edgecolors='none')
        # 1:1 line
        ax_in.plot([inset_lo, inset_hi], [inset_lo, inset_hi],
                   'k--', lw=0.8, alpha=0.5)
        ax_in.set_xlim(inset_lo, inset_hi)
        ax_in.set_ylim(inset_lo, inset_hi)
        ax_in.set_aspect('equal', adjustable='box')
        ax_in.set_title(f"σx={d['ss_x']:.2f}  R²={d['r2']:.3f}",
                        fontsize=6, pad=2)
        ax_in.tick_params(labelsize=5)

        # 흐린 화살표: scatter 데이터 포인트 → inset 차트 하단 중앙
        data_pt = (d['ss_x'], r2_list[d['idx']])
        # inset 중앙 위치 (axes fraction → data coords 근사)
        inset_bottom_x = rect[0] + rect[2] / 2  # axes fraction
        inset_bottom_y = rect[1] + rect[3] / 2
        ax.annotate(
            '', xy=(inset_bottom_x, inset_bottom_y),
            xycoords='axes fraction',
            xytext=data_pt, textcoords='data',
            arrowprops=dict(
                arrowstyle='->', color='gray', alpha=0.4,
                lw=1.2, connectionstyle='arc3,rad=0.15',
            ),
        )

    out_path = f"{sys.argv[0].rstrip('.py')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved: {out_path}")
    plt.show()

    return calculated


if __name__ == '__main__':
    calculated = show_charts()
    print("\n=== Calculated Series ===")
    print(calculated.to_string())
