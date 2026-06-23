"""
(copyLeft) yRocket, 2026.4.8
Data Generation & 1:1 Line Comparison Chart
- Gaussian distribution 기반 2D 데이터 생성
- shape, offset, angle 파라미터로 분포 형태 제어
- 17개 통계 메트릭 계산 및 시각화
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

    # Scale: major axis = sigma * shape (x축, 0°), minor axis = sigma (y축, 90°)
    # → angle_deg 회전 시 major axis가 angle_deg 방향으로 감
    z[:, 0] *= sigma * shape   # major axis (0° 방향)
    z[:, 1] *= sigma           # minor axis (90° 방향)

    # Rotate by angle
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]])
    data = z @ R.T

    # 1:1 라인 위의 중심점: (mu, mu)
    # offset: 1:1 라인에 수직인 방향 (-1, 1)/sqrt(2) 으로 이동
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
    """17개 통계 메트릭 계산."""
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)

    # Pearson R
    r_val, _ = stats.pearsonr(x, y)
    r2 = r_val ** 2

    # CCC components
    v = sx / sy if sy > 0 else 1.0
    u = (mx - my) / np.sqrt(sx * sy) if (sx * sy) > 0 else 0.0
    cb = 2.0 / (v + 1.0 / v + u ** 2)
    ccc = r_val * cb

    # CAI = 1 / (1 + u²)
    cai = 1.0 / (1.0 + u ** 2)

    # Differences (d = x - y)
    d = x - y
    abs_d = np.abs(d)

    rmsd_1to1 = np.sqrt(np.mean(d ** 2))
    mad_1to1 = np.mean(abs_d)
    tdi = np.percentile(abs_d, 95)

    # CP: P(|x-y| < threshold), threshold = 10% of overall mean
    threshold = 0.1 * np.mean(np.concatenate([x, y]))
    cp = np.mean(abs_d < threshold) if threshold > 0 else 0.0

    rmse = rmsd_1to1
    mae = mad_1to1
    mbe = np.mean(y - x)
    mb = np.mean(d)
    nmbe = (mbe / mx * 100) if mx != 0 else 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_bias = np.where(x != 0, (y - x) / x, 0.0)
    mrb = np.mean(rel_bias) * 100
    pbias = (np.sum(y - x) / np.sum(x) * 100) if np.sum(x) != 0 else 0.0
    mbd = mbe

    return {
        'CCC': ccc, 'CB': cb, 'CAI': cai,
        'RMSD1to1': rmsd_1to1, 'MAD1to1': mad_1to1,
        'TDI': tdi, 'CP': cp,
        'R': r_val, 'R2': r2,
        'RMSE': rmse, 'MAE': mae,
        'MBE': mbe, 'MB': mb, 'NMBE': nmbe,
        'MRB': mrb, 'PBIAS': pbias, 'MBD': mbd,
    }


def point_to_line_foot(px: float, py: float) -> tuple[float, float]:
    """점 (px, py)에서 1:1 라인(y=x) 위의 수선의 발 좌표."""
    mid = (px + py) / 2.0
    return mid, mid


def main() -> None:
    # --- Hyperparameters ---
    mu: float = 50.0
    n: int = 256

    shapes = {'Circle': 1, 'Ellipse': 4}
    dispersions = {'Tight': 1, 'Wide': 5}
    offsets = {'On 1:1': 0, 'Off 1:1': 20}
    angles = {'45°': 45, '70°': 70, '90°': 90, '135°': 135}

    print("=" * 80)
    print("HYPERPARAMETERS")
    print("=" * 80)
    print(f"  n (samples)  : {n}")
    print(f"  mu (center)  : {mu}")
    print(f"  shapes       : {shapes}")
    print(f"  dispersions  : {dispersions}")
    print(f"  offsets      : {offsets}")
    print(f"  angles       : {angles}")
    print("=" * 80)

    # --- Matrix layout: 8 rows × 4 cols ---
    # Rows: (offset, angle)  = 2 × 4 = 8
    # Cols: (shape, dispersion)   = 2 × 2 = 4
    row_configs = [(o_lbl, a_lbl) for o_lbl in offsets for a_lbl in angles]
    col_configs = [(s_lbl, w_lbl) for s_lbl in shapes for w_lbl in dispersions]

    n_rows = len(row_configs)  # 8
    n_cols = len(col_configs)  # 4

    all_metrics: list[dict] = []
    all_dfs: list[pd.DataFrame] = []

    # Pre-generate data and compute metrics (print table first, then chart)
    plot_data: list[list[dict]] = [[{} for _ in range(n_cols)] for _ in range(n_rows)]

    for ri, (o_lbl, a_lbl) in enumerate(row_configs):
        for ci, (s_lbl, w_lbl) in enumerate(col_configs):
            shp = shapes[s_lbl]
            sig = dispersions[w_lbl]
            off = offsets[o_lbl]
            ang = angles[a_lbl]

            # seed: offset에 의존하지 않음 → on/off 1:1이 같은 기본 패턴 사용
            seed = hash((s_lbl, w_lbl, a_lbl)) % (2**31)

            df = generate_2d_gaussian(
                n=n, mu=mu, sigma=sig, shape=shp,
                offset=off, angle_deg=ang, random_seed=seed,
                shape_label=s_lbl, dispersion_label=w_lbl,
                offset_label=o_lbl, angle_label=a_lbl,
            )
            all_dfs.append(df)

            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            m = calc_metrics(x, y)

            label = f"{s_lbl}, {w_lbl}, {o_lbl}, {a_lbl}"
            m['Label'] = label
            all_metrics.append(m)

            plot_data[ri][ci] = {'x': x, 'y': y, 'm': m, 'label': label,
                                 's_lbl': s_lbl, 'w_lbl': w_lbl,
                                 'o_lbl': o_lbl, 'a_lbl': a_lbl}

    # --- Combined DataFrame ---
    combined_df = pd.concat(all_dfs, axis=1)

    # --- Metrics Table (print first) ---
    df_metrics = pd.DataFrame(all_metrics)
    metric_cols = ['Label', 'CCC', 'CB', 'CAI', 'RMSD1to1', 'MAD1to1',
                   'TDI', 'CP', 'R', 'R2', 'RMSE', 'MAE',
                   'MBE', 'MB', 'NMBE', 'MRB', 'PBIAS', 'MBD']
    df_display = df_metrics[metric_cols].copy()
    for col in metric_cols[1:]:
        df_display[col] = df_display[col].map(lambda v: f'{v:.4f}')

    print("METRICS TABLE")
    print("=" * 120)
    print(df_display.to_string(index=False))
    print("=" * 120)

    # --- Compute global axis range (x와 y 동일 범위 → 1:1 스케일 보장) ---
    global_min = np.inf
    global_max = -np.inf
    for ri in range(n_rows):
        for ci in range(n_cols):
            pd_ = plot_data[ri][ci]
            vals = np.concatenate([pd_['x'], pd_['y']])
            global_min = min(global_min, np.min(vals))
            global_max = max(global_max, np.max(vals))
    gmin, gmax = global_min - 5, global_max + 5

    # --- Chart (8 rows × 4 cols, narrow matrix) ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 16),
                             sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0, hspace=0, left=0.08, right=0.98,
                        top=0.92, bottom=0.05)
    fig.suptitle('Mean Bias and Variance Bias vs the 1:1 Line',
                 fontsize=14, fontweight='bold', y=0.97)

    # x와 y에 동일 범위 설정 → 축 스케일 동일
    axes[0, 0].set_xlim(gmin, gmax)
    axes[0, 0].set_ylim(gmin, gmax)

    for ri in range(n_rows):
        for ci in range(n_cols):
            ax = axes[ri, ci]
            pd_ = plot_data[ri][ci]
            x, y, m = pd_['x'], pd_['y'], pd_['m']

            # 1:1 line
            ax.plot([gmin, gmax], [gmin, gmax], 'k--', lw=1, alpha=0.5)
            # Scatter
            ax.scatter(x, y, s=8, alpha=0.5, c='steelblue', edgecolors='none')
            # Data center
            cx, cy = np.mean(x), np.mean(y)
            ax.plot(cx, cy, 'r+', markersize=10, markeredgewidth=2)

            # Arrow: center → 1:1 line
            fx, fy = point_to_line_foot(cx, cy)
            dist = np.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)
            if dist > 0.3:
                ax.annotate('',
                            xy=(fx, fy), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle='->', color='red',
                                            lw=1.5, shrinkA=0, shrinkB=0))
                mid_ax, mid_ay = (cx + fx) / 2, (cy + fy) / 2
                ax.text(mid_ax, mid_ay, f'd={dist:.1f}',
                        fontsize=5, color='red', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                  alpha=0.7, edgecolor='none'))

            # Metrics text
            text_str = (f"R²={m['R2']:.3f}\n"
                        f"CAI={m['CAI']:.3f}\n"
                        f"CCC={m['CCC']:.3f}")
            ax.text(0.03, 0.97, text_str, transform=ax.transAxes,
                    fontsize=6, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow',
                              alpha=0.85, edgecolor='gray'))

            ax.tick_params(labelsize=5)

            # Hide tick labels for inner subplots
            if ri < n_rows - 1:
                ax.tick_params(labelbottom=False)
            if ci > 0:
                ax.tick_params(labelleft=False)

            # Column header (top row only)
            if ri == 0:
                ax.set_title(f"{pd_['s_lbl']}, {pd_['w_lbl']}",
                             fontsize=8, fontweight='bold')
            # Row label (left column only)
            if ci == 0:
                ax.set_ylabel(f"{pd_['o_lbl']}, {pd_['a_lbl']}",
                              fontsize=7, fontweight='bold')

    plt.savefig(f"{sys.argv[0].rstrip('.py')}.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
