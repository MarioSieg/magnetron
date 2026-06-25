from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def compact_shapes(s: str) -> str:
    return str(s).replace(', ', ',')


def shorten(s: str, n: int = 64) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n - 3] + '...'


def make_label(row, i: int) -> str:
    op = str(row['op'])
    dtype = str(row['dtype'])
    kind = str(row['kind'])
    shapes = shorten(compact_shapes(row['shapes']), 52)
    return f'#{i:02d} {op} {dtype} {kind} {shapes}'


def plot_barh(df, value_col: str, xlabel: str, title: str, out_path: Path) -> None:
    df = df.copy()
    df['label'] = [make_label(row, i) for i, row in df.iterrows()]

    plt.figure(figsize=(18, max(7, len(df) * 0.45)))
    bars = plt.barh(df['label'], df[value_col])
    # Annotate bars with values
    for bar, val in zip(bars, df[value_col]):
        plt.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height() / 2,
                 f'{val:.2f}', va='center', fontsize=7)
    plt.xlabel(xlabel)
    plt.ylabel('')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.44, right=0.96, top=0.92, bottom=0.08)
    plt.savefig(out_path, dpi=300)
    plt.close()


def render_table(df, title: str, out_path: Path) -> None:
    shown = df.copy()
    shown.insert(0, 'idx', [f'#{i:02d}' for i in range(len(shown))])
    shown['shapes'] = shown['shapes'].map(compact_shapes)
    shown['strides'] = shown['strides'].map(compact_shapes)

    cols = ['idx', 'op', 'calls', 'dtype', 'kind', 'shapes', 'strides', 'total_ms', 'avg_us', 'max_us']
    shown = shown[cols]

    fig_h = max(3, 0.38 * len(shown) + 1.2)
    fig, ax = plt.subplots(figsize=(24, fig_h))
    ax.axis('off')
    ax.set_title(title, pad=14)

    table = ax.table(
        cellText=shown.values,
        colLabels=shown.columns,
        loc='center',
        cellLoc='left',
        colLoc='left',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)

    for col in range(len(cols)):
        table.auto_set_column_width(col)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def render_summary_panel(df_total, df_avg, df_max, out_path: Path, top: int) -> None:
    """Single figure with three side-by-side horizontal bar charts."""
    fig = plt.figure(figsize=(28, max(8, top * 0.4 + 2)))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.55)

    specs = [
        (df_total, 'total_ms', 'Total time (ms)', gs[0]),
        (df_avg,   'avg_us',   'Avg latency (µs)', gs[1]),
        (df_max,   'max_us',   'Max latency (µs)', gs[2]),
    ]

    for df, col, xlabel, gslot in specs:
        ax = fig.add_subplot(gslot)
        labels = [make_label(row, i) for i, row in df.iterrows()]
        ax.barh(labels, df[col])
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelsize=7)
        ax.set_title(xlabel)

    fig.suptitle(f'Magnetron Op Profile — top {top}', fontsize=13, y=1.01)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def load_profile_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [
        'calls', 'op', 'kind', 'dtype', 'shapes', 'strides',
        'total_ms', 'avg_us', 'max_us',
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f'Missing required CSV columns: {", ".join(missing)}')

    df = df[required_cols].copy()

    df['calls'] = pd.to_numeric(df['calls'])
    df['total_ms'] = pd.to_numeric(df['total_ms'])
    df['avg_us'] = pd.to_numeric(df['avg_us'])
    df['max_us'] = pd.to_numeric(df['max_us'])

    df['op'] = df['op'].astype(str)
    df['kind'] = df['kind'].astype(str)
    df['dtype'] = df['dtype'].astype(str)
    df['shapes'] = df['shapes'].astype(str)
    df['strides'] = df['strides'].astype(str)

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--out-dir', default='.')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_profile_csv(csv_path)
    stem = csv_path.stem
    top = args.top

    views = [
        ('total_ms', 'Total time [ms]', 'total time'),
        ('avg_us',   'Average latency [µs]', 'avg latency'),
        ('max_us',   'Max latency [µs]', 'max latency'),
    ]

    parts = {}
    for col, xlabel, name in views:
        part = df.sort_values(col, ascending=False).head(top).reset_index(drop=True)
        parts[col] = part

        plot_barh(
            part, col, xlabel,
            f'Top {len(part)} Magnetron ops by {name}',
            out_dir / f'{stem}_{col}.png',
        )

        render_table(
            part,
            f'Legend/table for top {len(part)} by {name}',
            out_dir / f'{stem}_{col}_table.png',
        )

    render_summary_panel(
        parts['total_ms'].head(top),
        parts['avg_us'].head(top),
        parts['max_us'].head(top),
        out_dir / f'{stem}_summary.png',
        min(top, 20),
    )

    print(f'Wrote plots and tables to {out_dir}')


if __name__ == '__main__':
    main()
