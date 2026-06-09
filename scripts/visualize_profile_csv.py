from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def compact_shape(s: str) -> str:
    return str(s).replace(', ', ',')


def shorten(s: str, n: int = 58) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 3] + '...'


def make_label(row, i: int) -> str:
    dtype = str(row['dtype'])
    kind = str(row['kind'])
    shape = shorten(compact_shape(row['shape']), 44)
    return f'#{i:02d} {dtype} {kind} {shape}'


def plot_barh(df, value_col: str, xlabel: str, title: str, out_path: Path):
    df = df.copy()
    df['label'] = [make_label(row, i) for i, row in df.iterrows()]

    plt.figure(figsize=(18, max(7, len(df) * 0.45)))
    plt.barh(df['label'], df[value_col])
    plt.xlabel(xlabel)
    plt.ylabel('')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.42, right=0.98, top=0.92, bottom=0.08)
    plt.savefig(out_path, dpi=300)
    plt.close()


def render_table(df, title: str, out_path: Path):
    shown = df.copy()
    shown.insert(0, 'idx', [f'#{i:02d}' for i in range(len(shown))])
    shown['shape'] = shown['shape'].map(compact_shape)

    cols = ['idx', 'calls', 'dtype', 'kind', 'shape', 'total_ms', 'avg_us', 'max_us']
    shown = shown[cols]

    fig_h = max(3, 0.38 * len(shown) + 1.2)
    fig, ax = plt.subplots(figsize=(22, fig_h))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--out-dir', default='.')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[['calls', 'shape', 'kind', 'dtype', 'total_ms', 'avg_us', 'max_us']]

    stem = csv_path.stem

    views = [
        ('total_ms', 'Total time [ms]', 'total time'),
        ('avg_us', 'Average latency [µs]', 'average latency'),
        #('max_us', 'Max latency [µs]', 'max latency'),
    ]

    for col, xlabel, name in views:
        part = df.sort_values(col, ascending=False).head(args.top).reset_index(drop=True)

        plot_barh(
            part,
            col,
            xlabel,
            f'Top {len(part)} Magnetron matmul ops by {name}',
            out_dir / f'{stem}_{col}.png',
        )

        render_table(
            part,
            f'Legend/table for top {len(part)} by {name}',
            out_dir / f'{stem}_{col}_table.png',
        )

    print(f'Wrote plots and tables to {out_dir}')


if __name__ == '__main__':
    main()
