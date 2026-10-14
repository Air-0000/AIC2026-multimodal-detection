import argparse, cv2, numpy as np, glob, os

def load_depth(p):
    d = cv2.imread(p, -1)
    if d is None:
        raise FileNotFoundError(p)
    return d.astype(np.float32)

def to_color(d, cmap):
    if d.size == 0:
        return np.zeros((1, 1, 3), np.uint8)
    lo, hi = float(d.min()), float(d.max())
    if hi <= lo:
        return np.zeros_like(d, np.uint8)
    dn = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
    return cv2.applyColorMap((dn * 255).astype(np.uint8), cmap)

def main():
    ap = argparse.ArgumentParser(description="depth -> colorized for viewing only (NOT for training)")
    ap.add_argument('indir')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--compare', default=None,
                    help='original depth dir; build side-by-side orig|refined color')
    ap.add_argument('--cmap', default='JET')
    ap.add_argument('--ext', default='png')
    args = ap.parse_args()
    cmap = getattr(cv2, 'COLORMAP_' + args.cmap.upper(), cv2.COLORMAP_JET)
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.indir, '*.' + args.ext)))
    n = 0
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0].replace('_refined', '')
        d = load_depth(f)
        if args.compare:
            og = os.path.join(args.compare, base + '.png')
            if not os.path.exists(og):
                continue
            oc = to_color(load_depth(og), cmap)
            rc = to_color(d, cmap)
            h = min(oc.shape[0], rc.shape[0])
            cv2.imwrite(os.path.join(args.out_dir, base + '_cmp.png'),
                        np.hstack([oc[:h], rc[:h]]))
        else:
            cv2.imwrite(os.path.join(args.out_dir, base + '_color.png'), to_color(d, cmap))
        n += 1
    print('colorized', n, '->', args.out_dir)

if __name__ == '__main__':
    main()
