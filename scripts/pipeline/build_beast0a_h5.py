#!/usr/bin/env python3
"""beast-0a training H5: antman 1-push boards + beast root boards, tightest-label form.

Labels (per cell, on value_mask&r_mask):
  exact:   opener=1.0, verified setup=0.9           (ceiling_mask=0)
  ceiling: antman non-opener -> 0.9 ("didn't open in 1, never looked deeper")
           beast root dead   -> 0.81 ("exhaustively proven no 2-push finish")  (ceiling_mask=1)
Dedup key=(xml, object_id): beast root row supersedes the antman row (tighter); within-source keep-first.
NO post-setup boards. Output lzf-compressed, per-row chunks.
"""
import h5py, numpy as np, time, os

A = '/common/users/dm1487/scratch_namo/curriculum2/control_r5/mistakes_arm_train.h5'
B = '/common/users/dm1487/scratch_namo/curriculum2/beast/round0/h5/beast0_g0.9.h5'
OUT = '/common/users/dm1487/scratch_namo/curriculum2/beast/round0/h5/beast0a_train.h5'
COLS = ['ctx', 'contact_px', 'r_mask', 'value_target', 'value_mask', 'ceiling_mask', 'xml', 'object_id']
dec = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]
t0 = time.time()

# ---- pass 1: beast root rows (chunked sequential over the 107GB file) ----
hb = h5py.File(B, 'r'); NB = int(hb.attrs['n_samples'])
buf = {k: [] for k in ('ctx', 'contact_px', 'r_mask', 'value_target', 'value_mask', 'xml', 'object_id')}
CH = 100000
for s in range(0, NB, CH):
    e = min(s + CH, NB)
    nk = np.array(dec(hb['node_kind'][s:e]))
    rm = (nk == 'root')
    if not rm.any():
        continue
    for k in buf:
        buf[k].append(hb[k][s:e][rm])
    print(f"  beast chunk {s//CH} ({time.time()-t0:.0f}s)", flush=True)
beast = {k: np.concatenate(v) for k, v in buf.items()}
print(f"beast root rows: {len(beast['xml']):,} ({time.time()-t0:.0f}s)", flush=True)

# in-source dedup (narrow+wide double collection): keep first per (xml,object)
bk = list(zip(dec(beast['xml']), dec(beast['object_id'])))
seen, keep = set(), []
for i, k in enumerate(bk):
    if k not in seen:
        seen.add(k); keep.append(i)
keep = np.array(keep)
beast = {k: v[keep] for k, v in beast.items()}
bkeys = set(np.array(bk, dtype=object)[keep].map(tuple)) if False else {bk[i] for i in keep}
print(f"beast unique rows: {len(keep):,}", flush=True)

# relabel beast root: setups(0.9)/openers(1) exact; dead(0 on tried&reachable) -> ceiling 0.81
v, m, r = beast['value_target'], beast['value_mask'], beast['r_mask']
tried = (m == 1) & (r == 1)
ceil = ((v == 0.0) & tried).astype(np.float32)
v = v.copy(); v[ceil == 1] = 0.81
beast['value_target'] = v; beast['ceiling_mask'] = ceil

# ---- pass 2: antman rows, skipping keys beast covers + in-source dups ----
ha = h5py.File(A, 'r'); NA = int(ha.attrs.get('n_samples', ha['ctx'].shape[0]))
ax, ao = dec(ha['xml'][:]), dec(ha['object_id'][:])
rows = []
seenA = set()
for i in range(NA):
    k = (ax[i], ao[i])
    if k in bkeys or k in seenA:
        continue
    seenA.add(k); rows.append(i)
rows = np.array(rows)
print(f"antman rows kept: {len(rows):,} of {NA:,} ({time.time()-t0:.0f}s)", flush=True)

ant = {}
SL = 20000                       # sorted fancy-index in slabs (NFS-friendly)
for k in ('ctx', 'contact_px', 'r_mask', 'value_target', 'value_mask', 'xml', 'object_id'):
    parts = [ha[k][rows[s:s+SL]] for s in range(0, len(rows), SL)]
    ant[k] = np.concatenate(parts)
    print(f"  antman col {k} ({time.time()-t0:.0f}s)", flush=True)

# relabel antman: opener(1) exact; non-opener tried -> ceiling 0.9  (vt has {-1,0,1})
v, m, r = ant['value_target'], ant['value_mask'], ant['r_mask']
tried = (m == 1) & (r == 1)
ceil = ((v == 0.0) & tried).astype(np.float32)
v = v.copy(); v[ceil == 1] = 0.9
ant['value_target'] = v; ant['ceiling_mask'] = ceil

# ---- write merged ----
merged = {k: np.concatenate([beast[k], ant[k]]) for k in COLS}
N = len(merged['xml'])
str_dt = h5py.string_dtype()
with h5py.File(OUT, 'w') as o:
    for k in COLS:
        arr = merged[k]
        if arr.dtype == object or arr.dtype.kind in ('S', 'U', 'O'):
            o.create_dataset(k, data=np.array(dec(arr), dtype=object), dtype=str_dt)
        elif arr.ndim > 1:
            o.create_dataset(k, data=arr, compression='lzf', chunks=(1,) + arr.shape[1:])
        else:
            o.create_dataset(k, data=arr)
    o.attrs['n_samples'] = N
    o.attrs['gamma'] = 0.9
    o.attrs['label_scheme'] = ('beast-0a tightest-form: exact opener=1 / setup=0.9 (ceiling_mask=0); '
                               'ceiling<=0.9 (antman non-opener) / <=0.81 (beast exhausted dead) (ceiling_mask=1). '
                               'Root/start-state boards only; no post-setup rows.')
print(f"WROTE {OUT} rows={N:,} size={os.path.getsize(OUT)/1e9:.2f}GB ({time.time()-t0:.0f}s)", flush=True)
# sanity
with h5py.File(OUT, 'r') as o:
    v = o['value_target'][:5000]; c = o['ceiling_mask'][:5000]; m = (o['value_mask'][:5000] == 1) & (o['r_mask'][:5000] == 1)
    print("sample: exact1=", int(((v == 1) & m & (c == 0)).sum()), " exact.9=", int((np.isclose(v, .9) & m & (c == 0)).sum()),
          " ceil.81=", int((np.isclose(v, .81) & m & (c == 1)).sum()), " ceil.9=", int((np.isclose(v, .9) & m & (c == 1)).sum()), flush=True)
