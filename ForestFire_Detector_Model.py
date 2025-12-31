import os
from datetime import datetime, timedelta
import ee, geemap
PROJECT_ID = 'earth-engine-project-470617'
try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate(); ee.Initialize(project=PROJECT_ID)
print('EE initialized with project:', PROJECT_ID)

Uttarakhand = ee.Geometry.Rectangle([77.5, 28.7, 81.5, 31.3])


START_DATE = '2016-04-09'
END_DATE   = '2016-06-15'

OUTPUT_DIR = 'daily_stacks'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPORT_DAILY_STACKS = True
print(f'Config ready: {START_DATE} -> {END_DATE}  EXPORT_DAILY_STACKS={EXPORT_DAILY_STACKS}')

BUILD_STACK_IN_RAM = False

if BUILD_STACK_IN_RAM:
    stack_list = []
    for band_file in stack_files:
        
        pass
   
    stack = np.stack(stack_list, axis=0)
else:
    print('Skipping in-RAM stack build; use the Out-of-core stack builder (memmap) cell below.')

EXPORT_SEPARATE_BANDS = True   
SEPARATE_DIR = 'daily_bands'

if EXPORT_DAILY_STACKS:
    import pathlib
    scale = 500
    crs = 'EPSG:4326'
    pathlib.Path(SEPARATE_DIR).mkdir(exist_ok=True)
    
    lulc = ee.ImageCollection('ESA/WorldCover/v100').first().select('Map').rename('LULC').clip(Uttarakhand)
    dem = ee.Image('USGS/SRTMGL1_003').clip(Uttarakhand).rename('DEM')
    terrain = ee.Terrain.products(dem)
    slope = terrain.select('slope').rename('Slope')
    aspect = terrain.select('aspect').rename('Aspect')
    hillshade = ee.Terrain.hillshade(dem).rename('Hillshade')
    
    era5 = ee.ImageCollection('ECMWF/ERA5/DAILY')
    ndvi_ic = ee.ImageCollection('MODIS/061/MOD13Q1').select('NDVI')
    burn_ic = ee.ImageCollection('MODIS/061/MCD64A1').select('BurnDate')
    start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_dt   = datetime.strptime(END_DATE, '%Y-%m-%d')
    cur = start_dt
    print('Beginning daily export loop .')
    while cur <= end_dt:
        dstr = cur.strftime('%Y-%m-%d')
        day = ee.Date(dstr)
        
        temp = era5.filterDate(day, day.advance(1,'day')).select('mean_2m_air_temperature').mean().add(-273.15).rename('TempC')
        u    = era5.filterDate(day, day.advance(1,'day')).select('u_component_of_wind_10m').mean().rename('U10')
        v    = era5.filterDate(day, day.advance(1,'day')).select('v_component_of_wind_10m').mean().rename('V10')
        wspd = u.pow(2).add(v.pow(2)).sqrt().rename('WindSpeed')
        ndvi = ndvi_ic.filterDate(day.advance(-16,'day'), day.advance(1,'day')).mean().multiply(0.0001).rename('NDVI')
        burn = burn_ic.filterDate(day.advance(-32,'day'), day.advance(1,'day')).mosaic().rename('BurnDate')
        
        daily_stack = temp.addBands([u, v, wspd, ndvi, burn, lulc, dem, slope, aspect, hillshade])
        out_stack_path = os.path.join(OUTPUT_DIR, f'Uttarakhand_stack_{dstr}.tif')

        if not os.path.exists(out_stack_path):
            print('Exporting stacked', out_stack_path)
            geemap.ee_export_image(daily_stack, filename=out_stack_path, region=Uttarakhand, scale=scale, crs=crs, file_per_band=False)
        else:
            print('Stack exists, skip', out_stack_path)
       
        if EXPORT_SEPARATE_BANDS:
            band_dir = os.path.join(SEPARATE_DIR, dstr)
            pathlib.Path(band_dir).mkdir(parents=True, exist_ok=True)
            band_names = ['TempC','U10','V10','WindSpeed','NDVI','BurnDate','LULC','DEM','Slope','Aspect','Hillshade']
            for idx, bname in enumerate(band_names):
                band_file = os.path.join(band_dir, f'{bname}.tif')
                if os.path.exists(band_file):
                    continue
                
                single = daily_stack.select(idx)
                try:
                    geemap.ee_export_image(single, filename=band_file, region=Uttarakhand, scale=scale, crs=crs, file_per_band=False)
                except Exception as e:
                    print('Band export failed', bname, e)
        cur += timedelta(days=1)
    print('Daily exports complete.')
else:
    print('Skipping export (EXPORT_DAILY_STACKS False).')
import glob, rasterio, numpy as np, os
STACK_PATTERN = os.path.join(OUTPUT_DIR, 'Uttarakhand_stack_*.tif')
stack_files = sorted(glob.glob(STACK_PATTERN))
stack_list = []
dates = []
if stack_files:
    print(f'Found {len(stack_files)} stacked daily files.')
    for f in stack_files:
        dates.append(f.split('_')[-1].replace('.tif',''))
        with rasterio.open(f) as src:
            arr = src.read()  
        if arr.shape[0] != 11:
            raise RuntimeError('Unexpected band count in ' + f)
        stack_list.append(arr.astype('float32'))
else:
   
    BAND_DIR_ROOT = 'daily_bands'
    band_names = ['TempC','U10','V10','WindSpeed','NDVI','BurnDate','LULC','DEM','Slope','Aspect','Hillshade']
    date_dirs = sorted([d for d in os.listdir(BAND_DIR_ROOT) if os.path.isdir(os.path.join(BAND_DIR_ROOT,d))])
    if not date_dirs:
        raise SystemExit('No stacked files or per-band directories found. Run Cell 2.')
    print('Reconstructing stack from per-band directories...')
    for d in date_dirs:
        band_arrays = []
        valid = True
        first_shape = None
        for b in band_names:
            path = os.path.join(BAND_DIR_ROOT, d, f'{b}.tif')
            if not os.path.exists(path):
                print('Missing band', b, 'for date', d, 'skipping date.')
                valid = False; break
            with rasterio.open(path) as src:
                arr = src.read(1) 
            if first_shape is None:
                first_shape = arr.shape
            else:
                if arr.shape != first_shape:
                    print('Shape mismatch for', d, b, 'skipping date.')
                    valid = False; break
            band_arrays.append(arr.astype('float32'))
        if not valid:
            continue
        stack_list.append(np.stack(band_arrays, axis=0))
        dates.append(d)
    if not stack_list:
        raise SystemExit('No valid reconstructed days.')

stack = np.stack(stack_list, axis=0)  
T,B,H,W = stack.shape
print('Loaded data shape:', stack.shape, '| Days:', len(dates))

B_TEMP,B_U,B_V,B_WS,B_NDVI,B_BURN = 0,1,2,3,4,5
B_LULC,B_DEM,B_SLOPE,B_ASPECT,B_HILL = 6,7,8,9,10
means_day0 = stack[0].reshape(11,-1).mean(axis=1)
print('Day0 band means:', means_day0)

WRITE_RECONSTRUCTED_STACKS = True  
STACK_DTYPE = 'float32'
if WRITE_RECONSTRUCTED_STACKS and not stack_files:
    import rasterio
    from rasterio.transform import from_bounds
   
    sample_day_dir = os.path.join('daily_bands', dates[0])
    sample_band_path = os.path.join(sample_day_dir, 'TempC.tif')
    with rasterio.open(sample_band_path) as ssrc:
        profile = ssrc.profile.copy()
        transform = ssrc.transform
        crs = ssrc.crs
    profile.update(count=11, dtype=STACK_DTYPE)
    print('Writing stacked daily GeoTIFFs to', OUTPUT_DIR)
    band_names_order = ['TempC','U10','V10','WindSpeed','NDVI','BurnDate','LULC','DEM','Slope','Aspect','Hillshade']
    for arr, dstr in zip(stack_list, dates):
        out_path = os.path.join(OUTPUT_DIR, f'pauri_stack_{dstr}.tif')
        if os.path.exists(out_path):
            continue
        with rasterio.open(out_path, 'w', **profile) as dst:
            for i in range(11):
                dst.write(arr[i].astype(STACK_DTYPE), i+1)
            dst.update_tags(**{f'B{i+1}': band_names_order[i] for i in range(11)})
        print('Wrote', out_path)
    print('Local stacking complete.')
elif WRITE_RECONSTRUCTED_STACKS and stack_files:
    print('Stacked files already exist; skipping local write.')
else:
    print('Skipped writing reconstructed stacks.')

SHOW_LAST = True     
SAVE_FIGS = False     
FIG_DIR = 'figures'
KEY_BANDS = [
    ('TempC', B_TEMP, '°C air temperature (higher = hotter fuel/air)', 'inferno'),
    ('WindSpeed', B_WS, 'm/s wind speed (higher = more spread potential)', 'plasma'),
    ('NDVI', B_NDVI, 'NDVI vegetation index (-1..1)', 'Greens'),
    ('Burn mask', B_BURN, 'Already burned (red = burned)', 'Reds'),
    ('DEM', B_DEM, 'Elevation (m)', 'terrain'),
    ('Slope', B_SLOPE, 'Slope (degrees)', 'viridis')
]
import numpy as np, os
import matplotlib.pyplot as plt
if SAVE_FIGS:
    os.makedirs(FIG_DIR, exist_ok=True)
sel_days = [0] + ([T-1] if SHOW_LAST and T>1 else [])
for di in sel_days:
    fig, axes = plt.subplots(2,3, figsize=(11,6))
    fig.suptitle(f'Day {di+1}/{T} ({dates[di] if di < len(dates) else di})')
    for ax, (name, idx, desc, cmap) in zip(axes.ravel(), KEY_BANDS):
        data = stack[di, idx]
        if name == 'Burn mask':
            data_plot = (data>0).astype(float)
            vmin, vmax = 0,1
        else:
            data_plot = data
            vmin, vmax = np.nanpercentile(data_plot,2), np.nanpercentile(data_plot,98)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin==vmax:
                vmin, vmax = np.nanmin(data_plot), np.nanmax(data_plot)
        im = ax.imshow(data_plot, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'{name}', fontsize=10)
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(desc, fontsize=7)
    expl = (
        'Interpretation: Higher TempC & WindSpeed usually increase fire spread risk. NDVI: 0.2-0.5 moderate veg; >0.6 dense fuels; near 0 sparse/barren. '
        'Burn mask shows existing burned areas (cannot ignite again immediately). Terrain (DEM, Slope): fires move faster upslope; steep slopes raise intensity.'
    )
    fig.text(0.5, 0.005, expl, ha='center', va='bottom', fontsize=8, wrap=True)
    plt.tight_layout(rect=(0,0.03,1,0.97))
    if SAVE_FIGS:
        out = os.path.join(FIG_DIR, f'key_bands_day{di+1}.png')
        plt.savefig(out, dpi=150)
        print('Saved figure', out)
    plt.show()


import numpy as np
import os
from scipy.ndimage import binary_dilation

# Feature engineering with on-disk memmaps (prevents OOM)
LAG_DAYS = 3                 # number of lagged dynamic days
INCLUDE_DIFFS = True         # include per-lag diffs for dynamics
DILATE_LABELS = True         # optional label dilation
DILATION_RADIUS = 1          # not used directly (3x3 struct now)
MIN_TRAIN_POS_IGNITION = 50  # ignition model off if too few positives

DYN_BASE_IDXS = [B_TEMP, B_U, B_V, B_WS, B_NDVI]
STATIC_IDXS = [B_LULC, B_DEM, B_SLOPE, B_ASPECT, B_HILL]

if LAG_DAYS < 1:
    raise ValueError('LAG_DAYS must be >=1')

structure = np.ones((3,3), dtype=bool) if DILATE_LABELS else None

# Determine number of usable transitions and feature dimension F using first valid t
first_t = LAG_DAYS - 1
last_t = T - 2  # we predict t+1, so last input t is T-2
usable_transitions = max(0, last_t - first_t + 1)
if usable_transitions <= 0:
    raise SystemExit('Not enough days to build transitions. Check LAG_DAYS and T.')

# Build once to discover F
_dyn_lags = [stack[first_t-li, DYN_BASE_IDXS] for li in range(LAG_DAYS)]
_dyn_lags = _dyn_lags[::-1]
_dyn_lags_arr = np.stack(_dyn_lags, axis=0)
_diff_feats_arr = None
if INCLUDE_DIFFS and LAG_DAYS > 1:
    dfs = [_dyn_lags_arr[k]-_dyn_lags_arr[k-1] for k in range(1, LAG_DAYS)]
    _diff_feats_arr = np.stack(dfs, axis=0).reshape(-1, _dyn_lags_arr.shape[-2], _dyn_lags_arr.shape[-1])
_burn_now = (stack[first_t, B_BURN] > 0).astype('uint8')
_static_feats = stack[first_t, STATIC_IDXS]
_dyn_flat = _dyn_lags_arr.reshape(LAG_DAYS*len(DYN_BASE_IDXS), H, W)
_parts = [_dyn_flat]
if _diff_feats_arr is not None and _diff_feats_arr.size:
    _parts.append(_diff_feats_arr)
_parts.append(_burn_now[None, ...])
_parts.append(_static_feats)
_feats_full = np.concatenate(_parts, axis=0)
F = int(_feats_full.shape[0])
del _dyn_lags, _dyn_lags_arr, _diff_feats_arr, _burn_now, _static_feats, _dyn_flat, _parts, _feats_full

pts_per_transition = H * W
N_total = usable_transitions * pts_per_transition
print(f'Preparing memmaps for features: N={N_total} F={F} (transitions={usable_transitions}, pts/transition={pts_per_transition})')

# Prepare memmaps on disk (absolute paths; robust to locked files on Windows)
MMAP_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, 'memmap'))
os.makedirs(MMAP_DIR, exist_ok=True)

X_MEMMAP_PATH = os.path.abspath(os.path.join(MMAP_DIR, 'X_anyburn.dat'))
Y_ANY_MEMMAP_PATH = os.path.abspath(os.path.join(MMAP_DIR, 'y_any.dat'))
Y_IGN_MEMMAP_PATH = os.path.abspath(os.path.join(MMAP_DIR, 'y_ign.dat'))

def _create_memmap_safe(path, dtype, shape):
    try:
        return np.memmap(path, dtype=dtype, mode='w+', shape=shape)
    except OSError as e:
        base, ext = os.path.splitext(path)
        alt = f"{base}_{np.random.randint(1_000_000_000)}{ext}"
        print(f'Memmap path busy or invalid, using fallback: {alt}')
        return np.memmap(alt, dtype=dtype, mode='w+', shape=shape)

X = _create_memmap_safe(X_MEMMAP_PATH, dtype='float32', shape=(N_total, F))
y_any = _create_memmap_safe(Y_ANY_MEMMAP_PATH, dtype='uint8', shape=(N_total,))
y_ign = _create_memmap_safe(Y_IGN_MEMMAP_PATH, dtype='uint8', shape=(N_total,))

any_counts = []  # positives per transition
write_ptr = 0

for t in range(first_t, last_t + 1):
    # Build features for day t
    dyn_lags = [stack[t-li, DYN_BASE_IDXS] for li in range(LAG_DAYS)]
    dyn_lags = dyn_lags[::-1]
    dyn_lags_arr = np.stack(dyn_lags, axis=0)

    diff_feats_arr = None
    if INCLUDE_DIFFS and LAG_DAYS > 1:
        dfs = [dyn_lags_arr[k]-dyn_lags_arr[k-1] for k in range(1, LAG_DAYS)]
        diff_feats_arr = np.stack(dfs, axis=0).reshape(-1, H, W)

    burn_now = (stack[t, B_BURN] > 0).astype('uint8')
    static_feats = stack[t, STATIC_IDXS]

    dyn_flat = dyn_lags_arr.reshape(LAG_DAYS*len(DYN_BASE_IDXS), H, W)
    parts = [dyn_flat]
    if diff_feats_arr is not None and diff_feats_arr.size:
        parts.append(diff_feats_arr)
    parts.append(burn_now[None, ...])
    parts.append(static_feats)

    feats_full = np.concatenate(parts, axis=0)
    F_chk = feats_full.shape[0]
    if F_chk != F:
        raise SystemExit(f'Feature dimension changed at t={t}: {F_chk} vs expected {F}')

    # Labels for t+1
    burn_next = (stack[t+1, B_BURN] > 0).astype('uint8')
    any_burn = burn_next.copy()
    if DILATE_LABELS and any_burn.any():
        any_burn = binary_dilation(any_burn, structure=structure).astype('uint8')
    new_ignition = ((burn_now == 0) & (burn_next == 1)).astype('uint8')

    # Flatten chunk and write to memmaps
    X_chunk = feats_full.reshape(F, -1).T.astype('float32', copy=False)
    y_any_chunk = any_burn.reshape(-1)
    y_ign_chunk = new_ignition.reshape(-1)

    end_ptr = write_ptr + X_chunk.shape[0]
    X[write_ptr:end_ptr] = X_chunk
    y_any[write_ptr:end_ptr] = y_any_chunk
    y_ign[write_ptr:end_ptr] = y_ign_chunk

    any_counts.append(int(y_any_chunk.sum()))
    write_ptr = end_ptr


X.flush(); y_any.flush(); y_ign.flush()

print('Feature memmaps ready:', X.shape, y_any.shape, y_ign.shape)
print('ANY_BURN positives total:', int(y_any.sum()), 'ratio:', float(y_any.mean()))
print('NEW_IGN  positives total:', int(y_ign.sum()), 'ratio:', float(y_ign.mean()))

initial_train_trans = int(0.8 * usable_transitions)
train_trans = initial_train_trans
while train_trans < usable_transitions - 1 and sum(any_counts[train_trans:]) == 0:
    train_trans -= 1
if train_trans <= 0:
    train_trans = initial_train_trans

train_pts = train_trans * pts_per_transition


X_train = X[:train_pts]; X_test = X[train_pts:]
y_any_train = y_any[:train_pts]; y_any_test = y_any[train_pts:]
y_ign_train = y_ign[:train_pts]; y_ign_test = y_ign[train_pts:]

print(f'Train transitions: {train_trans}  Test transitions: {usable_transitions-train_trans}')
print('ANY_BURN train pos:', int(y_any_train.sum()), 'test pos:', int(y_any_test.sum()))
print('IGNITION train pos:', int(y_ign_train.sum()), 'test pos:', int(y_ign_test.sum()))

TRAIN_IGNITION_MODEL = int(y_ign_train.sum()) >= MIN_TRAIN_POS_IGNITION
if not TRAIN_IGNITION_MODEL:
    print(f'Skip ignition model (<{MIN_TRAIN_POS_IGNITION} train positives).')


feature_rows = [None] * usable_transitions  
any_counts = any_counts 
anyburn_rows = None  


import numpy as np, time
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

MAX_TRAIN_SAMPLES = int(8e5)   
NEG_PER_POS      = 4          
RANDOM_STATE     = 42
MAX_TEST_EVAL    = 200_000    
BATCH_PRED       = 200_000     

RF_PARAMS  = dict(
    n_estimators=250,        
    max_depth=20,              
    min_samples_leaf=2,      
    max_features='sqrt',
    n_jobs=-1,
    class_weight='balanced_subsample',
    max_samples=0.5,          
    random_state=RANDOM_STATE,
)
HGB_PARAMS = dict(max_depth=None, learning_rate=0.1, max_iter=400, random_state=RANDOM_STATE)
FAST_MODEL = 'RF'

def downsample(Xd, yd, max_neg_per_pos):
    pos_idx = np.where(yd == 1)[0]
    neg_idx = np.where(yd == 0)[0]
    if pos_idx.size == 0:
        return Xd[:0], yd[:0]
    rng = np.random.RandomState(RANDOM_STATE)
    target_neg = min(len(neg_idx), max_neg_per_pos * len(pos_idx))
    neg_sel = rng.choice(neg_idx, size=target_neg, replace=False)
    keep = np.concatenate([pos_idx, neg_sel])

    
    if keep.size > MAX_TRAIN_SAMPLES:
        keep = rng.choice(keep, size=MAX_TRAIN_SAMPLES, replace=False)

    keep_sorted = np.sort(keep)
    X_small = np.asarray(Xd[keep_sorted])
    y_small = np.asarray(yd[keep_sorted])
    perm = rng.permutation(X_small.shape[0])
    return X_small[perm], y_small[perm]

def fit_eval(name, Xtr, ytr, Xte, yte):
    clf = HistGradientBoostingClassifier(**HGB_PARAMS) if FAST_MODEL.upper()=='HGB' else RandomForestClassifier(**RF_PARAMS)
    Xtr_ds, ytr_ds = downsample(Xtr, ytr, NEG_PER_POS)
    print(f'[{name}] original:{Xtr.shape} pos={int(ytr.sum())} ({float(ytr.mean()):.5f})')
    print(f'[{name}] sampled :{Xtr_ds.shape} pos={int(ytr_ds.sum())} ({float(ytr_ds.mean()):.3f})')
    t0 = time.time()
    clf.fit(Xtr_ds, ytr_ds)
    print(f'[{name}] fit time: {time.time()-t0:.1f}s')

    rng = np.random.RandomState(RANDOM_STATE)
    pos = np.where(yte==1)[0]
    neg = np.where(yte==0)[0]
    n_pos_eval = min(len(pos), MAX_TEST_EVAL//2)
    n_neg_eval = min(len(neg), MAX_TEST_EVAL - n_pos_eval, max(1000, 3*n_pos_eval))
    sel_pos = pos[:n_pos_eval]
    sel_neg = rng.choice(neg, size=n_neg_eval, replace=False) if n_neg_eval>0 else np.array([], dtype=int)
    idx_eval = np.unique(np.sort(np.concatenate([sel_pos, sel_neg]))) if (n_pos_eval+n_neg_eval)>0 else np.array([], dtype=int)

    Xev = Xte[idx_eval]
    yev = yte[idx_eval]

    def proba_for_one_local(model, Xc):
        out = np.empty(Xc.shape[0], dtype=np.float32)
        for s in range(0, Xc.shape[0], BATCH_PRED):
            e = min(s+BATCH_PRED, Xc.shape[0])
            if hasattr(model, 'predict_proba'):
                pp = model.predict_proba(Xc[s:e])
                if hasattr(model, 'classes_') and 1 in list(model.classes_):
                    idx1 = int(np.where(model.classes_ == 1)[0][0])
                    out[s:e] = pp[:, idx1]
                else:
                    out[s:e] = 0.0
            elif hasattr(model, 'decision_function'):
                from sklearn.preprocessing import MinMaxScaler
                z = model.decision_function(Xc[s:e]).reshape(-1,1)
                out[s:e] = MinMaxScaler().fit_transform(z).ravel()
            else:
                out[s:e] = 0.0
        return out

    prob = proba_for_one_local(clf, Xev).astype('float32')
    from sklearn.metrics import f1_score
    ths = np.linspace(0.1, 0.9, 17)
    f1s = [(th, f1_score(yev, (prob>=th).astype('uint8'), zero_division=0)) for th in ths]
    thr = max(f1s, key=lambda x:x[1])[0]
    roc = roc_auc_score(yev, prob) if len(np.unique(yev))>1 else float('nan')
    pr  = average_precision_score(yev, prob) if len(np.unique(yev))>1 else float('nan')
    return clf, float(thr), dict(roc_auc=roc, pr_auc=pr, f1=max(f1s,key=lambda x:x[1])[1])

MODEL_ANY, THR_ANY, METRICS_ANY = fit_eval('ANY_BURN', X_train, y_any_train, X_test, y_any_test)
if TRAIN_IGNITION_MODEL:
    MODEL_IGN, THR_IGN, METRICS_IGN = fit_eval('IGNITION', X_train, y_ign_train, X_test, y_ign_test)


# ...existing code...
# Balanced sampling without large concatenations
import time
if 'MODEL_ANY' in globals() and MODEL_ANY is not None:
    print('MODEL_ANY already trained in previous cell; skipping duplicate training.')
else:
    rng = np.random.RandomState(RANDOM_STATE)

    # Positive and negative indices from memmaps (train view)
    pos_idx = np.where(y_any_train == 1)[0]
    neg_idx = np.where(y_any_train == 0)[0]

    n_pos = len(pos_idx)
    n_neg = min(len(neg_idx), NEG_PER_POS * n_pos)

    if n_pos == 0:
        raise RuntimeError('No positive samples in y_any_train; cannot train ANY_BURN model. Adjust split or sampling.')

    neg_sel = rng.choice(neg_idx, size=n_neg, replace=False)
    keep = np.concatenate([pos_idx, neg_sel])

    if keep.size > MAX_TRAIN_SAMPLES:
        keep = rng.choice(keep, size=MAX_TRAIN_SAMPLES, replace=False)

    keep_sorted = np.sort(keep)
    X_tr_bal = np.asarray(X_train[keep_sorted])
    y_tr_bal = np.asarray(y_any_train[keep_sorted])
    perm = rng.permutation(X_tr_bal.shape[0])
    X_tr_bal = X_tr_bal[perm]
    y_tr_bal = y_tr_bal[perm]

    MODEL_ANY = RandomForestClassifier(**RF_PARAMS)
    t0 = time.time()
    MODEL_ANY.fit(X_tr_bal, y_tr_bal)
    print(f'[ANY_BURN] fit time: {time.time()-t0:.1f}s')

    te_pos = np.where(y_any_test == 1)[0]
    te_neg = np.where(y_any_test == 0)[0]
    te_pos = te_pos[:MAX_TEST_EVAL//2]
    te_neg = te_neg[:min(len(te_neg), MAX_TEST_EVAL - len(te_pos), 3*len(te_pos)+1000)]
    sel_te = np.concatenate([te_pos, te_neg])
    sel_te.sort()
    X_tune = X_test[sel_te]
    y_tune = y_any_test[sel_te]

    def proba_for_one(model, Xc, batch=BATCH_PRED):
        out = np.empty(Xc.shape[0], dtype='float32')
        for s in range(0, Xc.shape[0], batch):
            e = min(s+batch, Xc.shape[0])
            pp = model.predict_proba(Xc[s:e])
            if hasattr(model, 'classes_') and 1 in list(model.classes_):
                idx1 = int(np.where(model.classes_ == 1)[0][0])
                out[s:e] = pp[:, idx1]
            else:
                out[s:e] = 0.0
        return out

    prob_tune = proba_for_one(MODEL_ANY, X_tune).astype('float32')
    from sklearn.metrics import f1_score
    ths = np.linspace(0.1, 0.9, 17)
    f1s = []
    for th in ths:
        pred = (prob_tune >= th).astype('uint8')
        f1s.append(f1_score(y_tune, pred, zero_division=0))
    THR_ANY = float(ths[int(np.argmax(f1s))])

    import joblib, os
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': MODEL_ANY, 'thr': THR_ANY, 'train_transitions': int(train_trans)}, 'models/anyburn_rf.joblib')
    print('Saved -> models/anyburn_rf.joblib (thr=', THR_ANY, ')')
  
import matplotlib.pyplot as plt
from imageio.v2 import imread, mimsave
import os
import numpy as np
import joblib


def proba_for_one(model, Xc):
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(Xc)
        if p.ndim == 2:
            if p.shape[1] == 2:
                try:
                    idx1 = int(np.where(model.classes_ == 1)[0][0])
                except Exception:
                    idx1 = 1
                return p[:, idx1]
            if p.shape[1] == 1:
                only = int(getattr(model, 'classes_', np.array([1]))[0])
                return np.full(Xc.shape[0], 1.0 if only == 1 else 0.0, dtype='float32')
        return p.ravel().astype('float32', copy=False)
    if hasattr(model, 'decision_function'):
        df = model.decision_function(Xc).astype('float32', copy=False)
        mn, mx = float(df.min()), float(df.max())
        return (df - mn) / (mx - mn + 1e-9)
    pred = model.predict(Xc).astype('float32', copy=False)
    return pred

PRED_TARGET = 'ANY_BURN'  
HORIZON = 1


g_model, g_thr = None, None
if PRED_TARGET.upper()=='ANY_BURN':
    if 'MODEL_ANY' in globals() and MODEL_ANY is not None:
        g_model, g_thr = MODEL_ANY, THR_ANY if 'THR_ANY' in globals() else 0.5
    else:
        if os.path.exists('models/anyburn_rf.joblib'):
            bundle = joblib.load('models/anyburn_rf.joblib')
            g_model = bundle['model']
            g_thr = float(bundle.get('thr', 0.5))
            
            if 'train_trans' not in globals():
                train_trans = int(bundle.get('train_transitions', 0))
            print('Loaded model from models/anyburn_rf.joblib')
        else:
            raise SystemExit('No in-memory model and no saved model bundle found.')
elif PRED_TARGET.upper()=='IGNITION' and 'MODEL_IGN' in globals() and MODEL_IGN is not None:
    g_model, g_thr = MODEL_IGN, THR_IGN if 'THR_IGN' in globals() else 0.5
else:
    raise SystemExit('Invalid PRED_TARGET or model not available.')


first_test_t = (LAG_DAYS - 1) + train_trans
last_input_t = T - 1 - HORIZON
start_t = min(max(first_test_t, LAG_DAYS-1), last_input_t)
end_t = last_input_t + 1 
print(f'GIF over test inputs t in [{start_t}, {last_input_t}] (inclusive)')

os.makedirs(FIG_DIR, exist_ok=True)
gif_tmp = os.path.join(FIG_DIR, 'gif_tmp')
os.makedirs(gif_tmp, exist_ok=True)
frames = []

first_frame_png = os.path.join(FIG_DIR, 'first_frame_debug.png')
first_saved = False

for t in range(start_t, end_t):

    dyn_lags = []
    for lag in range(LAG_DAYS):
        dyn_lags.append(stack[t-lag, DYN_BASE_IDXS])
    dyn_lags_arr = np.stack(dyn_lags, axis=0)
    dfs = [dyn_lags_arr[li]-dyn_lags_arr[li-1] for li in range(1, LAG_DAYS)]
    diff_feats_arr = np.stack(dfs, axis=0).reshape(-1, H, W) if dfs else []
    burn_now = (stack[t, B_BURN] > 0).astype('uint8')
    static_feats = stack[t, STATIC_IDXS]
    dyn_flat = dyn_lags_arr.reshape(LAG_DAYS*len(DYN_BASE_IDXS), H, W)
    parts = [dyn_flat]
    if len(diff_feats_arr): parts.append(diff_feats_arr)
    parts.append(burn_now[None, ...])
    parts.append(static_feats)
    feats_full = np.concatenate(parts, axis=0)
    X_t = feats_full.reshape(feats_full.shape[0], -1).T
    prob = proba_for_one(g_model, X_t.astype('float32', copy=False))
   
    prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)
    risk = prob.reshape(H, W)

    base = stack[t, B_NDVI]
    
    try:
        vmin_base, vmax_base = np.nanpercentile(base,5), np.nanpercentile(base,95)
    except Exception:
        vmin_base, vmax_base = float(np.nanmin(base)), float(np.nanmax(base))
    if not np.isfinite(vmin_base) or not np.isfinite(vmax_base) or vmin_base == vmax_base:
        vmin_base, vmax_base = float(np.nanmin(base)), float(np.nanmax(base))
        if not np.isfinite(vmin_base) or not np.isfinite(vmax_base) or vmin_base == vmax_base:
            vmin_base, vmax_base = 0.0, 1.0

    if t == start_t:
        above_thr = int((prob >= float(g_thr) if g_thr is not None else prob >= 0.5).sum())
        p5, p95 = np.nanpercentile(prob, 5), np.nanpercentile(prob, 95)
        print(f'[Diag t={t}] risk min={float(risk.min()):.3f} max={float(risk.max()):.3f} mean={float(risk.mean()):.3f}  >=thr count={above_thr}  p5={p5:.3f} p95={p95:.3f}')

    try:
        r5, r95 = np.nanpercentile(risk, 5), np.nanpercentile(risk, 95)
    except Exception:
        r5, r95 = 0.0, 1.0
    if not np.isfinite(r5) or not np.isfinite(r95) or r5 == r95:
        r5, r95 = 0.0, 1.0
    risk_stretch = np.clip((risk - r5) / (r95 - r5 + 1e-9), 0.0, 1.0)

    thr_plot = float(g_thr) if g_thr is not None else 0.5
    pred_mask = (risk >= thr_plot).astype(float)
    pred_count = int(pred_mask.sum())

    fig = plt.figure(figsize=(14,7))

    ax1 = plt.subplot(2,2,1)
    im1 = ax1.imshow(base, cmap='Greens', vmin=vmin_base, vmax=vmax_base)
    ax1.set_title(f'NDVI t={t}')
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.ax.tick_params(labelsize=7)
    cbar1.set_label('NDVI', fontsize=8)

    ax2 = plt.subplot(2,2,2)
    im2 = ax2.imshow(risk, cmap='inferno', vmin=0, vmax=1)
    ax2.set_title('Risk (probability 0–1)')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.ax.tick_params(labelsize=7)
    cbar2.set_label('Risk (0–1)', fontsize=8)

    ax3 = plt.subplot(2,2,3)
    im3 = ax3.imshow(risk_stretch, cmap='inferno', vmin=0, vmax=1)
    ax3.set_title('Risk (stretched 5–95%)')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)
    cbar3.ax.tick_params(labelsize=7)
    cbar3.set_label('Stretched risk', fontsize=8)

    ax4 = plt.subplot(2,2,4)
    ax4.imshow(base, cmap='Greens', vmin=vmin_base, vmax=vmax_base)
    ax4.imshow(np.ma.masked_where(pred_mask==0, pred_mask), cmap='autumn', alpha=0.6)
    ax4.set_title(f'Predicted mask (thr={thr_plot:.2f}, count={pred_count})')
    ax4.axis('off')

    out_png = os.path.join(gif_tmp, f'frame_{t:04d}.png')
    plt.tight_layout(); plt.savefig(out_png, dpi=110)

    if not first_saved:
        plt.savefig(first_frame_png, dpi=120)
        print('Saved first frame preview ->', first_frame_png)
        first_saved = True

    plt.close(fig)
    frames.append(imread(out_png))

GIF_PATH = os.path.join(FIG_DIR, f'test_risk_h{HORIZON}_{PRED_TARGET.lower()}.gif')
if len(frames) == 0:
    print('No frames to write — check start/end indices (start_t > last_input_t).')
else:
    mimsave(GIF_PATH, frames, fps=3)
    print('Saved GIF ->', GIF_PATH, '| frames:', len(frames))

import os, re, json, pathlib, numpy as np, rasterio
from glob import glob

if 'stack_files' not in globals() or not stack_files:
    pattern = STACK_PATTERN if 'STACK_PATTERN' in globals() else os.path.join(OUTPUT_DIR, 'Uttarakhand_stack_*.tif')
    stack_files = sorted(glob(pattern))
    if not stack_files:
        raise SystemExit('No stacks found — run earlier export/discovery cells first.')

T_guess = len(stack_files)
if T_guess == 0:
    raise SystemExit('No stack files available.')


with rasterio.open(stack_files[0]) as src0:
    B, H, W = src0.count, src0.height, src0.width
print('Detected shape: T=?, B,H,W=', B, H, W)


MMAP_DIR = os.path.join(OUTPUT_DIR, 'memmap')
pathlib.Path(MMAP_DIR).mkdir(parents=True, exist_ok=True)
STACK_MMAP_PATH = os.path.join(MMAP_DIR, 'stack_mmap.dat')
print('Creating memmap at:', STACK_MMAP_PATH)
mm = np.memmap(STACK_MMAP_PATH, dtype='float32', mode='w+', shape=(T_guess, B, H, W))


for i, p in enumerate(stack_files):
    with rasterio.open(p) as src:
        arr = src.read().astype('float32', copy=False)
        if arr.shape != (B, H, W):
            raise SystemExit(f'Shape mismatch at {p}: {arr.shape} != {(B,H,W)}')
        mm[i] = arr
        if (i+1) % 5 == 0 or i == T_guess-1:
            print(f'Wrote {i+1}/{T_guess} days')

del mm
stack = np.memmap(STACK_MMAP_PATH, dtype='float32', mode='r+', shape=(T_guess, B, H, W))
T = T_guess
print('Memmap stack ready with shape:', tuple(stack.shape))

if 'dates' not in globals() or not dates:
    date_re = re.compile(r'(\d{4}-\d{2}-\d{2})')
    dates = []
    for p in stack_files:
        m = date_re.search(os.path.basename(p))
        dates.append(m.group(1) if m else os.path.basename(p))
    print('Derived dates count:', len(dates))


rng = np.random.RandomState(RANDOM_STATE)


pos_idx = np.where(y_any_train == 1)[0]
neg_idx = np.where(y_any_train == 0)[0]

n_pos = len(pos_idx)
n_neg = min(len(neg_idx), NEG_PER_POS * n_pos)

if n_pos == 0:
    raise RuntimeError('No positive samples in y_any_train; cannot train ANY_BURN model. Adjust split or sampling.')

neg_sel = rng.choice(neg_idx, size=n_neg, replace=False)
keep = np.concatenate([pos_idx, neg_sel])
rng.shuffle(keep)

if keep.size > MAX_TRAIN_SAMPLES:
    keep = keep[:MAX_TRAIN_SAMPLES]

X_tr_bal = X_train[keep]
y_tr_bal = y_any_train[keep]


MODEL_ANY = RandomForestClassifier(**RF_PARAMS)
MODEL_ANY.fit(X_tr_bal, y_tr_bal)


te_pos = np.where(y_any_test == 1)[0]
te_neg = np.where(y_any_test == 0)[0]
te_neg = rng.choice(te_neg, size=min(3*len(te_pos)+1000, len(te_neg)), replace=False) if len(te_pos) > 0 else te_neg[:5000]
sel_te = np.concatenate([te_pos, te_neg])
rng.shuffle(sel_te)
X_tune = X_test[sel_te]
y_tune = y_any_test[sel_te]

def proba_for_one(model, Xc):
    if hasattr(model, 'predict_proba'):
        pp = model.predict_proba(Xc)
        if hasattr(model, 'classes_') and 1 in list(model.classes_):
            idx1 = int(np.where(model.classes_ == 1)[0][0])
            return pp[:, idx1]
        else:
            return np.zeros(Xc.shape[0], dtype=np.float32)
    elif hasattr(model, 'decision_function'):
        from sklearn.preprocessing import MinMaxScaler
        z = model.decision_function(Xc).reshape(-1,1)
        return MinMaxScaler().fit_transform(z).ravel()
    else:
        return np.zeros(Xc.shape[0], dtype=np.float32)

prob_tune = proba_for_one(MODEL_ANY, X_tune).astype('float32')
from sklearn.metrics import f1_score
ths = np.linspace(0.1, 0.9, 17)
f1s = []
for th in ths:
    pred = (prob_tune >= th).astype('uint8')
    f1s.append(f1_score(y_tune, pred, zero_division=0))
THR_ANY = float(ths[int(np.argmax(f1s))])


import joblib, os
os.makedirs('models', exist_ok=True)
joblib.dump({'model': MODEL_ANY, 'thr': THR_ANY, 'train_transitions': int(train_trans)}, 'models/anyburn_rf.joblib')
print('Saved -> models/anyburn_rf.joblib (thr=', THR_ANY, ')')


# Inference and GIF (test period only) — ANY_BURN, batched
import os, numpy as np, joblib, matplotlib.pyplot as plt
from imageio.v2 import imread, mimsave  # use PIL to read PNGs as RGB
from PIL import Image

# Safe probability for class 1 even if model trained on a single class
def proba_for_one(model, Xc):
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(Xc)
        if p.ndim == 2:
            if p.shape[1] == 2:
                try:
                    idx1 = int(np.where(model.classes_ == 1)[0][0])
                except Exception:
                    idx1 = 1
                return p[:, idx1]
            if p.shape[1] == 1:
                only = int(getattr(model, 'classes_', np.array([1]))[0])
                return np.full(Xc.shape[0], 1.0 if only == 1 else 0.0, dtype='float32')
        return p.ravel().astype('float32', copy=False)
    if hasattr(model, 'decision_function'):
        df = model.decision_function(Xc).astype('float32', copy=False)
        mn, mx = float(df.min()), float(df.max())
        return (df - mn) / (mx - mn + 1e-9)
    pred = model.predict(Xc).astype('float32', copy=False)
    return pred

bundle = joblib.load('models/anyburn_rf.joblib')
MODEL_ANY = bundle['model']
THR_ANY = float(bundle['thr'])
train_trans = int(bundle.get('train_transitions', 0))

HORIZON = 1
first_test_t = (LAG_DAYS - 1) + train_trans
last_input_t = T - 1 - HORIZON
start_t = min(max(first_test_t, LAG_DAYS-1), last_input_t)
end_t = last_input_t + 1  
print(f'Generating test GIF for t in [{start_t}, {last_input_t}] (inclusive)')

os.makedirs(FIG_DIR, exist_ok=True)
gif_tmp = os.path.join(FIG_DIR, 'gif_tmp'); os.makedirs(gif_tmp, exist_ok=True)
frames = []
BATCH = 200_000

for t in range(start_t, end_t):
    dyn_lags = [stack[t-li, DYN_BASE_IDXS] for li in range(LAG_DAYS)]
    dyn_lags_arr = np.stack(dyn_lags, axis=0)
    dfs = [dyn_lags_arr[li]-dyn_lags_arr[li-1] for li in range(1, LAG_DAYS)]
    diff_feats_arr = np.stack(dfs, axis=0).reshape(-1, H, W) if dfs else []
    burn_now = (stack[t, B_BURN] > 0).astype('uint8')
    static_feats = stack[t, STATIC_IDXS]
    dyn_flat = dyn_lags_arr.reshape(LAG_DAYS*len(DYN_BASE_IDXS), H, W)
    parts = [dyn_flat]
    if len(diff_feats_arr): parts.append(diff_feats_arr)
    parts.append(burn_now[None, ...])
    parts.append(static_feats)
    feats_full = np.concatenate(parts, axis=0)

    X_t_flat = feats_full.reshape(feats_full.shape[0], -1).T.astype('float32', copy=False)
    prob = np.empty(X_t_flat.shape[0], dtype='float32')
    for s in range(0, X_t_flat.shape[0], BATCH):
        e = min(s+BATCH, X_t_flat.shape[0])
        prob[s:e] = proba_for_one(MODEL_ANY, X_t_flat[s:e])
  
    prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)
    risk = prob.reshape(H, W)

    base = stack[t, B_NDVI]
    
    try:
        vmin_base, vmax_base = np.nanpercentile(base,5), np.nanpercentile(base,95)
    except Exception:
        vmin_base, vmax_base = float(np.nanmin(base)), float(np.nanmax(base))
    if not np.isfinite(vmin_base) or not np.isfinite(vmax_base) or vmin_base == vmax_base:
        vmin_base, vmax_base = float(np.nanmin(base)), float(np.nanmax(base))
        if not np.isfinite(vmin_base) or not np.isfinite(vmax_base) or vmin_base == vmax_base:
            vmin_base, vmax_base = 0.0, 1.0

    
    if t == start_t:
        above_thr = int((prob >= float(THR_ANY) if THR_ANY is not None else prob >= 0.5).sum())
        print(f'[Diag t={t}] risk min={float(risk.min()):.3f} max={float(risk.max()):.3f} mean={float(risk.mean()):.3f}  >=thr count={above_thr}')

    plt.figure(figsize=(12,4))
    ax1 = plt.subplot(1,3,1)
    im1 = ax1.imshow(base, cmap='Greens', vmin=vmin_base, vmax=vmax_base)
    ax1.set_title(f'NDVI t={t}')
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.ax.tick_params(labelsize=7)
    cbar1.set_label('NDVI', fontsize=8)

    ax2 = plt.subplot(1,3,2)
    im2 = ax2.imshow(risk, cmap='inferno', vmin=0, vmax=1)
    ax2.set_title('Risk (probability)')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.ax.tick_params(labelsize=7)
    cbar2.set_label('Risk (0–1)', fontsize=8)

    actual = (stack[t+HORIZON, B_BURN] > 0).astype(float)
    ax3 = plt.subplot(1,3,3)
    ax3.imshow(base, cmap='Greens', vmin=vmin_base, vmax=vmax_base)
    ax3.imshow(np.ma.masked_where(actual==0, actual), cmap='autumn', alpha=0.6)
    ax3.set_title(f'Actual t+{HORIZON}')
    ax3.axis('off')

    out_png = os.path.join(gif_tmp, f'frame_{t:04d}.png')
    plt.tight_layout(); plt.savefig(out_png, dpi=110); plt.close()


    if not 'first_saved' in globals() or first_saved is False:
        try:
            prob_arr = np.asarray(prob).reshape(-1)
            uniq_rounded = np.unique(np.round(prob_arr, 3))
            print(f'prob stats t={t}: min={prob_arr.min():.4f}, max={prob_arr.max():.4f}, mean={prob_arr.mean():.4f}, std={prob_arr.std():.4f}, unique≈{len(uniq_rounded)} (rounded 3dp)')
            print('MODEL_ANY.classes_ =', getattr(MODEL_ANY, 'classes_', None))
           
            try:
                print('X_t_flat shape:', getattr(globals(), 'X_t_flat', np.array([])).shape)
                if 'X_t_flat' in globals():
                    col_std = np.nanstd(X_t_flat, axis=0)
                    print(f'feature std t={t}: min={np.min(col_std):.3e}, median={np.median(col_std):.3e}, max={np.max(col_std):.3e}')
                    if np.all(col_std < 1e-9):
                        print('WARNING: All features nearly constant across pixels at this t; risk will be flat.')
            except Exception as fe:
                print('Feature diag error:', fe)
           
            try:
                import matplotlib.pyplot as plt
                hist_png = os.path.join(FIG_DIR, f'risk_hist_t{t:04d}.png')
                plt.figure(figsize=(4,3)); plt.hist(prob_arr, bins=50, range=(0,1)); plt.title(f'Risk histogram t={t}'); plt.tight_layout(); plt.savefig(hist_png); plt.close()
                print('Saved risk histogram ->', hist_png)
            except Exception as he:
                print('Histogram save error:', he)
        except Exception as de:
            print('Diag error:', de)
 
    arr_rgb = np.array(Image.open(out_png).convert('RGB'))
    frames.append(arr_rgb)

GIF_PATH = os.path.join(FIG_DIR, 'test_risk_anyburn.gif')
if len(frames) == 0:
    print('No frames to write — check start/end indices (start_t > last_input_t).')
else:
    
    mimsave(GIF_PATH, frames, fps=3)
    print('Saved GIF ->', GIF_PATH, '| frames:', len(frames))


import os, glob
import numpy as np
from imageio.v2 import imread, mimsave

png_dir = os.path.join(FIG_DIR, 'gif_tmp')
png_paths = sorted(glob.glob(os.path.join(png_dir, 'frame_*.png')))
print(f'Re-encoding GIF from {len(png_paths)} PNG frames in {png_dir!r}')

frames_rgb = []
for p in png_paths:
    arr = imread(p)
  
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
       
        a = np.asarray(arr, dtype=np.float32)
        a = np.clip(a, 0, 1)
        arr = (a * 255).astype(np.uint8)
    frames_rgb.append(arr)

out_gif = os.path.join(FIG_DIR, 'test_risk_anyburn_rgb.gif')
if frames_rgb:
    mimsave(out_gif, frames_rgb, fps=3)
    print('Saved RGB GIF ->', out_gif, '| frames:', len(frames_rgb))
else:
    print('No frames found to encode. Ensure the previous GIF cell created PNGs into figures/gif_tmp.')

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score
)


if 'MODEL_ANY' not in globals() or MODEL_ANY is None:
    raise SystemExit('MODEL_ANY is not available. Train or load the model in earlier cells.')
if 'X_test' not in globals() or 'y_any_test' not in globals():
    raise SystemExit('Test split not found. Run the feature/memmap + split cells first.')

BATCH = int(globals().get('BATCH_PRED', 200_000))
N = X_test.shape[0]
probs = np.empty(N, dtype=np.float32)

if hasattr(MODEL_ANY, 'predict_proba'):
    
    for s in range(0, N, BATCH):
        e = min(s + BATCH, N)
        pp = MODEL_ANY.predict_proba(X_test[s:e])
        if hasattr(MODEL_ANY, 'classes_') and 1 in list(MODEL_ANY.classes_):
            idx1 = int(np.where(MODEL_ANY.classes_ == 1)[0][0])
            probs[s:e] = pp[:, idx1]
        else:
            probs[s:e] = 0.0
elif hasattr(MODEL_ANY, 'decision_function'):
    
    from sklearn.preprocessing import MinMaxScaler
    for s in range(0, N, BATCH):
        e = min(s + BATCH, N)
        z = MODEL_ANY.decision_function(X_test[s:e]).reshape(-1, 1)
        probs[s:e] = MinMaxScaler().fit_transform(z).ravel()
else:
  
    for s in range(0, N, BATCH):
        e = min(s + BATCH, N)
        probs[s:e] = MODEL_ANY.predict(X_test[s:e]).astype(np.float32)


probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
probs = np.clip(probs, 0.0, 1.0)


unique_y = np.unique(y_any_test)
roc_auc = roc_auc_score(y_any_test, probs) if len(unique_y) > 1 else float('nan')
pr_auc = average_precision_score(y_any_test, probs) if len(unique_y) > 1 else float('nan')
print(f'ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}')


if 'THR_ANY' in globals():
    thr = float(THR_ANY)
else:
    ths = np.linspace(0.05, 0.95, 19)
    f1s = []
    for th in ths:
        pred = (probs >= th).astype('uint8')
        f1s.append(f1_score(y_any_test, pred, zero_division=0))
    best_idx = int(np.argmax(f1s))
    thr = float(ths[best_idx])
    print(f'Best F1 threshold from grid: {thr:.3f} (F1={f1s[best_idx]:.4f})')

pred_labels = (probs >= thr).astype('uint8')
acc = accuracy_score(y_any_test, pred_labels)
cm = confusion_matrix(y_any_test, pred_labels)
report = classification_report(y_any_test, pred_labels, digits=4, zero_division=0)
print(f'Accuracy: {acc:.4f}')
print('Confusion matrix:\n', cm)
print('Classification report:\n', report)


FIG_DIR = globals().get('FIG_DIR', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


try:
    fpr, tpr, _ = roc_curve(y_any_test, probs)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.3f}')
    plt.plot([0,1], [0,1], 'k--', alpha=0.4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (ANY_BURN)')
    plt.legend()
    roc_png = os.path.join(FIG_DIR, 'eval_roc_anyburn.png')
    plt.tight_layout(); plt.savefig(roc_png, dpi=140); plt.show()
    print('Saved:', roc_png)
except Exception as e:
    print('ROC curve plotting skipped:', e)

try:
    prec, rec, _ = precision_recall_curve(y_any_test, probs)
    plt.figure(figsize=(5,4))
    plt.plot(rec, prec, label=f'PR AUC={pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (ANY_BURN)')
    plt.legend()
    pr_png = os.path.join(FIG_DIR, 'eval_pr_anyburn.png')
    plt.tight_layout(); plt.savefig(pr_png, dpi=140); plt.show()
    print('Saved:', pr_png)
except Exception as e:
    print('PR curve plotting skipped:', e)

plt.figure(figsize=(5,4))
plt.hist(probs, bins=50, range=(0,1), color='steelblue', alpha=0.8)
plt.axvline(thr, color='crimson', linestyle='--', label=f'Threshold={thr:.2f}')
plt.xlabel('Predicted probability')
plt.ylabel('Count')
plt.title('Probability Histogram (ANY_BURN)')
plt.legend()
hist_png = os.path.join(FIG_DIR, 'eval_hist_anyburn.png')
plt.tight_layout(); plt.savefig(hist_png, dpi=140); plt.show()
print('Saved:', hist_png)

try:
    import json
    metrics = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'accuracy': float(acc),
        'threshold': float(thr),
        'confusion_matrix': cm.tolist(),
    }
    with open(os.path.join(FIG_DIR, 'eval_metrics_anyburn.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Saved metrics JSON ->', os.path.join(FIG_DIR, 'eval_metrics_anyburn.json'))
except Exception as e:
    print('Saving metrics JSON failed:', e)

import os, json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, f1_score, accuracy_score
)
import matplotlib.pyplot as plt


if 'X' not in globals() or 'y_any' not in globals():
    raise SystemExit('X/y_any not found. Run feature/memmap cell first.')

RANDOM_STATE   = int(globals().get('RANDOM_STATE', 42))
NEG_PER_POS    = int(globals().get('NEG_PER_POS', 4))
MAX_TRAIN_SAMP = int(globals().get('MAX_TRAIN_SAMPLES', 800_000))
MAX_TEST_EVAL  = int(globals().get('MAX_TEST_EVAL', 300_000))  
BATCH_PRED     = int(globals().get('BATCH_PRED', 200_000))
FIG_DIR        = globals().get('FIG_DIR', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

N_all = int(y_any.shape[0])
all_idx = np.arange(N_all, dtype=np.int64)
tr_idx, te_idx = train_test_split(all_idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y_any)
print('Index-only split -> train:', tr_idx.shape, 'test:', te_idx.shape)

rng = np.random.RandomState(RANDOM_STATE)
pos_tr = tr_idx[y_any[tr_idx] == 1]
neg_tr = tr_idx[y_any[tr_idx] == 0]
if pos_tr.size == 0:
    raise SystemExit('No positives in training split; cannot train.')

n_neg_target = min(neg_tr.size, NEG_PER_POS * pos_tr.size)
neg_sel = rng.choice(neg_tr, size=n_neg_target, replace=False)
keep_tr = np.concatenate([pos_tr, neg_sel])
if keep_tr.size > MAX_TRAIN_SAMP:
    keep_tr = rng.choice(keep_tr, size=MAX_TRAIN_SAMP, replace=False)
keep_tr.sort()

X_tr_bal = np.asarray(X[keep_tr])
y_tr_bal = np.asarray(y_any[keep_tr])
perm = rng.permutation(X_tr_bal.shape[0])
X_tr_bal = X_tr_bal[perm]
y_tr_bal = y_tr_bal[perm]
print('Train subset ->', X_tr_bal.shape, 'positives:', int(y_tr_bal.sum()))

RF_PARAMS = dict(
    n_estimators=250,
    max_depth=20,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    class_weight='balanced_subsample',
    max_samples=0.5,
    random_state=RANDOM_STATE,
)
rf_fast = RandomForestClassifier(**RF_PARAMS)
rf_fast.fit(X_tr_bal, y_tr_bal)


pos_te = te_idx[y_any[te_idx] == 1]
neg_te = te_idx[y_any[te_idx] == 0]

n_pos_keep = pos_te.size
n_neg_keep = max(0, min(neg_te.size, MAX_TEST_EVAL - n_pos_keep))
neg_te_keep = rng.choice(neg_te, size=n_neg_keep, replace=False) if n_neg_keep > 0 else np.array([], dtype=neg_te.dtype)
keep_te = np.concatenate([pos_te, neg_te_keep])
keep_te.sort()

X_te_cap = X[keep_te]
y_te_cap = y_any[keep_te]
print('Eval subset ->', X_te_cap.shape, 'positives:', int(y_te_cap.sum()), 'total:', X_te_cap.shape[0])

N = X_te_cap.shape[0]
probs = np.empty(N, dtype=np.float32)
if hasattr(rf_fast, 'predict_proba'):
    for s in range(0, N, BATCH_PRED):
        e = min(s + BATCH_PRED, N)
        pp = rf_fast.predict_proba(X_te_cap[s:e])
        idx1 = int(np.where(rf_fast.classes_ == 1)[0][0]) if 1 in list(rf_fast.classes_) else 1
        probs[s:e] = pp[:, idx1]
else:
    for s in range(0, N, BATCH_PRED):
        e = min(s + BATCH_PRED, N)
        probs[s:e] = rf_fast.predict(X_te_cap[s:e]).astype(np.float32)

probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
probs = np.clip(probs, 0.0, 1.0)


unique_y = np.unique(y_te_cap)
roc_auc = roc_auc_score(y_te_cap, probs) if len(unique_y) > 1 else float('nan')
pr_auc  = average_precision_score(y_te_cap, probs) if len(unique_y) > 1 else float('nan')
print(f'[RandSplit FAST] ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}')

ths = np.linspace(0.05, 0.95, 19)
f1s = []
for th in ths:
    pred = (probs >= th).astype('uint8')
    f1s.append(f1_score(y_te_cap, pred, zero_division=0))
best_idx = int(np.argmax(f1s))
thr = float(ths[best_idx])
print(f'[RandSplit FAST] Best F1 thr: {thr:.3f} (F1={f1s[best_idx]:.4f})')

pred_labels = (probs >= thr).astype('uint8')
acc = accuracy_score(y_te_cap, pred_labels)
cm  = confusion_matrix(y_te_cap, pred_labels)
report = classification_report(y_te_cap, pred_labels, digits=4, zero_division=0)
print(f'[RandSplit FAST] Accuracy: {acc:.4f}')
print('[RandSplit FAST] Confusion matrix:\n', cm)
print('[RandSplit FAST] Classification report:\n', report)

suffix = 'randsplit_fast'
try:
    fpr, tpr, _ = roc_curve(y_te_cap, probs)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.3f}')
    plt.plot([0,1], [0,1], 'k--', alpha=0.4)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (ANY_BURN, RandSplit FAST)'); plt.legend()
    out_png = os.path.join(FIG_DIR, f'eval_roc_anyburn_{suffix}.png')
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.show()
    print('Saved:', out_png)
except Exception as e:
    print('ROC plot skipped:', e)

try:
    prec, rec, _ = precision_recall_curve(y_te_cap, probs)
    plt.figure(figsize=(5,4))
    plt.plot(rec, prec, label=f'PR AUC={pr_auc:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall (ANY_BURN, RandSplit FAST)'); plt.legend()
    out_png = os.path.join(FIG_DIR, f'eval_pr_anyburn_{suffix}.png')
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.show()
    print('Saved:', out_png)
except Exception as e:
    print('PR plot skipped:', e)

plt.figure(figsize=(5,4))
plt.hist(probs, bins=50, range=(0,1), color='steelblue', alpha=0.8)
plt.axvline(thr, color='crimson', linestyle='--', label=f'Threshold={thr:.2f}')
plt.xlabel('Predicted probability'); plt.ylabel('Count')
plt.title('Probability Histogram (ANY_BURN, RandSplit FAST)'); plt.legend()
out_png = os.path.join(FIG_DIR, f'eval_hist_anyburn_{suffix}.png')
plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.show()
print('Saved:', out_png)

try:
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': rf_fast, 'thr': thr, 'split': 'random_stratified_fast'}, f'models/anyburn_rf_{suffix}.joblib')
    print(f'Saved -> models/anyburn_rf_{suffix}.joblib')
except Exception as e:
    print('Saving model bundle failed:', e)

try:
    metrics = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'accuracy': float(acc),
        'threshold': float(thr),
        'confusion_matrix': cm.tolist(),
        'n_test_eval': int(N),
    }
    with open(os.path.join(FIG_DIR, f'eval_metrics_anyburn_{suffix}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Saved metrics JSON ->', os.path.join(FIG_DIR, f'eval_metrics_anyburn_{suffix}.json'))
except Exception as e:
    print('Saving metrics JSON failed:', e)

import os, json
import numpy as np
import math


if 'X' not in globals() or 'y_any' not in globals():
    raise SystemExit('X/y_any not found. Run recovery or feature build cell first.')

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    print('PyTorch not available:', e)

if not TORCH_OK:
    raise SystemExit('Install PyTorch to run DL baselines.')

RANDOM_STATE   = int(globals().get('RANDOM_STATE', 42))
NEG_PER_POS    = int(globals().get('NEG_PER_POS', 4))
MAX_TRAIN_SAMP = int(globals().get('MAX_TRAIN_SAMPLES', 400_000))  
MAX_TEST_EVAL  = int(globals().get('MAX_TEST_EVAL', 300_000))
BATCH_TRAIN    = 4096
BATCH_EVAL     = 200_000
EPOCHS         = 3  
LR             = 1e-3
FIG_DIR        = globals().get('FIG_DIR', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

from sklearn.model_selection import train_test_split
rng = np.random.RandomState(RANDOM_STATE)
N_all = int(y_any.shape[0])
all_idx = np.arange(N_all, dtype=np.int64)
tr_idx, te_idx = train_test_split(all_idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y_any)


pos_tr = tr_idx[y_any[tr_idx] == 1]
neg_tr = tr_idx[y_any[tr_idx] == 0]
if pos_tr.size == 0:
    raise SystemExit('No positives in training split; cannot train DL models.')
neg_target = min(neg_tr.size, NEG_PER_POS * pos_tr.size)
neg_sel = rng.choice(neg_tr, size=neg_target, replace=False)
keep_tr = np.concatenate([pos_tr, neg_sel])
if keep_tr.size > MAX_TRAIN_SAMP:
    keep_tr = rng.choice(keep_tr, size=MAX_TRAIN_SAMP, replace=False)
keep_tr.sort()

X_tr = np.asarray(X[keep_tr])
y_tr = np.asarray(y_any[keep_tr])
perm = rng.permutation(X_tr.shape[0])
X_tr = X_tr[perm]
y_tr = y_tr[perm]

pos_te = te_idx[y_any[te_idx] == 1]
neg_te = te_idx[y_any[te_idx] == 0]
neg_keep = max(0, min(neg_te.size, MAX_TEST_EVAL - pos_te.size))
neg_sel_te = rng.choice(neg_te, size=neg_keep, replace=False) if neg_keep>0 else np.array([], dtype=np.int64)
keep_te = np.concatenate([pos_te, neg_sel_te]); keep_te.sort()
X_te = X[keep_te]; y_te = y_any[keep_te]

F = X_tr.shape[1]

class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = ArrayDataset(X_tr, y_tr)
train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=0, pin_memory=False)

class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class CNN1D(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*32, 128), nn.ReLU(), nn.Linear(128, 1)
        )
    def forward(self, x):
        x = x.unsqueeze(1)             
        x = self.conv(x)                
        x = x.view(x.size(0), -1)      
        return self.fc(x).squeeze(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(NEG_PER_POS)))

def train_model(model, loader, epochs=EPOCHS, lr=LR):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=False)
            yb = yb.to(DEVICE, non_blocking=False).float()
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * xb.size(0)
        print(f'Epoch {ep+1}/{epochs} loss={tot/len(loader.dataset):.4f}')
    return model


mlp = train_model(MLP(F), train_loader)
cnn = train_model(CNN1D(F), train_loader)


from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_recall_curve

def predict_proba_torch(model, Xc, batch=BATCH_EVAL):
    model.eval(); out = np.empty(Xc.shape[0], dtype=np.float32)
    with torch.no_grad():
        for s in range(0, Xc.shape[0], batch):
            e = min(s+batch, Xc.shape[0])
            xb = torch.as_tensor(Xc[s:e], dtype=torch.float32, device=DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            out[s:e] = probs.ravel()
    return out

probs_mlp = predict_proba_torch(mlp, X_te)
probs_cnn = predict_proba_torch(cnn, X_te)


def eval_scores(name, probs, y_true):
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, 0.0, 1.0)
    uniq = np.unique(y_true)
    roc = roc_auc_score(y_true, probs) if len(uniq)>1 else float('nan')
    pr  = average_precision_score(y_true, probs) if len(uniq)>1 else float('nan')
    ths = np.linspace(0.05, 0.95, 19)
    f1s = []
    for th in ths:
        pred = (probs >= th).astype('uint8')
        f1s.append(f1_score(y_true, pred, zero_division=0))
    best = int(np.argmax(f1s)); thr = float(ths[best])
    pred = (probs >= thr).astype('uint8')
    acc = accuracy_score(y_true, pred)
    print(f'[{name}] ROC={roc:.4f} PR={pr:.4f} F1@best={f1s[best]:.4f} thr={thr:.2f} ACC={acc:.4f}')
    return dict(roc_auc=float(roc), pr_auc=float(pr), f1=float(f1s[best]), thr=float(thr), acc=float(acc))

scores_mlp = eval_scores('MLP', probs_mlp, y_te)
scores_cnn = eval_scores('CNN1D', probs_cnn, y_te)

rf_metrics_paths = [
    os.path.join(FIG_DIR, 'eval_metrics_anyburn_randsplit_fast.json'),
    os.path.join(FIG_DIR, 'eval_metrics_anyburn_randsplit.json'),
    os.path.join(FIG_DIR, 'eval_metrics_anyburn.json'),
]
rf_scores = None
for p in rf_metrics_paths:
    if os.path.exists(p):
        try:
            with open(p, 'r') as f:
                rf_scores = json.load(f)
            print('Loaded RF metrics from', p)
            break
        except Exception:
            pass

def fmt(k, d):
    return f"{d.get(k, float('nan')):.4f}" if k in d else 'nan'

print('\n=== Model comparison (capped test) ===')
if rf_scores is not None:
    print('RandomForest: ROC', fmt('roc_auc', rf_scores), 'PR', fmt('pr_auc', rf_scores), 'ACC', fmt('accuracy', rf_scores))
else:
    print('RandomForest: metrics not found (run RF eval cell to create JSON).')
print('MLP         : ROC', f"{scores_mlp['roc_auc']:.4f}", 'PR', f"{scores_mlp['pr_auc']:.4f}", 'ACC', f"{scores_mlp['acc']:.4f}")
print('CNN1D       : ROC', f"{scores_cnn['roc_auc']:.4f}", 'PR', f"{scores_cnn['pr_auc']:.4f}", 'ACC', f"{scores_cnn['acc']:.4f}")

try:
    import joblib
    os.makedirs('models', exist_ok=True)
    torch.save(mlp.state_dict(), 'models/anyburn_mlp.pt')
    torch.save(cnn.state_dict(), 'models/anyburn_cnn1d.pt')
    with open(os.path.join(FIG_DIR, 'eval_metrics_anyburn_mlp.json'), 'w') as f:
        json.dump(scores_mlp, f, indent=2)
    with open(os.path.join(FIG_DIR, 'eval_metrics_anyburn_cnn1d.json'), 'w') as f:
        json.dump(scores_cnn, f, indent=2)
    print('Saved DL models and metrics to models/ and figures/.')
except Exception as e:
    print('Saving DL artifacts failed:', e)


import os, json
import numpy as np

if 'stack' not in globals():
    raise SystemExit('`stack` not found. Run earlier cells that build/load the stack.')

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise SystemExit(f'PyTorch required for 3D CNN. Install it first. ({e})')


LAG = int(globals().get('LAG_DAYS', 3))
PATCH = 32
RANDOM_STATE = int(globals().get('RANDOM_STATE', 42))
POS_TRAIN_CAP = 12000
POS_VAL_CAP   = 3000
NEG_PER_POS   = int(globals().get('NEG_PER_POS', 4))
EPOCHS        = 30            
BATCH_TRAIN   = 32
BATCH_EVAL    = 128
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 5             
FIG_DIR       = globals().get('FIG_DIR', 'figures')
PATCH_SPLIT_MODE = str(globals().get('PATCH_SPLIT_MODE', 'stratified')).lower() 
os.makedirs(FIG_DIR, exist_ok=True)


if 'DYN_BASE_IDXS' not in globals():
    DYN_BASE_IDXS = [0,1,2,3,4] 
if 'STATIC_IDXS' not in globals():
    STATIC_IDXS = [6,7,8,9,10]   
if 'B_BURN' not in globals():
    B_BURN = 5

T, B, H, W = int(stack.shape[0]), int(stack.shape[1]), int(stack.shape[2]), int(stack.shape[3])
first_t = LAG - 1
last_t  = T - 2
if last_t < first_t:
    raise SystemExit('Not enough time steps for 3D CNN training.')

rng = np.random.RandomState(RANDOM_STATE)


all_ts = np.arange(first_t, last_t + 1)

pad = PATCH // 2

def extract_patch(img2d, y, x):
    """Reflect-pad and extract PATCH x PATCH patch centered at (y,x)."""
    if pad > 0:
        img2d = np.pad(img2d, ((pad, pad), (pad, pad)), mode='reflect')
        return img2d[y:y+PATCH, x:x+PATCH]
    else:
        return img2d[y:y+PATCH, x:x+PATCH]

def sample_coords(ts, pos_cap):
    pos_coords = []
    for t in ts:
        m = (stack[t+1, B_BURN] > 0)
        ys, xs = np.where(m)
        if ys.size:
            take = min(ys.size, max(200, pos_cap // max(1, ts.size)))
            sel = rng.choice(np.arange(ys.size), size=take, replace=False)
            for j in sel:
                pos_coords.append((t, int(ys[j]), int(xs[j])))
        if len(pos_coords) >= pos_cap:
            break
    pos_coords = pos_coords[:pos_cap]

    neg_coords = []
    need_neg = min(len(pos_coords) * NEG_PER_POS, 4 * pos_cap)
    attempts = 0
    max_attempts = need_neg * 10 + 10000
    while len(neg_coords) < need_neg and attempts < max_attempts:
        t = int(rng.choice(ts))
        y = int(rng.randint(0, H))
        x = int(rng.randint(0, W))
        if stack[t+1, B_BURN, y, x] <= 0:
            neg_coords.append((t, y, x))
        attempts += 1
    return pos_coords, neg_coords


if PATCH_SPLIT_MODE == 'time':
    rng.shuffle(all_ts)
    split = int(0.8 * all_ts.size)
    train_ts, val_ts = np.sort(all_ts[:split]), np.sort(all_ts[split:])
    pos_tr, neg_tr = sample_coords(train_ts, POS_TRAIN_CAP)
    pos_va, neg_va = sample_coords(val_ts, POS_VAL_CAP)
else: 
    total_pos_cap = POS_TRAIN_CAP + POS_VAL_CAP
    pos_all, neg_all = sample_coords(all_ts, total_pos_cap)
    coords_all = [(c, 1) for c in pos_all] + [(c, 0) for c in neg_all]
    if not coords_all:
        raise SystemExit('No coordinates sampled; check data and caps.')
    labels_all = np.array([1]*len(pos_all) + [0]*len(neg_all), dtype=np.int64)
    idx = np.arange(len(coords_all))
    from sklearn.model_selection import train_test_split
    test_size = POS_VAL_CAP / max(1, (POS_TRAIN_CAP + POS_VAL_CAP))
    tr_idx, va_idx = train_test_split(idx, test_size=test_size, random_state=RANDOM_STATE, stratify=labels_all if labels_all.size>1 else None)
    coords_tr = [coords_all[i] for i in tr_idx]
    coords_va = [coords_all[i] for i in va_idx]
    pos_tr = [c for c,l in coords_tr if l==1]
    neg_tr = [c for c,l in coords_tr if l==0]
    pos_va = [c for c,l in coords_va if l==1]
    neg_va = [c for c,l in coords_va if l==0]

print(f'[3D CNN] Split={PATCH_SPLIT_MODE}  Train coords -> pos: {len(pos_tr)}, neg: {len(neg_tr)}')
print(f'[3D CNN] Split={PATCH_SPLIT_MODE}  Val   coords -> pos: {len(pos_va)}, neg: {len(neg_va)}')

C_dyn = len(DYN_BASE_IDXS)
C_stat = len(STATIC_IDXS)
C_in = C_dyn + C_stat

def safe_std(a):
    s = float(np.nanstd(a))
    return 1.0 if not np.isfinite(s) or s < 1e-6 else s

dyn_means, dyn_stds = [], []
for bidx in DYN_BASE_IDXS:
    
    if PATCH_SPLIT_MODE == 'time':
        ts_use = train_ts
    else:
        ts_use = all_ts
    arr = stack[ts_use][:, bidx].astype('float32', copy=False) 
    dyn_means.append(float(np.nanmean(arr)))
    dyn_stds.append(safe_std(arr))

stat_means, stat_stds = [], []
ref_t = int(all_ts[0]) if all_ts.size else 0
for bidx in STATIC_IDXS:
    arr = stack[ref_t, bidx].astype('float32', copy=False)  
    stat_means.append(float(np.nanmean(arr)))
    stat_stds.append(safe_std(arr))

CH_MEAN = np.asarray(dyn_means + stat_means, dtype=np.float32)  
CH_STD  = np.asarray(dyn_stds + stat_stds, dtype=np.float32)    

class PatchDataset(Dataset):
    def __init__(self, coords_pos, coords_neg, ch_mean=None, ch_std=None):
        self.coords = [(c, 1) for c in coords_pos] + [(c, 0) for c in coords_neg]
        rng.shuffle(self.coords)
        self.len_dyn = len(DYN_BASE_IDXS)
        self.len_stat = len(STATIC_IDXS)
        self.ch_mean = None if ch_mean is None else np.asarray(ch_mean, dtype=np.float32)
        self.ch_std  = None if ch_std  is None else np.asarray(ch_std, dtype=np.float32)
    def __len__(self):
        return len(self.coords)
    def __getitem__(self, idx):
        (t, y, x), label = self.coords[idx]

        dyn = np.empty((self.len_dyn, LAG, PATCH, PATCH), dtype=np.float32)
        for li in range(LAG):
            tt = t - (LAG - 1 - li)
            for ci, bidx in enumerate(DYN_BASE_IDXS):
                dyn[ci, li] = extract_patch(stack[tt, bidx], y, x)

        stat = np.empty((self.len_stat, LAG, PATCH, PATCH), dtype=np.float32)
        for si, bidx in enumerate(STATIC_IDXS):
            p2d = extract_patch(stack[t, bidx], y, x)
            stat[si] = np.repeat(p2d[None, ...], LAG, axis=0)
        x3d = np.concatenate([dyn, stat], axis=0)  

        if self.ch_mean is not None and self.ch_std is not None:
            x3d = (x3d - self.ch_mean[:, None, None, None]) / (self.ch_std[:, None, None, None] + 1e-6)
            x3d = np.clip(x3d, -10.0, 10.0) 
        return x3d, np.int64(label)

train_ds = PatchDataset(pos_tr, neg_tr, ch_mean=CH_MEAN, ch_std=CH_STD)
val_ds   = PatchDataset(pos_va, neg_va, ch_mean=CH_MEAN, ch_std=CH_STD)
train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_EVAL, shuffle=False, num_workers=0)

class Small3DCNN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2)),
            nn.AdaptiveAvgPool3d((1,1,1))
)
        self.head = nn.Linear(64, 1)
    def forward(self, x): 
        z = self.net(x)
        z = z.view(z.size(0), -1)
        return self.head(z).squeeze(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model3d = Small3DCNN(C_in).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(NEG_PER_POS), device=DEVICE))
opt = torch.optim.Adam(model3d.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2, min_lr=1e-5)  # no 'verbose' for older torch

def train_epoch(loader):
    model3d.train()
    tot = 0.0
    for xb, yb in loader:
        xb = torch.as_tensor(xb, dtype=torch.float32, device=DEVICE)
        yb = torch.as_tensor(yb, dtype=torch.float32, device=DEVICE)
        opt.zero_grad()
        logits = model3d(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model3d.parameters(), max_norm=1.0)
        opt.step()
        tot += float(loss.item()) * xb.size(0)
    return tot / len(loader.dataset)

@torch.no_grad()
def evaluate_pr(loader):
    model3d.eval()
    probs_all, ys_all = [], []
    for xb, yb in loader:
        xb = torch.as_tensor(xb, dtype=torch.float32, device=DEVICE)
        logits = model3d(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        probs_all.append(probs)
        ys_all.append(np.asarray(yb, dtype=np.int64))
    probs_all = np.concatenate(probs_all) if probs_all else np.zeros(0, dtype=np.float32)
    ys_all = np.concatenate(ys_all) if ys_all else np.zeros(0, dtype=np.int64)
    from sklearn.metrics import average_precision_score
    if ys_all.size == 0 or len(np.unique(ys_all)) < 2:
        return float('nan'), probs_all, ys_all
    return float(average_precision_score(ys_all, probs_all)), probs_all, ys_all

best_pr = -1.0
no_improve = 0
best_path = os.path.join('models', 'anyburn_3dcnn_best.pt')
os.makedirs('models', exist_ok=True)

for ep in range(1, EPOCHS+1):
    loss = train_epoch(train_loader)
    pr, _, _ = evaluate_pr(val_loader)
    cur_lr = opt.param_groups[0]['lr']
    print(f'[3D CNN] Epoch {ep}/{EPOCHS} loss={loss:.4f} PR(AUC)={pr:.4f} lr={cur_lr:.6f}')
    if np.isfinite(pr):
        scheduler.step(pr)
    if pr > best_pr:
        best_pr = pr
        no_improve = 0
        torch.save(model3d.state_dict(), best_path)
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f'Early stopping at epoch {ep} (no improve {no_improve} >= {PATIENCE}).')
            break

if os.path.exists(best_path):
    model3d.load_state_dict(torch.load(best_path, map_location=DEVICE))

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
probs_va, y_va = evaluate_pr(val_loader)[1:]
probs_va = probs_va if isinstance(probs_va, np.ndarray) else np.array(probs_va)
y_va = y_va if isinstance(y_va, np.ndarray) else np.array(y_va)
uniq = np.unique(y_va)
roc = roc_auc_score(y_va, probs_va) if y_va.size and len(uniq)>1 else float('nan')
pr  = average_precision_score(y_va, probs_va) if y_va.size and len(uniq)>1 else float('nan')
ths = np.linspace(0.05, 0.95, 19)
f1s = []
for th in ths:
    pred = (probs_va >= th).astype('uint8') if probs_va.size else np.zeros_like(y_va)
    f1s.append(f1_score(y_va, pred, zero_division=0) if y_va.size else 0.0)
best = int(np.argmax(f1s)) if len(f1s) else 0
thr = float(ths[best]) if len(ths) else 0.5
pred = (probs_va >= thr).astype('uint8') if probs_va.size else np.zeros_like(y_va)
acc = accuracy_score(y_va, pred) if y_va.size else float('nan')
print(f'[3D CNN] ROC={roc:.4f} PR={pr:.4f} F1@best={f1s[best]:.4f} thr={thr:.2f} ACC={acc:.4f}')

try:
    with open(os.path.join(FIG_DIR, 'eval_metrics_anyburn_3dcnn.json'), 'w') as f:
        json.dump({'roc_auc': float(roc), 'pr_auc': float(pr), 'f1': float(f1s[best]), 'thr': float(thr), 'acc': float(acc)}, f, indent=2)
    print('Saved 3D CNN metrics JSON.')
except Exception as e:
    print('Saving 3D metrics failed:', e)


rf_metrics_paths = [
    os.path.join(FIG_DIR, 'eval_metrics_anyburn_randsplit_fast.json'),
    os.path.join(FIG_DIR, 'eval_metrics_anyburn_randsplit.json'),
    os.path.join(FIG_DIR, 'eval_metrics_anyburn.json'),
]
rf_scores = None
for p in rf_metrics_paths:
    if os.path.exists(p):
        try:
            with open(p, 'r') as f:
                rf_scores = json.load(f)
            print('Loaded RF metrics from', p)
            break
        except Exception:
            pass

if rf_scores is not None:
    def fmt(k, d):
        return f"{d.get(k, float('nan')):.4f}" if k in d else 'nan'
    print('\n=== Comparison (time-holdout patches) ===')
    print('RandomForest: ROC', fmt('roc_auc', rf_scores), 'PR', fmt('pr_auc', rf_scores), 'ACC', fmt('accuracy', rf_scores))
    print('3D CNN      : ROC', f"{roc:.4f}", 'PR', f"{pr:.4f}", 'ACC', f"{acc:.4f}")
else:
    print('RF metrics JSON not found; run RF eval to compare.')


