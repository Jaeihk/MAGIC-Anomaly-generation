"""
Memory-efficient correspondence extractor
(top-bottom + center version, line∩object-mask search, line-viz save)
------------------------------------------------------------------
* Best points from top and bottom → line mask
* Within the intersection with the object mask in train/masks,
  select the final best point as the highest value in the
  'center-point similarity map'.
──────────────────────────────────────────────────────────────────
(2025-07-08) **Update**
  • Split the GT mask into connected components (islands) and
    compute center and 4-way points for each island.
  • Record defect image filename and component index
    (“region_idx”) in JSON.
(2025-07-09) **SPEED-MOD**
  • Cache defect features / up·down·center vectors on GPU at once.
  • Compute (K×H×W) similarity maps in batch with torch.einsum.
  • Extract features for normals in mini-batches.
  • Save JSON every 20 items only.
"""

###############################################################################
# 0) Required: handle GPU id (--gpu) + prevent CUDA memory fragmentation
###############################################################################
import os, sys, argparse, random, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

cli = argparse.ArgumentParser()
cli.add_argument("--gpu", default="0")
cli.add_argument("--mvtecad_dir", required=True)
cli.add_argument("--out_dir",       required=True)
cli.add_argument("--mask_root",     required=True)
cli.add_argument("--categories",    nargs="+", required=True)
cli.add_argument("--batch", type=int, default=8, help="batch size for normal images")  # SPEED-MOD
cli.add_argument("--img_size", type=int, default=512, help="square image size (default: 512)") 
args, _ = cli.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

###############################################################################
# 1) Common imports
###############################################################################
import glob, json, time, datetime
import numpy as np
from PIL import Image, ImageDraw
if not hasattr(Image, "LINEAR"):
    Image.LINEAR = Image.Resampling.BILINEAR

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils_correspondence import resize
from model_utils.extractor_sd import load_model, process_features_and_mask
from model_utils.extractor_dino import ViTExtractor
from model_utils.projection_network import AggregationNetwork
from preprocess_map import set_seed

###############################################################################
# 2) Environment setup & model loading
###############################################################################
set_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda")
num_patches = 60

sd_model, sd_aug = load_model(
    diffusion_ver="v1-5",
    image_size=num_patches * 16,
    num_timesteps=50,
    block_indices=[2, 5, 8, 11]
)
sd_model = sd_model.half().eval().to(device)

aggre_net = AggregationNetwork(
    feature_dims=[640, 1280, 1280, 768],
    projection_dim=768,
    device=device
).half().eval()
aggre_net.load_pretrained_weights(torch.load("results_spair/best_856.PTH"))

extractor_vit = ViTExtractor("dinov2_vitb14", stride=14, device="cpu")
extractor_vit.model.eval().requires_grad_(False)

###############################################################################
# Debug logging utils
###############################################################################
def _open_debug_writer(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "debug_log.txt")
    # open in append mode to accumulate logs
    fh = open(log_path, "a", encoding="utf-8")
    fh.write("\n" + "="*80 + "\n")
    fh.write(f"[START] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    fh.flush()
    return fh, log_path

def _log(fh, msg):
    # print to console via tqdm.write + write to file
    tqdm.write(msg)
    fh.write(msg + "\n")
    fh.flush()
    os.fsync(fh.fileno())

###############################################################################
# 3) Helper functions (some additions/changes)
###############################################################################
@torch.no_grad()
def get_processed_features(img_pil):
    with torch.cuda.amp.autocast(dtype=torch.float16):
        fsd = process_features_and_mask(
            sd_model, sd_aug,
            resize(img_pil, num_patches * 16, True, True),
            mask=False, raw=True
        )
    del fsd["s2"]

    dino_in = resize(img_pil, num_patches * 14, True, True)
    dino_tensor = extractor_vit.preprocess_pil(dino_in).to("cpu")
    fdino = extractor_vit.extract_descriptors(
        dino_tensor, layer=11, facet="token"
    ).permute(0, 1, 3, 2).reshape(1, -1, num_patches, num_patches)
    fdino = fdino.to(device=device, dtype=torch.float16, non_blocking=True)

    desc = torch.cat(
        [
            fsd["s3"].half(),
            F.interpolate(fsd["s4"].half(), (num_patches, num_patches),
                          mode="bilinear", align_corners=False),
            F.interpolate(fsd["s5"].half(), (num_patches, num_patches),
                          mode="bilinear", align_corners=False),
            fdino,
        ],
        dim=1,
    )
    return F.normalize(aggre_net(desc), dim=1)          # [1,C,h,w]


def mask_center(mask_np):
    ys, xs = np.where(mask_np > 127)
    return None if len(xs) == 0 else (int(xs.mean().round()), int(ys.mean().round()))


def mask_four_sym_points(mask_np):
    h, w = mask_np.shape
    c = mask_center(mask_np)
    if c is None:
        return None
    cx, cy = c
    if mask_np[cy, cx] <= 127:
        return None

    def move_dir(x, y, dx, dy):
        while (0 <= x + dx < w and 0 <= y + dy < h
               and mask_np[y + dy, x + dx] > 127):
            x += dx; y += dy
        return x, y

    x_l, _  = move_dir(cx, cy, -1, 0)
    x_r, _  = move_dir(cx, cy,  1, 0)
    _, y_u  = move_dir(cx, cy,  0, -1)
    _, y_d  = move_dir(cx, cy,  0,  1)
    return [(x_l, cy), (cx, y_u), (x_r, cy), (cx, y_d)]  # ← ↑ → ↓


###############################################################################
# 3-1) Utility to split a GT mask into components (islands)
###############################################################################
def split_mask_to_regions(mask_np, min_area=50):
    bin_np = (mask_np > 127).astype(np.uint8)
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_np, connectivity=8)
    regions = []
    for lbl in range(1, n_lbl):                 # 0 = background
        if stats[lbl, cv2.CC_STAT_AREA] < min_area:
            continue
        rmask = (labels == lbl).astype(np.uint8) * 255
        regions.append((rmask, lbl - 1))        # 0-based
    return regions


###############################################################################
# 4) Main loop
###############################################################################
def generate_correspondence_json(
    mvtecad_dir,
    categories,
    img_size=512,  # << 512 기준
    out_dir="output",
    out_json_name="matching_result_center.json",
    viz_dir_name="matched_visuals_line",
    mask_root="./normal_masks",
    batch=args.batch,                                         # SPEED-MOD
):
    os.makedirs(out_dir, exist_ok=True)
    viz_dir  = os.path.join(out_dir, viz_dir_name);           os.makedirs(viz_dir, exist_ok=True)
    line_dir = os.path.join(out_dir, viz_dir_name + "_line"); os.makedirs(line_dir, exist_ok=True)
    overlay_dir = os.path.join(out_dir, viz_dir_name + "_overlay"); os.makedirs(overlay_dir, exist_ok=True)
    out_json_path = os.path.join(out_dir, out_json_name)

    # open debug log file
    dbg_fh, dbg_path = _open_debug_writer(out_dir)
    _log(dbg_fh, f"[Info] Debug log path: {dbg_path}")
    _log(dbg_fh, f"[Args] mvtecad_dir={mvtecad_dir}")
    _log(dbg_fh, f"[Args] mask_root={mask_root}")
    _log(dbg_fh, f"[Args] categories={categories}")
    _log(dbg_fh, f"[Args] batch={batch}")
    _log(dbg_fh, f"[Args] img_size={img_size}")
    _log(dbg_fh, f"[Info] Using device: {device}")

    result = {}
    save_every = 20                                            # SPEED-MOD
    global_start = time.time()

    # overall summary statistics
    grand_total_components = 0
    grand_total_results = 0
    grand_no_intersections = 0

    for cat in categories:
        cat_start = time.time()
        cat_dir = os.path.join(mvtecad_dir, cat)
        is_mvtec_ad = os.path.isdir(os.path.join(cat_dir, "ground_truth"))
        test_dir = os.path.join(cat_dir, "test")
        if is_mvtec_ad:
            gt_root   = os.path.join(cat_dir, "ground_truth")
            norm_root = os.path.join(cat_dir, "train", "good")
        else:                                                 # MVTec-3D
            gt_root   = None
            norm_root = os.path.join(cat_dir, "train", "good", "rgb")
        if not os.path.exists(test_dir):
            _log(dbg_fh, f"[WARN][{cat}] test_dir not found -> skip: {test_dir}")
            continue

        defect_classes = sorted(
            d for d in os.listdir(test_dir)
            if d != "good" and os.path.isdir(os.path.join(test_dir, d))
        )
        normals = sorted(glob.glob(os.path.join(norm_root, "*.*")))
        result[cat] = {}

        _log(dbg_fh, f"[CAT] {cat} | is_mvtec_ad={is_mvtec_ad} | normals={len(normals)} | defect_classes={len(defect_classes)}")
        if len(normals) == 0:
            _log(dbg_fh, f"[WARN][{cat}] No normal images found in {norm_root}")

        for dcls in defect_classes:
            class_start = time.time()
            if is_mvtec_ad:
                dimgs_all = sorted(glob.glob(os.path.join(test_dir, dcls, "*.*")))
                gt_glob = lambda b: glob.glob(os.path.join(gt_root, dcls, f"{b}_mask.*")) or glob.glob(os.path.join(gt_root, dcls, f"{b}.*"))
            else:
                dimgs_all = sorted(glob.glob(os.path.join(test_dir, dcls, "rgb", "*.*")))
                gt_glob   = lambda b: glob.glob(os.path.join(test_dir, dcls, "gt", f"{b}.*"))

            use_cnt   = max(1, len(dimgs_all) // 3)
            dimgs_sel = dimgs_all[:use_cnt]
            _log(dbg_fh, f"[CLS] {cat}/{dcls} | test_imgs={len(dimgs_all)} | use_cnt={use_cnt}")

            # ──────────────── arrays for per-component information ───────────
            defect_feats   = []   # [1,C,h,w]
            defect_pts4    = []
            defect_centers = []
            defect_ids     = []   # (defect_img_name, region_idx)
            # debugging stats
            dbg_gt_found = 0
            dbg_region_total = 0
            # ────────────────────────────────────────────────────────────────

            for dpath in dimgs_sel:
                base = os.path.splitext(os.path.basename(dpath))[0]
                gt_files = gt_glob(base)
                if not gt_files:   # skip if no mask
                    _log(dbg_fh, f"[MISS][{cat}/{dcls}] GT not found for {base}")
                    continue
                dbg_gt_found += 1

                img_rgb = Image.open(dpath).convert("RGB")
                feat_def_img = get_processed_features(
                    resize(img_rgb, img_size, True, True)
                )                                                   # [1,C,h,w]

                mask_np_full = np.array(
                    Image.open(gt_files[0])
                    .convert("L")
                    .resize((img_size, img_size), Image.NEAREST)
                )
                regions = split_mask_to_regions(mask_np_full)
                dbg_region_total += len(regions)
                if len(regions) == 0:
                    _log(dbg_fh, f"[MISS][{cat}/{dcls}] No regions after CC for {base}")

                used_regions = 0
                for rmask_np, region_idx in regions:
                    pts4 = mask_four_sym_points(rmask_np)
                    ctr  = mask_center(rmask_np)
                    if pts4 is None or ctr is None:
                        continue
                    defect_feats.append(feat_def_img)
                    defect_pts4.append(pts4)
                    defect_centers.append(ctr)
                    defect_ids.append((os.path.basename(dpath), region_idx))
                    used_regions += 1

                if used_regions == 0:
                    _log(dbg_fh, f"[MISS][{cat}/{dcls}] All regions invalid (no center/4pts) for {base}")

            if not defect_feats:
                _log(dbg_fh, f"[SKIP][{cat}/{dcls}] No valid defect components. gt_found={dbg_gt_found}, cc_total={dbg_region_total}")
                continue

            K = len(defect_feats)                                   # SPEED-MOD
            grand_total_components += K
            _log(dbg_fh, f"[READY][{cat}/{dcls}] normals={len(normals)}, components(K)={K}, gt_found={dbg_gt_found}, cc_total={dbg_region_total}")

            result[cat][dcls] = []

            ###################################################################
            # SPEED-MOD ① : cache defect component features / vectors
            ###################################################################
            feat_stack = torch.cat([f.half() for f in defect_feats], dim=0) \
                                 .to(device)                         # [K,C,h,w]
            feat_stack_up = F.interpolate(
                feat_stack, (img_size, img_size),
                mode="bilinear", align_corners=False)                # [K,C,H,W]
            feat_stack_up = F.normalize(feat_stack_up, dim=1)

            up_y  = torch.tensor([p[1][1] for p in defect_pts4], device=device)
            up_x  = torch.tensor([p[1][0] for p in defect_pts4], device=device)
            dn_y  = torch.tensor([p[3][1] for p in defect_pts4], device=device)
            dn_x  = torch.tensor([p[3][0] for p in defect_pts4], device=device)
            ctr_y = torch.tensor([c[1]      for c in defect_centers], device=device)
            ctr_x = torch.tensor([c[0]      for c in defect_centers], device=device)

            vec_up  = feat_stack_up[torch.arange(K), :, up_y,  up_x] # [K,C]
            vec_dn  = feat_stack_up[torch.arange(K), :, dn_y,  dn_x] # [K,C]
            vec_ctr = feat_stack_up[torch.arange(K), :, ctr_y, ctr_x]# [K,C]

            # per-class stats
            cls_results = 0
            cls_no_intersections = 0
            cls_batches_with_zero_valid = 0

            ###################################################################
            # Batch loop over normal images
            ###################################################################
            for b0 in tqdm(range(0, len(normals), batch),
                           ncols=90, desc=f"{cat}/{dcls}"):
                b_idx = list(range(b0, min(b0+batch, len(normals))))
                npaths_batch = [normals[i] for i in b_idx]

                # ─ object mask · PIL resize pre-processing
                obj_masks = []
                imgs_pil  = []
                base_imgs = []

                batch_missing_mask = 0
                batch_empty_mask = 0

                for npath in npaths_batch:
                    base_img = os.path.splitext(os.path.basename(npath))[0]
                    base_imgs.append(base_img)
                    cand = []
                    if is_mvtec_ad:
                        cand = [
                            os.path.join(mask_root, cat, "train", "masks", f"{base_img}.png"),
                            os.path.join(mask_root, cat, "train", "masks", f"{base_img}_mask.png"),
                        ]
                    else:
                        cand = [
                            os.path.join(mask_root, cat, "train", "masks", f"{base_img}_mask.png"),
                            os.path.join(mask_root, cat, "train", "masks", f"{base_img}.png"),
                        ]

                    obj_path = next((p for p in cand if os.path.isfile(p)), None)

                    # ──────────────── obj_path sanity check ────────────────
                    if obj_path is None:
                        batch_missing_mask += 1
                        _log(dbg_fh,
                             f"[MISS-MASK][{cat}] {base_img} | tried: {cand[0]} || {cand[1]}")
                        raise FileNotFoundError(
                            f"Object mask not found for '{cat}/{base_img}'. "
                            f"Tried:\n  - {cand[0]}\n  - {cand[1]}\n"
                            f"Check --mask_root='{mask_root}' and file names."
                        )
                    try:
                        if os.path.getsize(obj_path) == 0:
                            raise OSError("file size is 0")
                        with Image.open(obj_path) as _im_chk:
                            _im_chk.verify()
                    except Exception as e:
                        _log(dbg_fh,
                             f"[ERROR-MASK][{cat}] Invalid mask file: {obj_path} | {e}")
                        raise FileNotFoundError(
                            f"Invalid or corrupted mask file: {obj_path} ({e})"
                        )
                    # ────────────────────────────────────────────────────────

                    # >>> FIX: mask resize same as image (keep_ratio+pad)
                    # 원본 마스크 크기/비율을 보존한 채로 정사각형 패딩하여 img_size×img_size로 만듭니다. (default: 512×512)
                    obj_pil_orig = Image.open(obj_path).convert("L")
                    obj_pil_resz = resize(obj_pil_orig, img_size, True, True)
                    obj_np = np.array(obj_pil_resz)

                    if (obj_np > 127).sum() == 0:
                        batch_empty_mask += 1
                        obj_masks.append(None); imgs_pil.append(None)
                        continue

                    obj_masks.append(torch.from_numpy(obj_np > 127).to(device))
                    imgs_pil.append(
                        resize(Image.open(npath).convert("RGB"),
                               img_size, True, True))

                # skip empty batch
                valid_idx = [i for i,m in enumerate(obj_masks) if m is not None]
                if not valid_idx:
                    cls_batches_with_zero_valid += 1
                    _log(dbg_fh, f"[BATCH][{cat}/{dcls}] b0={b0} | valid=0 | missing_mask={batch_missing_mask} | empty_mask={batch_empty_mask}")
                    continue
                else:
                    _log(dbg_fh, f"[BATCH][{cat}/{dcls}] b0={b0} | valid={len(valid_idx)}/{len(npaths_batch)} | missing_mask={batch_missing_mask} | empty_mask={batch_empty_mask}")

                # ─ extract normal features for the batch
                imgs_feat = torch.cat(
                    [get_processed_features(imgs_pil[i]) for i in valid_idx]
                )                                                    # [Bv,1,C,h,w]
                feat_norm_up = F.interpolate(
                    imgs_feat, (img_size, img_size),
                    mode="bilinear", align_corners=False
                ).squeeze(1)                                         # [Bv,C,H,W]
                feat_norm_up = F.normalize(feat_norm_up, dim=1)

                # ─ K×Bv cosine similarity maps at once
                sim_up_all   = torch.einsum("kc,bchw->kbhw", vec_up,  feat_norm_up)
                sim_dn_all   = torch.einsum("kc,bchw->kbhw", vec_dn,  feat_norm_up)
                sim_ctr_all  = torch.einsum("kc,bchw->kbhw", vec_ctr, feat_norm_up)

                # ─ per-normal / per-component loop (much lighter)
                for local_idx, b in enumerate(valid_idx):
                    obj_mask = obj_masks[b]
                    img_norm = imgs_pil[b]
                    bn       = base_imgs[b]

                    for k in range(K):
                        sim_up  = sim_up_all[k, local_idx]
                        sim_dn  = sim_dn_all[k, local_idx]
                        c_map   = sim_ctr_all[k, local_idx]

                        up_idx = sim_up.view(-1).argmax()
                        dn_idx = sim_dn.view(-1).argmax()
                        up_pt  = (int(up_idx % img_size), int(up_idx // img_size))
                        dn_pt  = (int(dn_idx % img_size), int(dn_idx // img_size))

                        line_np = np.zeros((img_size, img_size), dtype=np.uint8)
                        cv2.line(line_np, up_pt, dn_pt, 1, thickness=3)
                        line_mask = torch.from_numpy(line_np).to(device=device,
                                                                 dtype=torch.bool)

                        inter_mask = line_mask & obj_mask
                        if inter_mask.sum() == 0:
                            cls_no_intersections += 1
                            continue

                        masked = c_map.clone()
                        masked[~inter_mask] = -1e4
                        best_idx = masked.view(-1).argmax()
                        bx_final = int(best_idx % img_size)
                        by_final = int(best_idx // img_size)

                        # ─ ASSERT: best point must lie inside obj_mask
                        assert bool(obj_mask[by_final, bx_final].item()), \
                            (f"Chosen point ({bx_final},{by_final}) is outside obj_mask. "
                             f"cat={cat}, cls={dcls}, img={bn}, k={k}")

                        # ─ visualization (original)
                        ann  = img_norm.copy()
                        draw = ImageDraw.Draw(ann)
                        for p in [up_pt, dn_pt]:
                            draw.ellipse([(p[0]-4, p[1]-4),
                                          (p[0]+4, p[1]+4)],
                                         fill="blue", outline="blue")
                        draw.ellipse([(bx_final-5, by_final-5),
                                      (bx_final+5, by_final+5)],
                                     fill="red", outline="red")
                        vis_name = f"{cat}_{dcls}__{bn}_cmp{k}_best.png"
                        ann.save(os.path.join(viz_dir, vis_name))

                        # ─ visualization (overlay: mask contour in green + line in blue)
                        ann_cv = cv2.cvtColor(np.array(img_norm), cv2.COLOR_RGB2BGR)
                        obj_np_vis = (obj_mask.detach().to("cpu").numpy().astype(np.uint8) * 255)
                        contours, _ = cv2.findContours(obj_np_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(ann_cv, contours, -1, (0, 255, 0), 1)    # green
                        cv2.line(ann_cv, up_pt, dn_pt, (255, 0, 0), 1)              # blue line

                        ann_overlay = Image.fromarray(cv2.cvtColor(ann_cv, cv2.COLOR_BGR2RGB))
                        draw_ov = ImageDraw.Draw(ann_overlay)
                        for p in [up_pt, dn_pt]:
                            draw_ov.ellipse([(p[0]-4, p[1]-4),
                                             (p[0]+4, p[1]+4)],
                                            fill="blue", outline="blue")
                        draw_ov.ellipse([(bx_final-5, by_final-5),
                                         (bx_final+5, by_final+5)],
                                        fill="red", outline="red")

                        overlay_name = f"{cat}_{dcls}__{bn}_cmp{k}_best_overlay.png"
                        ann_overlay.save(os.path.join(overlay_dir, overlay_name))

                        defect_img, region_idx = defect_ids[k]
                        result[cat][dcls].append({
                            "normal_img": os.path.basename(npaths_batch[b]),
                            "best_x": bx_final,
                            "best_y": by_final,
                            "defect_img": defect_img,
                            "region_idx": region_idx
                        })
                        cls_results += 1

                        # ─ periodic JSON saving
                        if len(result[cat][dcls]) % save_every == 0:
                            with open(out_json_path, "w", encoding="utf-8") as jf:
                                json.dump(result, jf, indent=4)
                                jf.flush(); os.fsync(jf.fileno())

                torch.cuda.empty_cache(); gc.collect()

            # per-class summary
            grand_total_results += cls_results
            grand_no_intersections += cls_no_intersections
            _log(dbg_fh,
                 f"[SUM][{cat}/{dcls}] K={K} | results={cls_results} | "
                 f"no_intersections={cls_no_intersections} | "
                 f"batches_zero_valid={cls_batches_with_zero_valid} | "
                 f"time={time.time()-class_start:.2f}s")

        _log(dbg_fh, f"[CAT-END] {cat} | time={time.time()-cat_start:.2f}s")

    # final save
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    _log(dbg_fh, f"\n[FINAL] JSON saved to {out_json_path}")
    _log(dbg_fh, f"[FINAL] Visualizations saved to {viz_dir}")
    _log(dbg_fh, f"[FINAL] Line visualizations saved to {line_dir}")
    _log(dbg_fh, f"[FINAL] Overlay visualizations saved to {overlay_dir}")
    _log(dbg_fh, f"[TOTAL] components(K)={grand_total_components} | "
                 f"results={grand_total_results} | "
                 f"no_intersections={grand_no_intersections} | "
                 f"elapsed={time.time()-global_start:.2f}s")
    dbg_fh.close()


###############################################################################
# 5) Entry point
###############################################################################
if __name__ == "__main__":
    generate_correspondence_json(
        mvtecad_dir=args.mvtecad_dir,
        categories=args.categories,
        img_size=args.img_size,  
        out_dir=args.out_dir,
        mask_root=args.mask_root,
    )

