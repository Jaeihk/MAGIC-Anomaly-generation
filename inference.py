import argparse, os, random, re, json, types
import torch, numpy as np, cv2
from PIL import Image
from magic_ddim import DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_dynamic_anomaly_strength_guidance_scheduler_up import \
     StableDiffusionInpaintPipeline_dynamic
import glob
from diffusers.configuration_utils import FrozenDict   # ★ NEW: scheduler config 주입용
from torchvision import transforms

# 고정 해상도 (학습과 동일하게 512)
RESOLUTION = 512

# 학습 때와 동일한 Resize + CenterCrop
image_resize_center_crop = transforms.Compose(
    [
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(RESOLUTION),
    ]
)
mask_resize_center_crop = transforms.Compose(
    [
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(RESOLUTION),
    ]
)

# -------------------------------------------------------------------------
# 1) argument parser + miscellany
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference-time mask-alignment augmentation.")
    # ───────── 기존 인자 유지 ─────────
    parser.add_argument("--defect_json", required=True)
    parser.add_argument("--match_json",  required=True)
    parser.add_argument("--model_ckpt_root", required=True)
    parser.add_argument("--ddim_scheduler_root", required=True)
    parser.add_argument("--categories", nargs='+', default=None)
    parser.add_argument("--blur_factor", type=int, default=0)
    parser.add_argument("--text_noise_scale", type=float, default=0.0)
    parser.add_argument("--output_name", default="./")
    parser.add_argument("--anomaly_strength_min", type=float, default=0.0)
    parser.add_argument("--anomaly_strength_max", type=float, default=0.0)
    parser.add_argument("--anomaly_stop_step", type=int, default=999999)
    parser.add_argument("--eta_mask_stop_step", type=int, default=999999)
    parser.add_argument("--normal_masks", default="./normal_masks")
    parser.add_argument("--mask_dir",    default="./Aug_mask_3_shot")
    parser.add_argument("--base_dir",    default="./mvtecad")
    parser.add_argument("--CAMA",        action="store_true")
    parser.add_argument("--use_random_mask", action="store_true")
    parser.add_argument("--dataset_type", choices=["mvtec_3d", "mvtec","visa","DAGM"],
                        default="mvtec_3d",
                        help="mvtec_3d: MVTEC-3D Anomaly, mvtec: MVTEC-AD 2-D")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM eta (0~1). DDIM 스케줄러에서만 사용되고, 다른 스케줄러는 무시됩니다.")
    parser.add_argument("--eta_mask_max", type=float, default=0.0,
                        help="마스크 내부 eta_max(스케줄 사용 시 상한).")
    parser.add_argument("--eta_mask_min", type=float, default=0.0,
                        help="마스크 내부 eta_min(스케줄 사용 시 하한).")

    # ───────── NEW: eta_mask 스케줄러/가드 하이퍼들(DDIM 내부에 전달) ─────────
    parser.add_argument("--eta_mask_use_schedule", action="store_true",
                        help="True면 DDIM이 내부 스텝 스케줄을 사용(eta_mask 인자는 무시).")
    parser.add_argument("--eta_mask_schedule",
                        choices=["constant","linear_down","linear_up",
                                 "cosine_down","cosine_up",
                                 "poly_down","poly_up",
                                 "exp_down","exp_up",
                                 "sigmoid_down","sigmoid_up"],
                        default="constant")
    parser.add_argument("--eta_mask_power", type=float, default=2.0,
                        help="poly 스케줄 지수 p")
    parser.add_argument("--eta_mask_exp_k", type=float, default=3.0,
                        help="exp 스케줄 상수 k")
    parser.add_argument("--eta_mask_sigmoid_k", type=float, default=8.0,
                        help="sigmoid 스케줄 경사 k")
    parser.add_argument("--eta_mask_segmented", action="store_true",
                    help="양수(허용) 구간에서만 스케줄을 s∈[0,1]로 재매핑(구간 시작=max, 끝=0).")
    parser.add_argument("--eta_mask_guard",
                        choices=["none","clip_to_crit","zero_before_neg"],
                        default="none",
                        help="임계치 가드 모드")
    parser.add_argument("--eta_mask_guard_margin", type=float, default=0.99,
                        help="임계치 대비 여유(예: 0.99)")

    # ───────── 기존 CFG 스칼라 인자 (기본값 유지) ─────────
    parser.add_argument("--guidance_scale_inside", type=float, default=None,
                        help="마스크 안쪽 guidance scale (미지정 시 pipeline의 global guidance_scale 사용)")
    parser.add_argument("--guidance_scale_outside", type=float, default=None,
                        help="마스크 바깥 guidance scale (미지정 시 pipeline의 global guidance_scale 사용)")

    # ───────── NEW: guidance_scale_inside 스케줄링/샘플링 컨트롤 ─────────
    parser.add_argument("--gsi_use_schedule", action="store_true",
                        help="True면 guidance_scale_inside를 스텝별 스케줄(max→min)로 사용")
    parser.add_argument("--gsi_schedule",
                        choices=["constant","linear","linear_down","cosine","cosine_down",
                                 "poly","poly_down","exp","exp_down","sigmoid","sigmoid_down"],
                        default="linear",
                        help="선택 시 max→min으로 감소하는 스케줄")
    parser.add_argument("--gsi_min", type=float, default=None, help="guidance_scale_inside 하한")
    parser.add_argument("--gsi_max", type=float, default=None, help="guidance_scale_inside 상한")
    parser.add_argument("--gsi_power", type=float, default=2.0, help="poly 스케줄 지수")
    parser.add_argument("--gsi_exp_k", type=float, default=3.0, help="exp 스케줄 k")
    parser.add_argument("--gsi_sigmoid_k", type=float, default=8.0, help="sigmoid 스케줄 k")
    parser.add_argument("--gsi_sample_per_step", action="store_true",
                        help="스케줄을 쓰지 않는 경우 [gsi_min,gsi_max]에서 스텝별 샘플링")

    return parser.parse_args()

args = parse_args()


def extract_number_from_filename(fname):
    m = re.search(r"\d+", fname)
    return int(m.group()) if m else float("inf")


def monkey_patch_encode_prompt(pipe):
    old_encode = pipe.encode_prompt
    def new_encode_prompt(self, prompt, device, num_images_per_prompt,
                          do_classifier_free_guidance, negative_prompt=None,
                          prompt_embeds=None, negative_prompt_embeds=None,
                          lora_scale=None, clip_skip=None):
        prompt_embeds, neg_embeds = old_encode(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance,
            negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale, clip_skip=clip_skip)
        if getattr(self, "text_noise_scale", 0.0) > 0.0:
            s = self.text_noise_scale
            if do_classifier_free_guidance:
                half = prompt_embeds.shape[0] // 2
                uncond, cond = prompt_embeds[:half], prompt_embeds[half:]
                cond += torch.randn_like(cond) * s
                prompt_embeds = torch.cat([uncond, cond], 0)
            else:
                prompt_embeds += torch.randn_like(prompt_embeds) * s
        return prompt_embeds, neg_embeds
    pipe.encode_prompt = types.MethodType(new_encode_prompt, pipe)


def inpaint(pipe, image, prompt, mask=None, n_samples=4, device="cuda",
            blur_factor=0,
            anomaly_strength=0.0, anomaly_stop_step=999999, eta_mask_stop_step=999999,
            eta=0.0, eta_mask=0.0,
            guidance_scale_inside=None, guidance_scale_outside=None,
            # NEW: inside 스케줄/샘플링 옵션 전달
            gsi_use_schedule=False, gsi_schedule="linear",
            gsi_min=None, gsi_max=None,
            gsi_power=2.0, gsi_exp_k=3.0, gsi_sigmoid_k=8.0,
            gsi_sample_per_step=False):
    from PIL import Image as PilImage
    if isinstance(image, str):
        image_pil = PilImage.open(image).convert("RGB")
    else:
        image_pil = image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(mask, str):
        mask_pil = PilImage.open(mask).convert("RGB")
    else:
        mask_pil = mask.convert("RGB") if mask.mode != "RGB" else mask

    # (프로젝트에 이미 구현돼 있다면) 블러 적용
    mask_pil = pipe.mask_processor.blur(mask_pil, blur_factor=blur_factor)

    return pipe(
        prompt=[prompt]*n_samples,
        image=image_pil,
        mask_image=mask_pil,
        anomaly_strength=anomaly_strength,
        anomaly_stop_step=anomaly_stop_step,
        eta_mask_stop_step=eta_mask_stop_step,
        use_random_mask=args.use_random_mask,
        eta=eta,
        eta_mask=eta_mask,

        # 기존/NEW 인자 전달
        guidance_scale_inside=guidance_scale_inside,
        guidance_scale_outside=guidance_scale_outside,

        # NEW: inside 스케줄/샘플링 파라미터
        gsi_use_schedule=gsi_use_schedule,
        gsi_schedule=gsi_schedule,
        guidance_scale_inside_min=gsi_min,
        guidance_scale_inside_max=gsi_max,
        gsi_power=gsi_power,
        gsi_exp_k=gsi_exp_k,
        gsi_sigmoid_k=gsi_sigmoid_k,
        gsi_sample_per_step=gsi_sample_per_step,
    ).images


def get_random_image(img_dir):
    imgs = [f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not imgs:
        raise ValueError(f"No images in {img_dir}")
    return os.path.join(img_dir, random.choice(imgs))


def load_object_mask(category, normal_img_path, normal_masks_dir):
    cat_dir = os.path.join(normal_masks_dir, category)
    base = os.path.splitext(os.path.basename(normal_img_path))[0]

    cand_dirs = [
        os.path.join(cat_dir, "train", "masks"),
        os.path.join(cat_dir, "masks"),
        cat_dir,
    ]

    candidates = []
    for d in cand_dirs:
        if not os.path.isdir(d):
            continue
        candidates.append(os.path.join(d, f"{base}_mask.png"))
        candidates.extend(sorted(glob.glob(os.path.join(d, f"{base}_mask_*.png"))))
        candidates.append(os.path.join(d, f"{base}.png"))
        candidates.append(os.path.join(d, "mask.png"))
    if not candidates:
        print(f"[load_object_mask] No candidate mask files found for: {normal_img_path}")

    seen = set()
    ordered_paths = []
    for p in candidates:
        if p not in seen and os.path.exists(p):
            seen.add(p)
            ordered_paths.append(p)

    for mask_path in ordered_paths:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return (m > 127).astype(np.uint8)

    return None

def debug_save_masks(original_mask_bin, min_x, min_y, max_x, max_y,
                     shifted_mask_bin, debug_save_path):
    H, W = original_mask_bin.shape
    left = np.zeros((H, W, 3), np.uint8)
    left[original_mask_bin > 0] = (255, 255, 255)
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(original_mask_bin, 8)
    for lbl in range(1, n_lbl):
        x, y, bw, bh, _ = stats[lbl]
        if bw and bh:
            cv2.rectangle(left, (x, y), (x + bw - 1, y + bh - 1), (0, 0, 255), 2)

    right = np.zeros((H, W, 3), np.uint8)
    right[shifted_mask_bin > 0] = (255, 255, 255)
    cv2.imwrite(debug_save_path, np.concatenate([left, right], axis=1))

###############################################################################
# 3) CAMA: Context-Aware Mask Alignment
###############################################################################
def CAMA(
    class_val,
    code_mask_bin,
    obj_mask_np,
    normal_image_path,
    category,
    defect_class,
    defect_data,
    match_data,
    debug_save_dir=None,
    debug_name=None,
):
    """
    Return (final_mask, first_best_x, first_best_y, is_shifted)
    """
    H, W = code_mask_bin.shape
    base_normal = os.path.basename(normal_image_path)

    # ───────── ① Collect coordinates per defect_img ─────────
    by_defect = {}
    for it in match_data.get(category, {}).get(defect_class, []):
        if it["normal_img"] != base_normal:
            continue
        # 이제 JSON의 best_x, best_y가 이미 512x512 기준이므로 그대로 사용
        by_defect.setdefault(it["defect_img"], []).append((it["best_x"], it["best_y"]))

    if not by_defect:
        fallback = cv2.bitwise_and(code_mask_bin, obj_mask_np) if class_val == 0 else code_mask_bin
        if debug_save_dir and debug_name:
            debug_save_masks(
                code_mask_bin, 0, 0, 0, 0, fallback,
                os.path.join(debug_save_dir, f"{debug_name}_fallback.jpg")
            )
        return fallback, -1, -1, False

    # ───────── ② Randomly choose one defect_img ─────────
    chosen_defect, coords_all = random.choice(list(by_defect.items()))
    # coords_all: [(best_x, best_y), …]

    # ───────── ③ Extract components ─────────
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(code_mask_bin, 8)
    comps = list(range(1, n_lbl))  # 0 is background
    if not comps:  # mask is empty
        return code_mask_bin, -1, -1, False

    n_comp = len(comps)
    n_coords = len(coords_all)

    # ───────── ④ Match coordinate list size to the number of components ─────────
    def rand_point_inside(mask):
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            # If obj_mask is empty, sample from the whole image (이미 512 기준)
            return random.randint(0, W - 1), random.randint(0, H - 1)
        idx = random.randrange(ys.size)
        # 좌표도 그대로 (512 기준)
        return int(xs[idx]), int(ys[idx])

    if n_coords >= n_comp:
        coords = random.sample(coords_all, n_comp)
    else:
        coords = list(coords_all)
        # Fill the shortage with random coordinates
        for _ in range(n_comp - n_coords):
            rx, ry = rand_point_inside(obj_mask_np if class_val == 0 else np.ones_like(obj_mask_np))
            coords.append((rx, ry))

    # Now comps and coords have the same length (n_comp)
    target_pairs = list(zip(comps, coords))

    shifted = np.zeros_like(code_mask_bin, np.uint8)
    for lbl, (best_x, best_y) in target_pairs:
        # JSON과 랜덤 모두 512 좌표이므로 그대로 사용
        best_x = int(round(best_x))
        best_y = int(round(best_y))

        # If the component already contains the best point, keep it as is
        if 0 <= best_x < W and 0 <= best_y < H and code_mask_bin[best_y, best_x]:
            shifted |= (lbl_map == lbl).astype(np.uint8)
            continue

        x, y, bw, bh, _ = stats[lbl]
        if bw == 0 or bh == 0:
            continue

        crop = (lbl_map[y:y + bh, x:x + bw] == lbl).astype(np.uint8)

        # Simple translation without center correction
        tx = best_x - bw // 2
        ty = best_y - bh // 2
        for r in range(bh):
            for c in range(bw):
                if crop[r, c]:
                    yy = ty + r
                    xx = tx + c
                    if 0 <= xx < W and 0 <= yy < H:
                        shifted[yy, xx] = 1

    final = cv2.bitwise_and(shifted, obj_mask_np) if class_val == 0 else shifted

    if debug_save_dir and debug_name:
        ys, xs = np.where(code_mask_bin)
        debug_save_masks(
            code_mask_bin,
            xs.min() if xs.size else 0, ys.min() if ys.size else 0,
            xs.max() if xs.size else 0, ys.max() if ys.size else 0,
            final,
            os.path.join(debug_save_dir, f"{debug_name}.jpg")
        )

    first_best = coords[0]
    return final, first_best[0], first_best[1], True



def main():
    with open(args.defect_json, "r", encoding="utf-8") as f:
        defect_data = json.load(f)
    with open(args.match_json, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    if args.dataset_type == "mvtec_3d":
        default_cats = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                        'foam', 'peach', 'potato', 'rope', 'tire']
    elif args.dataset_type == "visa":
        default_cats = ['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum']
    elif args.dataset_type == "DAGM":
        default_cats = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    else:        
        default_cats = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                        'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    categories = args.categories or default_cats

    for category in categories:
        device = "cuda"

        if args.dataset_type == "mvtec_3d":
            gt_path     = os.path.join(args.base_dir, category, "test")
            normal_root = os.path.join(args.base_dir, category, "train", "good", "rgb")
        else:
            gt_path     = os.path.join(args.base_dir, category, "ground_truth")
            normal_root = os.path.join(args.base_dir, category, "train", "good")

        if not os.path.exists(gt_path):
            print(f"[WARN] ground_truth path not found: {gt_path}")
            continue
        if not os.path.exists(normal_root):
            print(f"[WARN] normal image path not found: {normal_root}")
            continue

        defect_classes = [d for d in os.listdir(gt_path)
                          if os.path.isdir(os.path.join(gt_path, d))
                          and d != "good"]
        for defect_class in defect_classes:
            if defect_class not in defect_data.get(category, {}):
                print(f"[WARN] {defect_class} not in defect_json → skip")
                continue
            class_val = defect_data[category][defect_class]
            print(f"Category={category}, Defect={defect_class}, class_val={class_val}")

            mask_root = os.path.join(args.mask_dir, category, defect_class)
            if not os.path.exists(mask_root):
                print(f"[WARN] {mask_root} not found → skip")
                continue

            ckpt_root = os.path.join(args.model_ckpt_root, category, defect_class)
            if not os.path.exists(ckpt_root):
                print(f"[WARN] checkpoint absent: {ckpt_root} → skip")
                continue

            pipe = StableDiffusionInpaintPipeline_dynamic.from_pretrained(
                ckpt_root, torch_dtype=torch.float16)
            # --- DDIM 스케줄러 로드
            pipe.scheduler = DDIMScheduler.from_pretrained(args.ddim_scheduler_root)

            # --- ★ parser에서 받은 eta_mask 스케줄/가드 하이퍼들을 DDIM config에 주입
            sched = pipe.scheduler
            new_cfg = dict(sched.config)
            new_cfg.update({
                "eta_mask_use_schedule": bool(args.eta_mask_use_schedule),
                "eta_mask_schedule": args.eta_mask_schedule,
                "eta_mask_min": float(args.eta_mask_min),
                "eta_mask_max": float(args.eta_mask_max),
                "eta_mask_power": float(args.eta_mask_power),
                "eta_mask_exp_k": float(args.eta_mask_exp_k),
                "eta_mask_sigmoid_k": float(args.eta_mask_sigmoid_k),
                "eta_mask_guard": args.eta_mask_guard,
                "eta_mask_guard_margin": float(args.eta_mask_guard_margin),
                "eta_mask_segmented": bool(args.eta_mask_segmented),
                # 스케줄 on일 때도 stop step을 존중하기 위해 DDIM에 전달
                "eta_mask_stop_step": int(args.eta_mask_stop_step),
            })
            sched._internal_dict = FrozenDict(new_cfg)

            pipe.text_noise_scale = args.text_noise_scale
            monkey_patch_encode_prompt(pipe)
            pipe.to(device)

            mask_imgs = sorted(
                (os.path.join(mask_root, f) for f in os.listdir(mask_root)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))),
                key=lambda x: extract_number_from_filename(os.path.basename(x)))

            suffix = (f"noise_{args.text_noise_scale}_"
                      f"guidance_scale_{args.gsi_min}_{args.gsi_max}_"
                      + ("align" if args.CAMA else "no_align"))
            save_root = os.path.join(args.output_name, suffix, category, defect_class)
            img_dir  = os.makedirs(os.path.join(save_root, "image"), exist_ok=True) or os.path.join(save_root, "image")
            norm_dir = os.makedirs(os.path.join(save_root, "normal"), exist_ok=True) or os.path.join(save_root, "normal")
            msk_dir  = os.makedirs(os.path.join(save_root, "masks"), exist_ok=True) or os.path.join(save_root, "masks")
            dbg_dir  = os.makedirs(os.path.join(save_root, "debug_mask"), exist_ok=True) or os.path.join(save_root, "debug_mask")

            for idx, mask_path in enumerate(mask_imgs):
                # -----------------------------
                # 1) normal image 512 resize + center crop
                # -----------------------------
                normal_img_path = get_random_image(normal_root)
                normal_img = Image.open(normal_img_path).convert("RGB")
                normal_img = image_resize_center_crop(normal_img)

                # -----------------------------
                # 2) aug mask 512 resize + center crop
                # -----------------------------
                raw_mask_pil = Image.open(mask_path).convert("L")
                raw_mask_pil = mask_resize_center_crop(raw_mask_pil)
                mask_np = (np.array(raw_mask_pil) > 127).astype(np.uint8)

                # -----------------------------
                # 3) object mask 512 resize + center crop (CAMA용)
                # -----------------------------
                obj_mask = load_object_mask(category, normal_img_path, args.normal_masks)
                if obj_mask is None:
                    obj_mask = np.ones_like(mask_np, np.uint8)
                else:
                    obj_mask_pil = Image.fromarray((obj_mask * 255).astype(np.uint8))
                    obj_mask_pil = mask_resize_center_crop(obj_mask_pil)
                    obj_mask = (np.array(obj_mask_pil) > 127).astype(np.uint8)

                # 그래도 혹시 shape 안 맞으면 한번 더 맞춰줌
                if obj_mask.shape != mask_np.shape:
                    obj_mask = cv2.resize(obj_mask, mask_np.shape[::-1], interpolation=cv2.INTER_NEAREST)
                    obj_mask = (obj_mask > 0).astype(np.uint8)

                # -----------------------------
                # 4) CAMA
                # -----------------------------
                if args.CAMA:
                    final_mask, *_ = CAMA(
                        class_val, mask_np, obj_mask, normal_img_path,
                        category, defect_class, defect_data, match_data,
                        debug_save_dir=dbg_dir, debug_name=f"{idx}",
                    )
                else:
                    final_mask = mask_np

                final_mask_pil = Image.fromarray((final_mask * 255).astype(np.uint8))

                a_strength = random.uniform(args.anomaly_strength_min, args.anomaly_strength_max)
                # NOTE: 스케줄 on이면 아래 eta_strength는 DDIM에서 무시됨(내부 스케줄 사용)
                eta_strength = random.uniform(args.eta_mask_min, args.eta_mask_max)

                imgs = inpaint(
                    pipe, normal_img,
                    prompt="a photo of a sks defect",
                    mask=final_mask_pil, n_samples=1, device=device,
                    blur_factor=args.blur_factor,
                    anomaly_strength=a_strength,
                    anomaly_stop_step=args.anomaly_stop_step,
                    eta_mask_stop_step=args.eta_mask_stop_step,
                    eta=args.eta,
                    eta_mask=eta_strength,

                    guidance_scale_inside=(args.guidance_scale_inside if args.guidance_scale_inside is not None else 3.0),
                    guidance_scale_outside=(args.guidance_scale_outside if args.guidance_scale_outside is not None else 7.5),

                    # NEW: inside 스케줄/샘플링 옵션 전달
                    gsi_use_schedule=args.gsi_use_schedule,
                    gsi_schedule=args.gsi_schedule,
                    gsi_min=args.gsi_min,
                    gsi_max=args.gsi_max,
                    gsi_power=args.gsi_power,
                    gsi_exp_k=args.gsi_exp_k,
                    gsi_sigmoid_k=args.gsi_sigmoid_k,
                    gsi_sample_per_step=args.gsi_sample_per_step,
                )

                out = f"{idx}.jpg"
                imgs[0].save(os.path.join(img_dir,  out))
                normal_img = normal_img.convert("RGB")
                normal_img.save(os.path.join(norm_dir, out), format="JPEG")
                final_mask_pil.convert("RGB").save(os.path.join(msk_dir, out))
                print(f"Saved {out}")


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=3 python inference_guidance_scale_scheduler_up_cc.py --model_ckpt_root='./model/dreambooth_batch_4_5000_noise_1.0_0.33' --text_noise_scale 1.0 --defect_json ./datasets/defect_classification_final_mvtec.json --match_json ./datasets/mvtec_multi_region_matching.json --normal_masks ./foreground_mask/mvtec_normal_foreground --output_name ./generated_image/mvtec/guidance_scale/noise_rand_flip_black_flitering_0/cosine_up --mask_dir ./anomaly_masks/mvtec/noise_rand_flip_black_flitering_0_512 --ddim_scheduler_root ./scheduler --dataset_type mvtec --gsi_use_schedule --gsi_schedule cosine --gsi_min 4.0 --gsi_max 7.5 --guidance_scale_outside 7.5 --CAMA --base_dir ./data/mvtecad
