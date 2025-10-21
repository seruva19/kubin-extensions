import torch
import numpy as np
import cv2
import json
import yaml
import decord
import requests
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
from fastvqa.models import DiViDeAddEvaluator


QUALITY_WEIGHTS = {
    "perceptual": {
        "technical": 0.18,
        "aesthetic": 0.27,
        "fastvqa": 0.45,
        "temporal": 0.10,
    },
    "technical": {
        "technical": 0.45,
        "aesthetic": 0.18,
        "fastvqa": 0.27,
        "temporal": 0.10,
    },
    "aesthetic": {
        "technical": 0.23,
        "aesthetic": 0.45,
        "fastvqa": 0.22,
        "temporal": 0.10,
    },
    "balanced": {
        "technical": 0.23,
        "aesthetic": 0.32,
        "fastvqa": 0.35,
        "temporal": 0.10,
    },
    "anime": {"technical": 0.14, "aesthetic": 0.50, "fastvqa": 0.26, "temporal": 0.10},
    "adaptive": {
        "technical": 0.25,
        "aesthetic": 0.30,
        "fastvqa": 0.35,
        "temporal": 0.10,
    },
}

TEMPORAL_WEIGHTS = {
    "consistency": 0.4,
    "motion_smoothness": 0.3,
    "quality_stability": 0.3,
}

FASTVQA_MODEL_URLS = {
    "FAST_VQA_3D_1_1.pth": "https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST_VQA_3D_1_1.pth",
    "FAST_VQA_B_1_4.pth": "https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST_VQA_B_1_4.pth",
    "FAST_VQA_M_1_4.pth": "https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST_VQA_M_1_4.pth",
}


def normalize_checkpoint_filename(filename):
    # Config files use asterisk, but actual files use underscore
    return filename.replace("*", "_")


def resize_frame_maintain_aspect(frame, min_dim):
    """
    Resize frame to have the specified minimum dimension while maintaining aspect ratio.

    Args:
        frame: Input frame as numpy array (H, W, C)
        min_dim: Target minimum dimension (int). The smaller of width/height will be set to this value.

    Returns:
        Resized frame maintaining aspect ratio
    """
    if min_dim <= 0:
        return frame

    h, w = frame.shape[:2]

    # If frame is already smaller than min_dim in both dimensions, don't upscale
    if h <= min_dim and w <= min_dim:
        return frame

    # Calculate scaling factor to make the smaller dimension equal to min_dim
    if h < w:
        # Height is smaller, scale so height = min_dim
        scale = min_dim / h
    else:
        # Width is smaller or equal, scale so width = min_dim
        scale = min_dim / w

    # Only downscale if scale < 1 (avoid upscaling)
    if scale >= 1.0:
        return frame

    new_h = int(h * scale)
    new_w = int(w * scale)

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def download_fastvqa_checkpoint(filename, destination_dir):
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    normalized_filename = normalize_checkpoint_filename(filename)
    destination_path = destination_dir / normalized_filename

    if destination_path.exists():
        print(f"[FastVQA] {normalized_filename} already exists")
        return destination_path

    url = FASTVQA_MODEL_URLS.get(normalized_filename)
    if not url:
        raise ValueError(f"Unknown model file: {normalized_filename}")

    print(f"[FastVQA] Downloading {normalized_filename}...")
    print(f"[FastVQA] URL: {url}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(destination_path, "wb") as f, tqdm(
            desc=f"Downloading {normalized_filename}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"[FastVQA] Successfully downloaded {normalized_filename}")
        return destination_path

    except Exception as e:
        if destination_path.exists():
            destination_path.unlink()
        raise RuntimeError(f"Failed to download {normalized_filename}: {e}") from e


class QualityScorer:
    def __init__(self, kubin):
        self.kubin = kubin
        self.aesthetic_model = None
        self.aesthetic_preprocessor = None
        self.fastvqa_evaluator = None
        self.device = None
        self.quality_cache = {}
        self.quality_strategy = "balanced"

    def load_aesthetic_model(self, device="cuda"):
        print("[Aesthetic] Loading aesthetic predictor v2.5...")

        self.device = device
        self.aesthetic_model, self.aesthetic_preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        self.aesthetic_model = self.aesthetic_model.to(torch.bfloat16).to(device)
        self.aesthetic_model.eval()

        print("[Aesthetic] Model loaded successfully")
        return True

    def load_fastvqa_model(self, model_type="FasterVQA", device="cuda"):
        print(f"[FastVQA] Loading model: {model_type}...")

        self.device = device

        opts = {
            "FasterVQA": "./options/fast/f3dvqa-b.yml",
            "FasterVQA-MS": "./options/fast/fastervqa-ms.yml",
            "FasterVQA-MT": "./options/fast/fastervqa-mt.yml",
            "FAST-VQA": "./options/fast/fast-b.yml",
            "FAST-VQA-M": "./options/fast/fast-m.yml",
        }

        opt_path = Path(__file__).parent.parent / opts[model_type]

        with open(opt_path, "r") as f:
            opt = yaml.safe_load(f)

        self.fastvqa_evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(device)

        # Load weights - construct path relative to extension directory
        # Normalize the path (config files may use asterisks)
        weights_path_str = str(opt["test_load_path"]).replace("*", "_")
        weights_path = Path(__file__).parent.parent / weights_path_str

        # Auto-download checkpoint if missing
        if not weights_path.exists():
            checkpoint_filename = weights_path.name
            pretrained_dir = weights_path.parent

            print(f"[FastVQA] Checkpoint not found: {weights_path.name}")
            print(f"[FastVQA] Attempting to download from GitHub releases...")

            try:
                download_fastvqa_checkpoint(checkpoint_filename, pretrained_dir)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download checkpoint {checkpoint_filename}. "
                    f"Please manually download from: "
                    f"https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/tag/v2.0.0"
                ) from e

        self.fastvqa_evaluator.load_state_dict(
            torch.load(str(weights_path), map_location=device)["state_dict"]
        )
        self.fastvqa_evaluator = self.fastvqa_evaluator.to(self.device)

        self.fastvqa_mean_std = {
            "FasterVQA": (0.14759505, 0.03613452),
            "FasterVQA-MS": (0.15218826, 0.03230298),
            "FasterVQA-MT": (0.14699507, 0.036453716),
            "FAST-VQA": (-0.110198185, 0.04178565),
            "FAST-VQA-M": (0.023889644, 0.030781006),
        }
        self.fastvqa_model_type = model_type
        self.fastvqa_opt = opt

        print("[FastVQA] Model loaded successfully")
        return True

    def extract_frames(self, video_path, num_frames=8, min_dimension=None):
        num_frames = int(num_frames)
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return None

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply resizing if specified (maintain aspect ratio)
                if min_dimension and min_dimension > 0:
                    frame = resize_frame_maintain_aspect(frame, min_dimension)
                frames.append(frame)

        cap.release()
        return frames if frames else None

    def calculate_motion_score(
        self, video_path, step=2, threshold_px=0.5, stabilize=True
    ):
        """
        Calculate motion score using optical flow with optional stabilization.

        Returns dict with:
        - scene_motion_mean: median optical flow magnitude
        - camera_motion_mean: median camera motion (if stabilize=True)
        - active_ratio: fraction of pixels with significant motion
        - label: "static", "mixed", or "dynamic"
        """
        step = int(step)
        cap = cv2.VideoCapture(str(video_path))

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return None

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 1)

        # For stabilization
        orb = cv2.ORB_create(1000) if stabilize else None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if stabilize else None

        flow_mags = []
        active_ratios = []
        camera_motions = []

        frame_count = 0
        while True:
            for _ in range(step):
                ret, frame = cap.read()
                if not ret:
                    break
            if not ret:
                break

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 1)

            # Stabilization (camera motion compensation)
            camera_motion = 0.0
            stabilized_curr = curr_gray

            if stabilize:
                try:
                    kp1, des1 = orb.detectAndCompute(prev_gray, None)
                    kp2, des2 = orb.detectAndCompute(curr_gray, None)

                    if (
                        des1 is not None
                        and des2 is not None
                        and len(kp1) > 8
                        and len(kp2) > 8
                    ):
                        matches = bf.match(des1, des2)
                        if len(matches) > 10:
                            matches = sorted(matches, key=lambda x: x.distance)[:200]
                            pts1 = np.float32(
                                [kp1[m.queryIdx].pt for m in matches]
                            ).reshape(-1, 1, 2)
                            pts2 = np.float32(
                                [kp2[m.trainIdx].pt for m in matches]
                            ).reshape(-1, 1, 2)

                            H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
                            if H is not None:
                                h, w = prev_gray.shape
                                stabilized_curr = cv2.warpPerspective(
                                    curr_gray, H, (w, h)
                                )

                                # Estimate camera motion from homography
                                corners = np.float32(
                                    [[0, 0], [w, 0], [w, h], [0, h]]
                                ).reshape(-1, 1, 2)
                                transformed = cv2.perspectiveTransform(corners, H)
                                camera_motion = np.mean(
                                    np.linalg.norm(transformed - corners, axis=2)
                                )
                except:
                    pass

            # Calculate optical flow
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    stabilized_curr,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )

                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flow_mags.append(np.mean(mag))
                active_ratios.append(np.mean(mag > threshold_px))
                camera_motions.append(camera_motion)

            except:
                pass

            prev_gray = curr_gray
            frame_count += 1

        cap.release()

        if not flow_mags:
            return {
                "scene_motion_mean": 0.0,
                "camera_motion_mean": 0.0,
                "active_ratio": 0.0,
                "label": "static",
            }

        # Use median to be robust to outliers
        scene_motion = float(np.median(flow_mags))
        active = float(np.median(active_ratios))
        camera = float(np.median(camera_motions)) if camera_motions else 0.0

        # Classify motion
        if scene_motion > 0.4 or active > 0.08:
            label = "dynamic"
        elif scene_motion > 0.15 or active > 0.02:
            label = "mixed"
        else:
            label = "static"

        return {
            "scene_motion_mean": scene_motion,
            "camera_motion_mean": camera,
            "active_ratio": active,
            "label": label,
        }

    def calculate_technical_quality(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Sharpness
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Contrast
        contrast = gray.std()

        # Brightness
        brightness = gray.mean()
        brightness_score = 1.0 - abs(brightness - 127.5) / 127.5

        # Noise level
        noise_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        noise_level = noise_sobel.var()
        noise_score = 1.0 - min(noise_level / 200.0, 1.0)

        # Color saturation
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].mean() / 255.0
        saturation_score = min(saturation * 2.0, 1.0)

        # Compression artifacts
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        artifact_score = 1.0 - min(edge_density * 15.0, 1.0)

        # Resolution quality
        h, w = gray.shape
        resolution_score = min(max(h, w) / 720.0, 1.0)

        # Normalize and combine
        sharpness_norm = min(sharpness / 1000.0, 1.0)
        contrast_norm = min(contrast / 64.0, 1.0)

        technical_score = (
            sharpness_norm * 0.25
            + contrast_norm * 0.20
            + brightness_score * 0.15
            + noise_score * 0.15
            + saturation_score * 0.10
            + artifact_score * 0.10
            + resolution_score * 0.05
        )

        return technical_score

    @torch.no_grad()
    def calculate_aesthetic_score(self, frame):
        if not self.aesthetic_model or not self.aesthetic_preprocessor:
            raise RuntimeError("Aesthetic model not loaded")

        pil_frame = Image.fromarray(frame).convert("RGB")
        pixel_values = self.aesthetic_preprocessor(
            images=pil_frame, return_tensors="pt"
        ).pixel_values

        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)

        with torch.inference_mode():
            score = (
                self.aesthetic_model(pixel_values)
                .logits.squeeze()
                .float()
                .cpu()
                .numpy()
            )
            if hasattr(score, "item"):
                score = score.item()

        # Normalize aesthetic score from 1-10 range to 0-1 range
        # Aesthetic predictor outputs scores in [1, 10] range
        normalized_score = (score - 1.0) / 9.0  # Convert [1,10] to [0,1]

        # Clamp to valid range [0, 1] to handle any edge cases
        normalized_score = max(0.0, min(1.0, float(normalized_score)))

        return normalized_score

    def assess_temporal_quality(self, frames):
        if len(frames) < 2:
            return 0.5

        try:
            gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]

            # Frame-to-frame consistency
            frame_diffs = []
            for i in range(len(gray_frames) - 1):
                diff = cv2.absdiff(gray_frames[i], gray_frames[i + 1])
                frame_diffs.append(np.mean(diff))

            consistency_score = 1.0 - min(
                np.std(frame_diffs) / (np.mean(frame_diffs) + 1e-6), 1.0
            )

            # Motion smoothness
            optical_flows = []
            for i in range(len(gray_frames) - 1):
                flow = cv2.calcOpticalFlowPyrLK(
                    gray_frames[i],
                    gray_frames[i + 1],
                    np.array([[100, 100]], dtype=np.float32).reshape(-1, 1, 2),
                    None,
                )[0]
                if flow is not None and len(flow) > 0:
                    optical_flows.append(np.std(flow))

            if optical_flows:
                motion_smoothness = 1.0 - min(
                    np.std(optical_flows) / (np.mean(optical_flows) + 1e-6), 1.0
                )
            else:
                motion_smoothness = 0.5

            # Quality stability
            quality_scores = [
                self.calculate_technical_quality(frame) for frame in frames
            ]
            quality_stability = 1.0 - min(np.std(quality_scores), 1.0)

            temporal_score = (
                consistency_score * TEMPORAL_WEIGHTS["consistency"]
                + motion_smoothness * TEMPORAL_WEIGHTS["motion_smoothness"]
                + quality_stability * TEMPORAL_WEIGHTS["quality_stability"]
            )

            return temporal_score

        except Exception as e:
            print(f"Error in temporal assessment: {e}")
            return 0.5

    def sigmoid_rescale(self, score):
        """Rescale FAST-VQA score to 0-1 range."""
        mean, std = self.fastvqa_mean_std[self.fastvqa_model_type]
        x = (score - mean) / std
        return 1 / (1 + np.exp(-x))

    @torch.no_grad()
    def calculate_fastvqa_score(self, video_path, min_dimension=None):
        """Calculate video quality score using FAST-VQA."""
        if not self.fastvqa_evaluator:
            raise RuntimeError("FAST-VQA model not loaded")

        prepared = self._prepare_fastvqa_input(video_path, min_dimension)
        if not prepared:
            raise RuntimeError(f"Failed to prepare FAST-VQA input for {video_path}")

        raw_score = self._run_fastvqa_batch([prepared])[0]
        return self.sigmoid_rescale(raw_score)

    def _prepare_fastvqa_input(self, video_path, min_dimension=None):
        if not self.fastvqa_evaluator:
            raise RuntimeError("FAST-VQA model not loaded")

        # Find any validation configuration in the data section
        # Try to find a validation dataset config (any section starting with 'val-')
        val_config = None
        for key in self.fastvqa_opt["data"].keys():
            if key.startswith("val-"):
                val_config = self.fastvqa_opt["data"][key]["args"]
                break

        if val_config is None:
            # Fallback to train config if no validation config found
            val_config = self.fastvqa_opt["data"]["train"]["args"]

        t_data_opt = val_config
        s_data_opt = t_data_opt["sample_types"]

        vsamples = {}
        clip_count = None

        for sample_type, sample_args in s_data_opt.items():
            frame_interval = sample_args.get(
                "frame_interval", t_data_opt.get("frame_interval", 1)
            )

            # Use official FAST-VQA frame sampling method to ensure proper alignment
            # This matches the sampling approach in vqa.py exactly
            import decord
            video_reader = decord.VideoReader(str(video_path))

            # Extract frames using FAST-VQA's official sampling
            frame_interval = sample_args.get(
                "frame_interval", t_data_opt.get("frame_interval", 1)
            )
            if t_data_opt.get("t_frag", 1) > 1:
                from fastvqa.datasets import FragmentSampleFrames
                sampler = FragmentSampleFrames(
                    fsize_t=sample_args["clip_len"] // sample_args.get("t_frag", 1),
                    fragments_t=sample_args.get("t_frag", 1),
                    num_clips=sample_args.get("num_clips", 1),
                    frame_interval=frame_interval,
                )
            else:
                from fastvqa.datasets import SampleFrames
                sampler = SampleFrames(
                    clip_len=sample_args["clip_len"],
                    frame_interval=frame_interval,
                    num_clips=sample_args.get("num_clips", 1),
                )

            frames = sampler(len(video_reader))
            frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}

            # Apply resizing if specified before converting to tensor
            imgs = []
            for idx in frames:
                frame = frame_dict[idx].asnumpy()  # H, W, C, BGR format from decord
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

                # Apply resizing if specified (maintain aspect ratio)
                if min_dimension and min_dimension > 0:
                    frame_rgb = resize_frame_maintain_aspect(frame_rgb, min_dimension)

                imgs.append(torch.from_numpy(frame_rgb))

            # Stack and permute to match official format: (C,T,H,W)
            video = torch.stack(imgs, 0).permute(3, 0, 1, 2)  # Results in (C,T,H,W) format

            sampled_video = get_spatial_fragments(video, **sample_args)

            # Apply official FAST-VQA normalization (0-255 range with original ImageNet values)
            mean = torch.FloatTensor([123.675, 116.28, 103.53])
            std = torch.FloatTensor([58.395, 57.12, 57.375])
            sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(
                3, 0, 1, 2
            )

            num_clips = sample_args.get("num_clips", 1)
            if clip_count is None:
                clip_count = num_clips
            elif clip_count != num_clips:
                raise ValueError("FAST-VQA requires consistent num_clips per sample")

            sampled_video = sampled_video.reshape(
                sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]
            ).transpose(0, 1)
            vsamples[sample_type] = sampled_video.contiguous()

        return {"samples": vsamples, "num_clips": clip_count}

    @torch.no_grad()
    def _run_fastvqa_batch(self, prepared_list):
        if not prepared_list:
            return []

        sample_types = prepared_list[0]["samples"].keys()
        batched_inputs = {}
        clip_counts = [item["num_clips"] for item in prepared_list]

        for sample_type in sample_types:
            batched_inputs[sample_type] = torch.cat(
                [item["samples"][sample_type] for item in prepared_list], dim=0
            ).to(self.device)

        total_clips = batched_inputs[next(iter(batched_inputs))].shape[0]
        scores = self.fastvqa_evaluator(batched_inputs)
        clip_scores = scores.reshape(total_clips, -1).mean(dim=1).detach().cpu()

        results = []
        start_idx = 0
        for clips in clip_counts:
            segment = clip_scores[start_idx : start_idx + clips]
            results.append(segment.mean().item())
            start_idx += clips

        return results

    def adaptive_weight_adjustment(self, tech_avg, tech_std, aes_avg, aes_std, fastvqa_score, base_weights):
        """Adjust weights based on score characteristics and reliability."""
        weights = base_weights.copy()

        # If technical quality is very low, reduce its influence
        if tech_avg < 0.3:
            weights["technical"] *= 0.7
            weights["fastvqa"] *= 1.15

        # If aesthetic scores are inconsistent (high std), reduce aesthetic weight
        if aes_std > 0.2:
            weights["aesthetic"] *= 0.8
            weights["fastvqa"] *= 1.1

        # If FastVQA score is extreme (very low/high), increase its influence
        if fastvqa_score < 0.2 or fastvqa_score > 0.8:
            weights["fastvqa"] *= 1.2
            weights["technical"] *= 0.9
            weights["aesthetic"] *= 0.9

        # Renormalize weights to sum to 1
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights

    def calculate_combined_score(self, tech_score, aes_score, fastvqa_score, weights, tech_std=0, aes_std=0):
        """Advanced score combination with outlier detection and smoothing."""

        # Basic weighted combination
        basic_score = (
            tech_score * weights["technical"]
            + aes_score * weights["aesthetic"]
            + fastvqa_score * weights["fastvqa"]
        )

        # Outlier detection - if one score is drastically different, apply smoothing
        scores = [tech_score, aes_score, fastvqa_score]
        score_mean = np.mean(scores)
        score_std = np.std(scores)

        # If there's high variance between different quality measures, apply conservative smoothing
        if score_std > 0.25:  # High disagreement between metrics
            # Pull extreme scores towards the mean
            smoothed_scores = []
            for score in scores:
                if abs(score - score_mean) > 1.5 * score_std:
                    # Smooth extreme outliers
                    smoothed_score = score_mean + 0.7 * (score - score_mean)
                    smoothed_scores.append(smoothed_score)
                else:
                    smoothed_scores.append(score)

            # Recalculate with smoothed scores
            final_score = (
                smoothed_scores[0] * weights["technical"]
                + smoothed_scores[1] * weights["aesthetic"]
                + smoothed_scores[2] * weights["fastvqa"]
            )

            print(f"[QUALITY] Applied outlier smoothing (std: {score_std:.3f})")
            return final_score

        # Apply confidence weighting based on measurement consistency
        confidence_factor = 1.0

        # Reduce confidence if technical measurements are inconsistent
        if tech_std > 0.15:
            confidence_factor *= 0.95

        # Reduce confidence if aesthetic measurements are inconsistent
        if aes_std > 0.15:
            confidence_factor *= 0.95

        # Conservative adjustment for low confidence
        if confidence_factor < 1.0:
            # Pull score slightly towards neutral (0.5)
            final_score = basic_score * confidence_factor + 0.5 * (1 - confidence_factor)
            print(f"[QUALITY] Applied confidence adjustment (factor: {confidence_factor:.3f})")
            return final_score

        return basic_score

    def assess_video_quality(self, video_path, num_frames=8, include_motion=True, min_dimension=None):
        results, errors = self.assess_video_quality_batch(
            [video_path], num_frames=num_frames, include_motion=include_motion, min_dimension=min_dimension
        )

        if results:
            return results[0][1]

        error_msg = errors.get(video_path, "Unknown error during quality assessment")
        raise RuntimeError(error_msg)

    def assess_video_quality_batch(self, video_paths, num_frames=8, include_motion=True, min_dimension=None):
        num_frames = int(num_frames)
        successful = []
        prepared_fastvqa = []
        errors = {}

        for video_path in video_paths:
            try:
                frames = self.extract_frames(video_path, num_frames, min_dimension)
                if not frames:
                    raise RuntimeError(f"No frames extracted from {video_path}")

                technical_scores = [
                    self.calculate_technical_quality(frame) for frame in frames
                ]
                technical_avg = float(np.mean(technical_scores))
                technical_std = float(np.std(technical_scores))

                aesthetic_scores = [
                    self.calculate_aesthetic_score(frame) for frame in frames
                ]
                aesthetic_avg = float(np.mean(aesthetic_scores))
                aesthetic_std = float(np.std(aesthetic_scores))

                temporal_score = float(self.assess_temporal_quality(frames))

                motion_result = None
                if include_motion:
                    motion_result = self.calculate_motion_score(video_path)

                prepared = self._prepare_fastvqa_input(video_path, min_dimension)
                if not prepared:
                    raise RuntimeError("FAST-VQA preparation returned empty data")

                successful.append(
                    {
                        "path": video_path,
                        "technical_avg": technical_avg,
                        "technical_std": technical_std,
                        "aesthetic_avg": aesthetic_avg,
                        "aesthetic_std": aesthetic_std,
                        "temporal_score": temporal_score,
                        "motion_result": motion_result,
                    }
                )
                prepared_fastvqa.append(prepared)

            except Exception as exc:
                errors[video_path] = str(exc)

        if not successful:
            return [], errors

        raw_scores = self._run_fastvqa_batch(prepared_fastvqa)
        results = []

        for item, raw_score in zip(successful, raw_scores):
            fastvqa_score = self.sigmoid_rescale(raw_score)
            weights = QUALITY_WEIGHTS[self.quality_strategy].copy()

            if self.quality_strategy == "adaptive":
                weights = self.adaptive_weight_adjustment(
                    item["technical_avg"], item["technical_std"],
                    item["aesthetic_avg"], item["aesthetic_std"],
                    fastvqa_score, weights
                )

            final_score = self.calculate_combined_score(
                item["technical_avg"], item["aesthetic_avg"], fastvqa_score,
                weights, item["technical_std"], item["aesthetic_std"]
            )
            final_score = final_score * (1 - weights["temporal"]) + item["temporal_score"] * weights["temporal"]

            result = {
                "final_score": float(final_score),
                "technical": item["technical_avg"],
                "aesthetic": item["aesthetic_avg"],
                "fastvqa": float(fastvqa_score),
                "temporal": item["temporal_score"],
                "weights_used": weights,
                "technical_std": item["technical_std"],
                "aesthetic_std": item["aesthetic_std"],
            }

            if item["motion_result"]:
                result["motion"] = item["motion_result"]

            results.append((item["path"], result))

        return results, errors

    def save_score_file(self, video_path, score_data, extension=".score"):
        video_path = Path(video_path)
        score_file = video_path.parent / (video_path.name + extension)

        with open(score_file, "w", encoding="utf-8") as f:
            json.dump(score_data, f, indent=2)

        return score_file

    def load_score_file(self, video_path, extension=".score"):
        video_path = Path(video_path)
        score_file = video_path.parent / (video_path.name + extension)

        if not score_file.exists():
            return None

        with open(score_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def calculate_quality_scores_per_file(
        self,
        video_dir,
        video_extensions=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
        num_frames=8,
        include_subdirectories=False,
        quality_strategy="balanced",
        score_extension=".score",
        overwrite_existing=False,
        include_motion=True,
        batch_size=1,
        min_dimension=None,
    ):
        num_frames = int(num_frames)
        batch_size = max(1, int(batch_size))
        self.quality_strategy = quality_strategy
        video_dir = Path(video_dir)

        if not video_dir.exists():
            return {"error": f"Directory {video_dir} does not exist"}

        if include_subdirectories:
            all_videos = []
            for ext in video_extensions:
                all_videos.extend(video_dir.rglob(f"*{ext}"))
        else:
            all_videos = []
            for ext in video_extensions:
                all_videos.extend(video_dir.glob(f"*{ext}"))

        print(f"Found {len(all_videos)} videos")

        videos_to_assess = []
        for video in all_videos:
            score_file = video.parent / (video.name + score_extension)
            if overwrite_existing or not score_file.exists():
                videos_to_assess.append(video)

        print(
            f"Assessing {len(videos_to_assess)} videos ({len(all_videos) - len(videos_to_assess)} already have .score files)"
        )

        processed_count = 0
        failed_count = 0

        for idx in tqdm(
            range(0, len(videos_to_assess), batch_size), desc="Quality assessment"
        ):
            batch_videos = videos_to_assess[idx : idx + batch_size]
            results, errors = self.assess_video_quality_batch(
                batch_videos, num_frames=num_frames, include_motion=include_motion, min_dimension=min_dimension
            )

            for video_path, quality_result in results:
                self.save_score_file(video_path, quality_result, score_extension)
                processed_count += 1

            for video_path in batch_videos:
                if video_path in errors:
                    print(f"Failed to assess {video_path}: {errors[video_path]}")
                    failed_count += 1

        return {
            "total_videos": len(all_videos),
            "assessed": processed_count,
            "skipped": len(all_videos) - len(videos_to_assess),
            "failed": failed_count,
        }

    def calculate_quality_scores(
        self,
        video_dir,
        video_extensions=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
        num_frames=8,
        cache_file=None,
        include_subdirectories=False,
        quality_strategy="balanced",
        min_dimension=None,
    ):
        self.quality_strategy = quality_strategy
        video_dir = Path(video_dir)

        if not video_dir.exists():
            return {"error": f"Directory {video_dir} does not exist"}

        if include_subdirectories:
            all_videos = []
            for ext in video_extensions:
                all_videos.extend(video_dir.rglob(f"*{ext}"))
        else:
            all_videos = []
            for ext in video_extensions:
                all_videos.extend(video_dir.glob(f"*{ext}"))

        print(f"Found {len(all_videos)} videos")

        if cache_file and Path(cache_file).exists():
            with open(cache_file, "r") as f:
                self.quality_cache = {
                    Path(k).resolve(): v for k, v in json.load(f).items()
                }
            print(f"Loaded {len(self.quality_cache)} cached quality scores")

        videos_to_assess = [
            v for v in all_videos if v.resolve() not in self.quality_cache
        ]

        print(f"Assessing {len(videos_to_assess)} new videos")

        for video in tqdm(videos_to_assess, desc="Quality assessment"):
            try:
                quality_result = self.assess_video_quality(video, num_frames, min_dimension=min_dimension)
                self.quality_cache[video.resolve()] = quality_result
            except Exception as e:
                print(f"Failed to assess {video}: {e}")
                continue

        if cache_file:
            with open(cache_file, "w") as f:
                json.dump({str(k): v for k, v in self.quality_cache.items()}, f)
            print(f"Saved quality cache to {cache_file}")

        return {
            "total_videos": len(all_videos),
            "assessed": len(videos_to_assess),
            "cached": len(all_videos) - len(videos_to_assess),
            "quality_scores": self.quality_cache,
        }

    def unload_models(self):
        """Unload models to free memory."""
        if self.aesthetic_model:
            self.aesthetic_model = self.aesthetic_model.to("cpu")
        if self.fastvqa_evaluator:
            self.fastvqa_evaluator = self.fastvqa_evaluator.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Quality] Models unloaded from GPU")
