"""
SAM3 Video Tracking Nodes for ComfyUI - Stateless Architecture

These nodes provide video object tracking and segmentation using SAM3.
All state is encoded in immutable outputs - no global mutable state.

Key design principles:
1. All nodes are stateless - state flows through outputs
2. SAM3VideoState is immutable - adding prompts returns NEW state
3. Inference state is reconstructed on-demand
4. Temp directories are automatically cleaned up at process exit
5. No manual SAM3CloseVideoSession needed
"""
import gc
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import folder_paths
import comfy.model_management

from .video_state import (
    SAM3VideoState,
    VideoPrompt,
    VideoConfig,
    create_video_state,
    cleanup_temp_dir,
)
from .inference_reconstructor import (
    get_inference_state,
    invalidate_session,
    clear_inference_cache,
)
from .sam3_model_patcher import SAM3ModelWrapper, SAM3ModelPatcher


# =============================================================================
# Autocast dtype detection - handles GPUs without bf16 support
# =============================================================================
def _get_autocast_dtype():
    """
    Get appropriate autocast dtype based on GPU capability.
    Returns None if autocast should not be used.
    """
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:  # Ampere+ supports bf16
        return torch.bfloat16
    elif major >= 7:  # Volta/Turing use fp16
        return torch.float16
    else:
        return None  # Older GPUs - no autocast


def _get_autocast_context():
    """Get autocast context manager based on GPU capability."""
    dtype = _get_autocast_dtype()
    if dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.no_grad()


# =============================================================================
# VRAM Debug Utility
# =============================================================================

def print_vram(label: str, detailed: bool = False):
    """Print current VRAM usage for debugging memory leaks."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] {label}: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved")
        if detailed:
            # Print memory stats breakdown
            stats = torch.cuda.memory_stats()
            print(f"[VRAM]   Active: {stats.get('active_bytes.all.current', 0) / 1024**3:.2f}GB")
            print(f"[VRAM]   Inactive: {stats.get('inactive_split_bytes.all.current', 0) / 1024**3:.2f}GB")
            print(f"[VRAM]   Allocated retries: {stats.get('num_alloc_retries', 0)}")


# =============================================================================
# Video Segmentation Nodes
# =============================================================================
# NOTE: SAM3VideoModelLoader has been removed.
# Use LoadSAM3Model instead - it returns a unified model that works for both
# image segmentation and video tracking.


# =============================================================================
# Video Segmentation (Unified Node)
# =============================================================================

class SAM3VideoSegmentation:
    """
    Initialize video tracking and add prompts.

    Select prompt_mode to choose between:
    - text: Track objects by text description (comma-separated for multiple)
    - point: Track objects by clicking points (positive/negative)
    - box: Track objects by drawing boxes (positive/negative)

    Note: SAM3 video does NOT support combining different prompt types.
    Each mode is mutually exclusive.
    """
    # Class-level cache for video state results
    _cache = {}

    PROMPT_MODES = ["text", "point", "box"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Video frames as batch of images [N, H, W, C]"
                }),
                "prompt_mode": (cls.PROMPT_MODES, {
                    "default": "text",
                    "tooltip": "Prompt type: text (describe objects), point (click on objects), or box (draw rectangles)"
                }),
            },
            "optional": {
                # Text mode inputs
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "[text mode] Text description(s) to track. Comma-separated for multiple objects (e.g., 'person, dog, car')"
                }),
                # Point mode inputs
                "positive_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "[point mode] Positive points - click on objects to track"
                }),
                "negative_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "[point mode] Negative points - click on areas to exclude"
                }),
                # Box mode inputs
                "positive_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "[box mode] Positive boxes - draw around objects to track"
                }),
                "negative_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "[box mode] Negative boxes - draw around areas to exclude"
                }),
                # Common inputs
                "frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame index to apply prompts (usually 0 for first frame)"
                }),
                "score_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Detection confidence threshold"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, video_frames, prompt_mode="text", text_prompt="",
                   positive_points=None, negative_points=None,
                   positive_boxes=None, negative_boxes=None,
                   frame_idx=0, score_threshold=0.3):
        # Use a stable hash based on video content
        # Don't use float(mean()) - it has floating point precision issues on GPU
        import hashlib

        # Create a stable hash from video frame content
        # Use shape + corner pixels from first and last frame (deterministic bytes, no float issues)
        h = hashlib.md5()
        h.update(str(video_frames.shape).encode())

        # Sample corner pixels from first and last frame
        first_frame = video_frames[0].cpu().numpy()
        last_frame = video_frames[-1].cpu().numpy()
        h.update(first_frame[0, 0, :].tobytes())      # top-left
        h.update(first_frame[-1, -1, :].tobytes())    # bottom-right
        h.update(last_frame[0, 0, :].tobytes())
        h.update(last_frame[-1, -1, :].tobytes())

        video_hash = h.hexdigest()

        result = hash((
            video_hash,
            prompt_mode,
            text_prompt,
            str(positive_points),
            str(negative_points),
            str(positive_boxes),
            str(negative_boxes),
            frame_idx,
            score_threshold,
        ))
        print(f"[IS_CHANGED DEBUG] SAM3VideoSegmentation: video_hash={video_hash}, prompt_mode={prompt_mode}")
        print(f"[IS_CHANGED DEBUG] SAM3VideoSegmentation: positive_points={positive_points}")
        print(f"[IS_CHANGED DEBUG] SAM3VideoSegmentation: negative_points={negative_points}")
        print(f"[IS_CHANGED DEBUG] SAM3VideoSegmentation: returning hash={result}")
        return result

    RETURN_TYPES = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES = ("video_state",)
    FUNCTION = "segment"
    CATEGORY = "SAM3/video"

    def segment(self, video_frames, prompt_mode="text", text_prompt="",
                positive_points=None, negative_points=None,
                positive_boxes=None, negative_boxes=None,
                frame_idx=0, score_threshold=0.3):
        """Initialize video state and add prompts based on selected mode."""
        # Create cache key from inputs
        import hashlib
        h = hashlib.md5()
        h.update(str(video_frames.shape).encode())
        # Sample corner pixels for video identity
        first_frame = video_frames[0].cpu().numpy()
        last_frame = video_frames[-1].cpu().numpy()
        h.update(first_frame[0, 0, :].tobytes())
        h.update(first_frame[-1, -1, :].tobytes())
        h.update(last_frame[0, 0, :].tobytes())
        h.update(last_frame[-1, -1, :].tobytes())
        h.update(prompt_mode.encode())
        h.update(text_prompt.encode())
        h.update(str(id(positive_points)).encode() if positive_points else b"none")
        h.update(str(id(negative_points)).encode() if negative_points else b"none")
        h.update(str(id(positive_boxes)).encode() if positive_boxes else b"none")
        h.update(str(id(negative_boxes)).encode() if negative_boxes else b"none")
        h.update(str(frame_idx).encode())
        h.update(str(score_threshold).encode())
        cache_key = h.hexdigest()

        # Check if we have cached result
        if cache_key in SAM3VideoSegmentation._cache:
            cached = SAM3VideoSegmentation._cache[cache_key]
            print(f"[SAM3 Video] CACHE HIT - returning cached video_state for key={cache_key[:8]}, session={cached.session_uuid[:8]}")
            return (cached,)

        print(f"[SAM3 Video] CACHE MISS - computing new video_state for key={cache_key[:8]}")
        print_vram("Before video segmentation")

        # 1. Initialize video state
        config = VideoConfig(
            score_threshold_detection=score_threshold,
        )
        video_state = create_video_state(
            video_frames=video_frames,
            config=config,
        )

        print(f"[SAM3 Video] Initialized session {video_state.session_uuid[:8]}")
        print(f"[SAM3 Video] Frames: {video_state.num_frames}, Size: {video_state.width}x{video_state.height}")
        print(f"[SAM3 Video] Prompt mode: {prompt_mode}")

        # 2. Add prompts based on mode (mutually exclusive)
        obj_id = 1

        if prompt_mode == "text":
            # Text mode: parse comma-separated text prompts
            if text_prompt and text_prompt.strip():
                for text in text_prompt.split(","):
                    text = text.strip()
                    if text:
                        prompt = VideoPrompt.create_text(frame_idx, obj_id, text)
                        video_state = video_state.with_prompt(prompt)
                        print(f"[SAM3 Video] Added text prompt: obj={obj_id}, text='{text}'")
                        obj_id += 1
            else:
                print("[SAM3 Video] Warning: text mode selected but no text_prompt provided")

        elif prompt_mode == "point":
            # Point mode: combine positive and negative points
            all_points = []
            all_labels = []

            if positive_points and positive_points.get("points"):
                for pt in positive_points["points"]:
                    all_points.append([float(pt[0]), float(pt[1])])
                    all_labels.append(1)  # Positive

            if negative_points and negative_points.get("points"):
                for pt in negative_points["points"]:
                    all_points.append([float(pt[0]), float(pt[1])])
                    all_labels.append(0)  # Negative

            if all_points:
                prompt = VideoPrompt.create_point(frame_idx, obj_id, all_points, all_labels)
                video_state = video_state.with_prompt(prompt)
                pos_count = len(positive_points.get("points", [])) if positive_points else 0
                neg_count = len(negative_points.get("points", [])) if negative_points else 0
                print(f"[SAM3 Video] Added point prompt: obj={obj_id}, "
                      f"positive={pos_count}, negative={neg_count}")
            else:
                print("[SAM3 Video] Warning: point mode selected but no points provided")

        elif prompt_mode == "box":
            # Box mode: add positive and/or negative boxes
            has_boxes = False

            if positive_boxes and positive_boxes.get("boxes"):
                box_data = positive_boxes["boxes"][0]  # First box
                cx, cy, w, h = box_data
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2], is_positive=True)
                video_state = video_state.with_prompt(prompt)
                print(f"[SAM3 Video] Added positive box: obj={obj_id}, "
                      f"box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                has_boxes = True

            if negative_boxes and negative_boxes.get("boxes"):
                box_data = negative_boxes["boxes"][0]  # First box
                cx, cy, w, h = box_data
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2], is_positive=False)
                video_state = video_state.with_prompt(prompt)
                print(f"[SAM3 Video] Added negative box: obj={obj_id}, "
                      f"box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                has_boxes = True

            if not has_boxes:
                print("[SAM3 Video] Warning: box mode selected but no boxes provided")

        # Validate at least one prompt was added
        if len(video_state.prompts) == 0:
            print(f"[SAM3 Video] Warning: No prompts added for mode '{prompt_mode}'")

        print(f"[SAM3 Video] Total prompts: {len(video_state.prompts)}")
        print_vram("After video segmentation")

        # Cache the result
        SAM3VideoSegmentation._cache[cache_key] = video_state

        return (video_state,)


# =============================================================================
# Propagation
# =============================================================================

class SAM3Propagate:
    """
    Run video propagation to track objects across frames.

    Reconstructs inference state on-demand from immutable video state.
    """
    # Class-level cache for propagation results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL", {
                    "tooltip": "SAM3 model (from LoadSAM3Model)"
                }),
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state with prompts"
                }),
            },
            "optional": {
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Start frame for propagation"
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "End frame (-1 for all)"
                }),
                "direction": (["forward", "backward", "both"], {
                    "default": "forward",
                    "tooltip": "Propagation direction: forward (future frames), backward (past frames), or both directions"
                }),
                "offload_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Move model to CPU after propagation to free VRAM (slower next run)"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_MASKS", "SAM3_VIDEO_SCORES", "SAM3_VIDEO_STATE")
    RETURN_NAMES = ("masks", "scores", "video_state")
    FUNCTION = "propagate"
    CATEGORY = "SAM3/video"

    @classmethod
    def IS_CHANGED(cls, sam3_model, video_state, start_frame=0, end_frame=-1, direction="forward", offload_model=False):
        # Use object identity for caching - if upstream node is cached,
        # it returns the same object, so id() will match
        # This is more reliable than hashing content since video_state is immutable
        result = (id(video_state), start_frame, end_frame, direction)
        print(f"[IS_CHANGED DEBUG] SAM3Propagate: video_state id={id(video_state)}, session={video_state.session_uuid if video_state else None}")
        print(f"[IS_CHANGED DEBUG] SAM3Propagate: returning {result}")
        return result

    def propagate(self, sam3_model, video_state, start_frame=0, end_frame=-1, direction="forward", offload_model=False):
        """Run propagation using reconstructed inference state."""
        # Create cache key using video_state object id (since it's immutable and cached upstream)
        cache_key = (id(video_state), start_frame, end_frame, direction)

        # Check if we have cached result
        if cache_key in SAM3Propagate._cache:
            cached = SAM3Propagate._cache[cache_key]
            print(f"[SAM3 Propagate] CACHE HIT - returning cached result for session={video_state.session_uuid[:8]}")
            # Still need to handle offload if requested
            if offload_model:
                print("[SAM3 Video] Offloading model to CPU to free VRAM...")
                if hasattr(sam3_model, 'model'):
                    sam3_model.model.cpu()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print_vram("After model offload")
            return cached

        print(f"[SAM3 Propagate] CACHE MISS - running propagation for session={video_state.session_uuid[:8]}")

        if len(video_state.prompts) == 0:
            raise ValueError("[SAM3 Video] No prompts added. Add point, box, or text prompts before propagating.")

        # Ensure model is on GPU before inference (may have been offloaded)
        if hasattr(sam3_model, 'model') and hasattr(sam3_model.model, 'to'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            sam3_model.model.to(device)

        print(f"[SAM3 Video] Starting propagation: frames {start_frame} to {end_frame if end_frame >= 0 else 'end'}")
        print(f"[SAM3 Video] Prompts: {len(video_state.prompts)}")
        print_vram("Before propagation start")

        # Determine frame range
        if end_frame < 0:
            end_frame = video_state.num_frames - 1

        # Build propagation request - uses predictor's handle_stream_request API
        # direction is already "forward", "backward", or "both"
        request = {
            "type": "propagate_in_video",
            "session_id": video_state.session_uuid,
            "propagation_direction": direction,
            "start_frame_index": start_frame,
            "max_frame_num_to_track": end_frame - start_frame + 1,
        }

        # Run ALL inference inside autocast context for dtype consistency
        # SAM3 requires bf16/fp16 - wrap reconstruction AND propagation
        masks_dict = {}
        scores_dict = {}  # Store confidence scores per frame
        # Use autocast with dtype based on GPU capability (bf16 for Ampere+, fp16 for Volta/Turing)
        autocast_context = _get_autocast_context()
        with autocast_context:
            print_vram("Before reconstruction (in autocast)")
            # Reconstruct inference state from immutable state
            inference_state = get_inference_state(sam3_model, video_state)
            print_vram("After reconstruction")

            # Run propagation
            try:
                for response in sam3_model.handle_stream_request(request):
                    frame_idx = response.get("frame_index", response.get("frame_idx"))
                    if frame_idx is None:
                        continue

                    outputs = response.get("outputs", response)
                    if outputs is None:
                        continue

                    # Try different possible mask keys
                    mask_key = None
                    for key in ["out_binary_masks", "video_res_masks", "masks"]:
                        if key in outputs and outputs[key] is not None:
                            mask_key = key
                            break

                    if mask_key:
                        # Move masks to CPU immediately to free GPU memory
                        mask = outputs[mask_key]
                        if hasattr(mask, 'cpu'):
                            mask = mask.cpu()
                        masks_dict[frame_idx] = mask

                    # Capture confidence scores
                    for score_key in ["out_probs", "scores", "confidences", "obj_scores"]:
                        if score_key in outputs and outputs[score_key] is not None:
                            probs = outputs[score_key]
                            if hasattr(probs, 'cpu'):
                                probs = probs.cpu()
                            elif isinstance(probs, np.ndarray):
                                probs = torch.from_numpy(probs)
                            scores_dict[frame_idx] = probs
                            break

                    # Periodic cleanup and VRAM monitoring
                    if frame_idx % 10 == 0:
                        print_vram(f"Frame {frame_idx}")
                        gc.collect()

            except Exception as e:
                print(f"[SAM3 Video] Propagation error: {e}")
                import traceback
                traceback.print_exc()
                raise

        print_vram("After propagation loop")
        print(f"[SAM3 Video] Propagation complete: {len(masks_dict)} frames processed")
        print(f"[SAM3 Video] Frames with scores: {len(scores_dict)}")

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Offload model to CPU if requested (Issue #28)
        if offload_model:
            print("[SAM3 Video] Offloading model to CPU to free VRAM...")
            if hasattr(sam3_model, 'model'):
                sam3_model.model.cpu()
            # Clear inference state cache to free GPU memory
            from .sam3_lib.sam3_video_predictor import Sam3VideoPredictor
            Sam3VideoPredictor._ALL_INFERENCE_STATES.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print_vram("After model offload")

        # Cache the result
        result = (masks_dict, scores_dict, video_state)
        SAM3Propagate._cache[cache_key] = result

        return result


# =============================================================================
# Output Extraction
# =============================================================================

class SAM3VideoOutput:
    """
    Extract masks from propagation results.

    Converts SAM3_VIDEO_MASKS to ComfyUI-compatible mask tensors.
    Returns all frames as a batch.

    Changing obj_id does NOT re-run propagation - only this node re-executes.
    """
    # Class-level cache for extraction results
    _cache: dict = {}

    # Pre-defined color palette as a class constant (avoids recreation)
    COLORS: torch.Tensor = torch.tensor([
        [0.0, 0.5, 1.0],   # Blue
        [1.0, 0.3, 0.3],   # Red
        [0.3, 1.0, 0.3],   # Green
        [1.0, 1.0, 0.0],   # Yellow
        [1.0, 0.0, 1.0],   # Magenta
        [0.0, 1.0, 1.0],   # Cyan
        [1.0, 0.5, 0.0],   # Orange
        [0.5, 0.0, 1.0],   # Purple
    ], dtype=torch.float32)

    # Pre-defined 3x5 pixel font patterns for digits and punctuation
    CHAR_PATTERNS: dict = {
        '0': [[1,1,1], [1,0,1], [1,0,1], [1,0,1], [1,1,1]],
        '1': [[0,1,0], [1,1,0], [0,1,0], [0,1,0], [1,1,1]],
        '2': [[1,1,1], [0,0,1], [1,1,1], [1,0,0], [1,1,1]],
        '3': [[1,1,1], [0,0,1], [1,1,1], [0,0,1], [1,1,1]],
        '4': [[1,0,1], [1,0,1], [1,1,1], [0,0,1], [0,0,1]],
        '5': [[1,1,1], [1,0,0], [1,1,1], [0,0,1], [1,1,1]],
        '6': [[1,1,1], [1,0,0], [1,1,1], [1,0,1], [1,1,1]],
        '7': [[1,1,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1]],
        '8': [[1,1,1], [1,0,1], [1,1,1], [1,0,1], [1,1,1]],
        '9': [[1,1,1], [1,0,1], [1,1,1], [0,0,1], [1,1,1]],
        ':': [[0,0,0], [0,1,0], [0,0,0], [0,1,0], [0,0,0]],
        '.': [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,1,0]],
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("SAM3_VIDEO_MASKS", {
                    "tooltip": "Masks from SAM3Propagate"
                }),
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state for dimensions"
                }),
            },
            "optional": {
                "scores": ("SAM3_VIDEO_SCORES", {
                    "tooltip": "Confidence scores from SAM3Propagate"
                }),
                "obj_id": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Specific object ID for mask output (-1 for all combined). Changing this is fast - no re-inference needed."
                }),
                "plot_all_masks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show all object masks in visualization (True) or only selected obj_id (False)"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(
        cls,
        masks,
        video_state,
        scores=None,
        obj_id: int = -1,
        plot_all_masks: bool = True
    ):
        return (id(masks), video_state.session_uuid, id(scores), obj_id, plot_all_masks)

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("masks", "frames", "visualization")
    FUNCTION = "extract"
    CATEGORY = "SAM3/video"

    def _render_text_to_tensor(
        self,
        text: str,
        scale: int
    ) -> torch.Tensor:
        """
        Render text string to a binary tensor using the pixel font.

        Returns a [height, width] float tensor with 1.0 for text pixels.
        """
        char_height = 5 * scale
        char_width = 3 * scale
        spacing = 1 * scale

        # Calculate total width needed
        total_width = 0
        for char in text:
            if char in self.CHAR_PATTERNS:
                total_width += char_width + spacing
            elif char == ' ':
                total_width += char_width + spacing

        if total_width > 0:
            total_width -= spacing  # Remove trailing space

        # Create output tensor
        text_tensor = torch.zeros(char_height, max(1, total_width), dtype=torch.float32)

        curr_x = 0
        for char in text:
            if char in self.CHAR_PATTERNS:
                pattern = self.CHAR_PATTERNS[char]
                pattern_tensor = torch.tensor(pattern, dtype=torch.float32)

                # Scale up the pattern using nearest-neighbor (repeat)
                scaled_pattern = pattern_tensor.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)

                # Place into output
                end_x = min(curr_x + char_width, text_tensor.shape[1])
                actual_width = end_x - curr_x
                text_tensor[:, curr_x:end_x] = scaled_pattern[:, :actual_width]

                curr_x += char_width + spacing
            elif char == ' ':
                curr_x += char_width + spacing

        return text_tensor

    def _render_legend_overlay(
        self,
        height: int,
        width: int,
        num_objects: int,
        obj_id: int,
        frame_scores: Optional[list],
        colors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render legend as RGB overlay and alpha mask tensors.

        Returns:
            overlay: [H, W, 3] RGB tensor
            alpha: [H, W] alpha tensor (0-1)
        """
        # Legend parameters scaled to image size
        box_size = max(16, min(32, height // 20))
        padding = max(4, box_size // 4)
        scale = max(1, box_size // 6)

        # Build list of (obj_id, score) pairs
        if obj_id >= 0:
            items = [(obj_id, frame_scores[obj_id] if frame_scores is not None and obj_id < len(frame_scores) else None)]
        else:
            items = []
            for oid in range(num_objects):
                score = frame_scores[oid] if frame_scores is not None and oid < len(frame_scores) else None
                items.append((oid, score))
            # Sort by score descending (highest confidence first), None scores go last
            items.sort(key=lambda x: (x[1] is None, -(x[1] if x[1] is not None else 0)))

        num_items = len(items)
        if num_items == 0:
            return torch.zeros(height, width, 3), torch.zeros(height, width)

        legend_item_height = box_size + padding

        # Pre-render all text items and find max width
        text_renders = []
        max_text_width = 0
        for oid, score in items:
            if score is not None:
                score_str = f"{oid}:{score:.2f}"
            else:
                score_str = f"{oid}"
            text_tensor = self._render_text_to_tensor(score_str, scale)
            text_renders.append(text_tensor)
            max_text_width = max(max_text_width, text_tensor.shape[1])

        # Calculate legend dimensions
        legend_height = num_items * legend_item_height + padding * 2
        legend_width = padding + box_size + padding + max_text_width + padding

        # Create legend tensors
        overlay = torch.zeros(height, width, 3, dtype=torch.float32)
        alpha = torch.zeros(height, width, dtype=torch.float32)

        # Clamp legend to image bounds
        leg_h = min(legend_height, height - padding)
        leg_w = min(legend_width, width - padding)

        # Draw semi-transparent background
        bg_color = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
        overlay[padding:padding + leg_h, padding:padding + leg_w, :] = bg_color
        alpha[padding:padding + leg_h, padding:padding + leg_w] = 0.7

        # Draw legend items
        for idx, ((oid, score), text_tensor) in enumerate(zip(items, text_renders)):
            item_y = padding + padding + idx * legend_item_height

            if item_y + box_size > height:
                break

            # Draw color box (vectorized)
            box_x_start = padding + padding
            box_x_end = min(box_x_start + box_size, width)
            box_y_end = min(item_y + box_size, height)

            color = colors[oid % len(colors)]
            overlay[item_y:box_y_end, box_x_start:box_x_end, :] = color
            alpha[item_y:box_y_end, box_x_start:box_x_end] = 1.0

            # Draw text (vectorized)
            text_x = box_x_end + padding
            text_h, text_w = text_tensor.shape
            text_x_end = min(text_x + text_w, width)
            text_y_end = min(item_y + text_h, height)
            actual_text_w = text_x_end - text_x
            actual_text_h = text_y_end - item_y

            if actual_text_w > 0 and actual_text_h > 0:
                text_mask = text_tensor[:actual_text_h, :actual_text_w]
                # White text
                for c in range(3):
                    overlay[item_y:text_y_end, text_x:text_x_end, c] = torch.where(
                        text_mask > 0.5,
                        torch.ones_like(text_mask),
                        overlay[item_y:text_y_end, text_x:text_x_end, c]
                    )
                alpha[item_y:text_y_end, text_x:text_x_end] = torch.maximum(
                    alpha[item_y:text_y_end, text_x:text_x_end],
                    text_mask
                )

        return overlay, alpha

    def _load_frame(
        self,
        frame_idx: int,
        temp_dir: str,
        height: int,
        width: int
    ) -> Tuple[int, np.ndarray]:
        """Load a single frame from disk. Used for parallel loading."""
        from PIL import Image
        import os

        frame_path = os.path.join(temp_dir, f"{frame_idx:05d}.jpg")
        if os.path.exists(frame_path):
            img = Image.open(frame_path).convert("RGB")
            img_np = np.array(img, dtype=np.float32)
            img_np *= (1.0 / 255.0)  # In-place normalization
            return frame_idx, img_np
        else:
            return frame_idx, np.zeros((height, width, 3), dtype=np.float32)

    def extract(
        self,
        masks,
        video_state,
        scores=None,
        obj_id: int = -1,
        plot_all_masks: bool = True
    ):
        """Extract all masks as a batch [N, H, W]."""
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial
        import os

        # Create cache key
        cache_key = (id(masks), video_state.session_uuid, id(scores), obj_id, plot_all_masks)

        # Check if we have cached result
        if cache_key in SAM3VideoOutput._cache:
            print(f"[SAM3 Video Output] CACHE HIT - returning cached result for session={video_state.session_uuid[:8]}")
            return SAM3VideoOutput._cache[cache_key]

        print(f"[SAM3 Video Output] CACHE MISS - extracting masks for session={video_state.session_uuid[:8]}")
        print_vram("Before extract")

        h, w = video_state.height, video_state.width
        num_frames = video_state.num_frames

        if not masks:
            print("[SAM3 Video] No masks to extract")
            empty_mask = torch.zeros(num_frames, h, w)
            empty_frames = torch.zeros(num_frames, h, w, 3)
            return (empty_mask, empty_frames, empty_frames)

        # Pre-allocate output tensors (avoids list accumulation and final stack copy)
        all_masks = torch.zeros(num_frames, h, w, dtype=torch.float32)
        all_frames = torch.zeros(num_frames, h, w, 3, dtype=torch.float32)
        all_vis = torch.zeros(num_frames, h, w, 3, dtype=torch.float32)

        # Get colors tensor (class-level, already created)
        colors = self.COLORS

        # Determine number of objects from first available mask
        num_objects = 0
        for frame_idx in masks:
            frame_mask = masks[frame_idx]
            if isinstance(frame_mask, np.ndarray):
                frame_mask = torch.from_numpy(frame_mask)
            if frame_mask.dim() == 4:
                frame_mask = frame_mask.squeeze(0)
            if frame_mask.dim() == 3 and frame_mask.shape[0] > 0:
                num_objects = frame_mask.shape[0]
                break
            elif frame_mask.dim() == 2:
                num_objects = 1
                break

        # Load all frames in parallel
        print(f"[SAM3 Video Output] Loading {num_frames} frames in parallel...")
        load_func = partial(
            self._load_frame,
            temp_dir=video_state.temp_dir,
            height=h,
            width=w
        )

        # Use ThreadPoolExecutor for parallel I/O (GIL is released during I/O)
        num_workers = min(16, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for frame_idx, img_np in executor.map(load_func, range(num_frames)):
                all_frames[frame_idx] = torch.from_numpy(img_np)

        print(f"[SAM3 Video Output] Frames loaded. Processing masks...")

        # Process masks - vectorized operations
        for frame_idx in range(num_frames):
            # Start with original frame as base for visualization
            vis_frame = all_frames[frame_idx].clone()

            if frame_idx not in masks:
                # No mask for this frame - output stays as zeros (pre-allocated)
                all_vis[frame_idx] = vis_frame
                continue

            frame_mask = masks[frame_idx]

            # Convert numpy to torch if needed
            if isinstance(frame_mask, np.ndarray):
                frame_mask = torch.from_numpy(frame_mask)

            # Remove batch dimension if present
            if frame_mask.dim() == 4:
                frame_mask = frame_mask.squeeze(0)

            # Handle empty mask
            if frame_mask.numel() == 0 or (frame_mask.dim() == 3 and frame_mask.shape[0] == 0):
                all_vis[frame_idx] = vis_frame
                continue

            # Process multi-object masks [num_objects, H, W]
            if frame_mask.dim() == 3 and frame_mask.shape[0] >= 1:
                num_obj_in_frame = frame_mask.shape[0]

                # Normalize mask values to 0-1 range
                frame_mask = frame_mask.float()
                if frame_mask.max() > 1.0:
                    frame_mask = frame_mask / 255.0

                # Create visualization with colored overlays (vectorized)
                if plot_all_masks:
                    # Overlay all objects
                    for oid in range(num_obj_in_frame):
                        obj_mask = frame_mask[oid]  # [H, W]
                        if obj_mask.max() < 1e-6:
                            continue

                        color = colors[oid % len(colors)]  # [3]
                        # Vectorized blending: vis = vis * (1 - 0.5*mask) + 0.5 * mask * color
                        mask_expanded = obj_mask.unsqueeze(-1)  # [H, W, 1]
                        blend_factor = 0.5 * mask_expanded
                        vis_frame = vis_frame * (1.0 - blend_factor) + blend_factor * color

                    # Combined mask for output
                    combined_mask = frame_mask.max(dim=0)[0]  # [H, W]
                else:
                    # Only show selected object in visualization
                    vis_oid = obj_id if 0 <= obj_id < num_obj_in_frame else 0
                    obj_mask = frame_mask[vis_oid]

                    if obj_mask.max() >= 1e-6:
                        color = colors[vis_oid % len(colors)]
                        mask_expanded = obj_mask.unsqueeze(-1)
                        blend_factor = 0.5 * mask_expanded
                        vis_frame = vis_frame * (1.0 - blend_factor) + blend_factor * color

                    # Combined mask for output
                    combined_mask = frame_mask.max(dim=0)[0]

                # Select output mask based on obj_id
                if 0 <= obj_id < num_obj_in_frame:
                    output_mask = frame_mask[obj_id]
                else:
                    output_mask = combined_mask

                all_masks[frame_idx] = output_mask

            else:
                # Single mask [H, W]
                if frame_mask.dim() == 3:
                    frame_mask = frame_mask.squeeze(0)
                frame_mask = frame_mask.float()
                if frame_mask.max() > 1.0:
                    frame_mask = frame_mask / 255.0

                # Visualize single mask
                if frame_mask.max() >= 1e-6:
                    color = colors[0]
                    mask_expanded = frame_mask.unsqueeze(-1)
                    blend_factor = 0.5 * mask_expanded
                    vis_frame = vis_frame * (1.0 - blend_factor) + blend_factor * color

                all_masks[frame_idx] = frame_mask

            all_vis[frame_idx] = vis_frame.clamp(0, 1)

            # Periodic progress logging
            if frame_idx > 0 and frame_idx % 500 == 0:
                print(f"[SAM3 Video Output] Processed {frame_idx}/{num_frames} frames")

        # Draw legend on all visualization frames (vectorized)
        if num_objects > 0:
            print(f"[SAM3 Video Output] Rendering legend overlay...")
            legend_obj_id = -1 if plot_all_masks else obj_id

            # For each frame, we might have different scores, so we need per-frame legends
            # But if scores don't vary much, we could optimize further by caching legend renders
            for frame_idx in range(num_frames):
                # Get scores for this frame
                frame_scores = None
                if scores is not None and frame_idx in scores:
                    frame_scores_tensor = scores[frame_idx]
                    if hasattr(frame_scores_tensor, 'tolist'):
                        frame_scores = frame_scores_tensor.tolist()
                        if frame_scores and isinstance(frame_scores[0], list):
                            frame_scores = frame_scores[0]
                    elif hasattr(frame_scores_tensor, '__iter__'):
                        frame_scores = list(frame_scores_tensor)

                # Render legend overlay (fully vectorized, no pixel loops)
                legend_overlay, legend_alpha = self._render_legend_overlay(
                    h, w, num_objects, legend_obj_id, frame_scores, colors
                )

                # Composite legend onto visualization (vectorized)
                alpha_expanded = legend_alpha.unsqueeze(-1)  # [H, W, 1]
                all_vis[frame_idx] = (
                    all_vis[frame_idx] * (1.0 - alpha_expanded) +
                    legend_overlay * alpha_expanded
                )

        print(f"[SAM3 Video] Output: {all_masks.shape[0]} masks, shape {all_masks.shape}")
        print(f"[SAM3 Video] Objects tracked: {num_objects}, plot_all_masks: {plot_all_masks}")
        print_vram("After extract")

        # Cache the result (consider disabling for very large videos)
        result = (all_masks, all_frames, all_vis)

        # Only cache if total size is reasonable (< 8GB)
        estimated_size_gb = (all_masks.numel() + all_frames.numel() + all_vis.numel()) * 4 / (1024**3)
        if estimated_size_gb < 1.0:
            SAM3VideoOutput._cache[cache_key] = result
            print(f"[SAM3 Video Output] Cached result ({estimated_size_gb:.2f} GB)")
        else:
            print(f"[SAM3 Video Output] Skipping cache - result too large ({estimated_size_gb:.2f} GB)")

        return result


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SAM3VideoSegmentation": SAM3VideoSegmentation,
    "SAM3Propagate": SAM3Propagate,
    "SAM3VideoOutput": SAM3VideoOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3VideoSegmentation": "SAM3 Video Segmentation",
    "SAM3Propagate": "SAM3 Propagate",
    "SAM3VideoOutput": "SAM3 Video Output",
}
