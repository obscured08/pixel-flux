import asyncio
import json
import pixelsort
from PIL import Image, ImageSequence, ImageFilter, ImageOps, ImageFile, ImageChops
from pixelsort import pixelsort as ps_func
import pixelsort.sorting as sorting_module

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Inject Custom Reverse Sorting Functions ---
if "reverse_lightness" not in sorting_module.choices:
    sorting_module.choices["reverse_lightness"] = lambda p: -sorting_module.lightness(p)
    sorting_module.choices["reverse_hue"] = lambda p: -sorting_module.hue(p)
    sorting_module.choices["reverse_saturation"] = lambda p: -sorting_module.saturation(p)
    sorting_module.choices["reverse_intensity"] = lambda p: -sorting_module.intensity(p)
    sorting_module.choices["reverse_minimum"] = lambda p: -sorting_module.minimum(p)

async def process_image(input_path, mask_path, interval_path, params_json, progress_callback=None):
    print("PYTHON STARTING...")
    params = json.loads(params_json)
    img = Image.open(input_path)
    
    mask_img = None
    if mask_path:
        mask_img = Image.open(mask_path).convert('L')
        if params['invert_mask']:
            mask_img = ImageOps.invert(mask_img)

    interval_img = None
    if interval_path:
        interval_img = Image.open(interval_path).convert('L')
        if params.get('invert_interval', False):
            interval_img = ImageOps.invert(interval_img)

    # --- Improved Frame Detection Logic ---
    input_is_animated = getattr(img, "is_animated", False) and img.n_frames > 1
    
    # Check if any "End" value differs from its "Start" value
    sliders_moved = (
        params['angle_start'] != params['angle_end'] or
        params['thresh_lower_start'] != params['thresh_lower_end'] or
        params['thresh_upper_start'] != params['thresh_upper_end'] or
        params['rand_start'] != params['rand_end'] or
        params['char_start'] != params['char_end'] or
        params['blur_start'] != params['blur_end'] or
        params['post_blur_start'] != params['post_blur_end'] or
        params.get('quantize_start', 0) != params.get('quantize_end', 0) or
        params.get('blend_start', 100) != params.get('blend_end', 100)
    )
    
    # We only generate a GIF if the source is already a GIF OR if the user is animating sliders
    # This prevents still images from turning into GIFs just because "Random" intervals are selected.
    target_frames = int(params.get('frame_count', 15))
    if not input_is_animated and not sliders_moved:
        target_frames = 1

    original_frames = [f.copy() for f in ImageSequence.Iterator(img)]
    
    if len(original_frames) == 1 and target_frames > 1:
        original_frames = [original_frames[0]] * target_frames
    elif len(original_frames) > 1 and target_frames > 1 and target_frames != len(original_frames):
        original_frames = [original_frames[i % len(original_frames)] for i in range(target_frames)]
    
    total_frames = len(original_frames)
    processed_frames = []
    prev_processed_frame = None

    fps = int(params.get('fps', 10))
    duration_ms = int(1000 / fps)
    if params.get('use_source_fps', False) and 'duration' in img.info:
         if img.info['duration'] > 20: duration_ms = img.info['duration']

    ghosting = float(params.get('ghosting', 0)) / 100.0

    for i, frame in enumerate(original_frames):
        if progress_callback:
            progress_callback(i + 1, total_frames)
        await asyncio.sleep(0) 

        t = i / total_frames if total_frames > 1 else 0
        
        cur_ang = params['angle_start'] + (params['angle_end'] - params['angle_start']) * t
        cur_tl = params['thresh_lower_start'] + (params['thresh_lower_end'] - params['thresh_lower_start']) * t
        cur_tu = params['thresh_upper_start'] + (params['thresh_upper_end'] - params['thresh_upper_start']) * t
        cur_rnd = params['rand_start'] + (params['rand_end'] - params['rand_start']) * t
        cur_cl = params['char_start'] + (params['char_end'] - params['char_start']) * t
        cur_blr = params['blur_start'] + (params['blur_end'] - params['blur_start']) * t
        cur_post_blr = params['post_blur_start'] + (params['post_blur_end'] - params['post_blur_start']) * t
        cur_quant = params.get('quantize_start', 0) + (params.get('quantize_end', 0) - params.get('quantize_start', 0)) * t
        cur_blend = (params.get('blend_start', 100) + (params.get('blend_end', 100) - params.get('blend_start', 100)) * t) / 100.0

        work_frame = frame.convert("RGB")
        original_work_frame = work_frame.copy()
            
        cur_mask = None
        if mask_img:
            cur_mask = mask_img.resize(work_frame.size)
            choke_val = int(params.get('mask_choke', 0))
            if choke_val > 0:
                for _ in range(choke_val): cur_mask = cur_mask.filter(ImageFilter.MaxFilter(3))
            elif choke_val < 0:
                for _ in range(abs(choke_val)): cur_mask = cur_mask.filter(ImageFilter.MinFilter(3))

        if cur_blr > 0:
            if params.get('mask_aware_pre_blur', False) and cur_mask:
                blurred_frame = work_frame.filter(ImageFilter.GaussianBlur(cur_blr))
                work_frame = Image.composite(blurred_frame, work_frame, cur_mask)
            else:
                work_frame = work_frame.filter(ImageFilter.GaussianBlur(cur_blr))

        if cur_quant >= 2:
            if params.get('mask_aware_palette', False) and cur_mask:
                palette_source = work_frame.copy()
                palette_source.putalpha(cur_mask)
                quantized_temp = palette_source.quantize(colors=int(cur_quant)).convert("RGB")
                work_frame = Image.composite(quantized_temp, work_frame, cur_mask)
            else:
                work_frame = work_frame.quantize(colors=int(cur_quant)).convert("RGB")

        cur_interval = None
        if interval_img:
            cur_interval = interval_img.resize(work_frame.size)

        try:
            sort_mask = cur_mask
            blend_mode = params.get('blend_mode', 'alpha')
            if cur_mask:
                if blend_mode == 'dither':
                    sort_mask = cur_mask 
                else:
                    sort_mask = cur_mask.point(lambda p: 255 if p > 0 else 0, mode='L')
            
            sort_func_name = params['sort_func']
            if params.get('invert_sort', False):
                sort_func_name = "reverse_" + sort_func_name
                
            ps_kwargs = dict(
                image=work_frame,
                mask_image=sort_mask,
                interval_image=cur_interval, 
                interval_function=params['interval_func'],
                sorting_function=sort_func_name,
                lower_threshold=float(cur_tl),
                upper_threshold=float(cur_tu),
                randomness=float(cur_rnd),
                char_length=int(cur_cl)
            )

            if params.get('flow_field', False):
                flow_offset = float(params.get('flow_offset', 90.0))
                sort1 = ps_func(**ps_kwargs, angle=float(cur_ang))
                sort2 = ps_func(**ps_kwargs, angle=float(cur_ang + flow_offset))
                flow_mask = work_frame.convert("L").filter(ImageFilter.GaussianBlur(10))
                sorted_frame = Image.composite(sort2.convert("RGB"), sort1.convert("RGB"), flow_mask)
            else:
                sorted_frame = ps_func(**ps_kwargs, angle=float(cur_ang))

            sorted_frame = sorted_frame.convert("RGB")

            if cur_blend < 1.0:
                blended_frame = Image.blend(work_frame, sorted_frame, cur_blend)
                if params.get('mask_aware_blend', False) and cur_mask:
                    sorted_frame = Image.composite(blended_frame, sorted_frame, cur_mask)
                else:
                    sorted_frame = blended_frame

            if cur_mask and blend_mode == 'alpha':
                sorted_frame = Image.composite(sorted_frame, work_frame, cur_mask)

            if cur_post_blr > 0:
                if params.get('mask_aware_post_blur', False) and cur_mask:
                    blurred_post = sorted_frame.filter(ImageFilter.GaussianBlur(cur_post_blr))
                    sorted_frame = Image.composite(blurred_post, sorted_frame, cur_mask)
                else:
                    sorted_frame = sorted_frame.filter(ImageFilter.GaussianBlur(cur_post_blr))

            if ghosting > 0.0 and prev_processed_frame is not None:
                sorted_frame = Image.blend(sorted_frame, prev_processed_frame, ghosting)
            prev_processed_frame = sorted_frame.copy()

            processed_frames.append(sorted_frame)
        except Exception as e:
            print(f"Python Error on Frame {i}: {e}")
            processed_frames.append(work_frame)

    if len(processed_frames) > 1:
        output_path = "/output.gif"
        processed_frames[0].save(output_path, save_all=True, append_images=processed_frames[1:], duration=duration_ms, loop=0)
        mime_type = "image/gif"
    else:
        output_path = "/output.png"
        processed_frames[0].save(output_path, format="PNG", optimize=True)
        mime_type = "image/png"

    with open(output_path, "rb") as f:
        return [f.read(), mime_type]