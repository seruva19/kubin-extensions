import os
from uuid import uuid4

import imageio


def kdv11_create_video(
    kubin,
    prompt,
    negative_prompt,
    width,
    height,
    fps,
    motion,
    keyframe_guidance_scale,
    guidance_weight_prompt,
    guidance_weight_image,
    interpolation_guidance_scale,
    noise_augmentation,
    keyframe_image,
    device,
    cache_dir,
    output_dir,
    encoder_path,
    yaml_config,
):
    video_config = yaml_config.read()

    if video_config["use_lowvram_pipeline"]:
        kubin.log("(KandinskyVideo 1.1) -> using low VRAM t2v pipeline")
        from kandinsky_video_optimized import get_T2V_pipeline

        t2v_pipe = get_T2V_pipeline(
            device,
            cache_dir=cache_dir,
            text_encoder_path=encoder_path,
        )
    else:
        kubin.log("(KandinskyVideo 1.1) -> using original t2v pipeline")
        from kandinsky_video import get_T2V_pipeline

        t2v_pipe = get_T2V_pipeline(device, cache_dir=cache_dir)

    video_frames = t2v_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=keyframe_image,
        width=int(width),
        height=int(height),
        fps=fps,
        motion=motion,
        key_frame_guidance_scale=keyframe_guidance_scale,
        guidance_weight_prompt=guidance_weight_prompt,
        guidance_weight_image=guidance_weight_image,
        interpolation_guidance_scale=interpolation_guidance_scale,
        noise_augmentation=noise_augmentation,
    )

    out_video_dir = os.path.join(output_dir, "video")
    video_output_path = os.path.join(out_video_dir, f"video-{uuid4()}.gif")

    video_frames[0].save(
        video_output_path,
        save_all=True,
        append_images=video_frames[1:],
        duration=int(5500 / len(video_frames)),
        loop=0,
    )

    return [video_output_path]
