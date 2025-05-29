import os
import json
import sys
import gradio as gr
import shutil

from PIL import Image, ImageDraw
from loguru import logger
from pathlib import Path

"""
xCensorNing 打码工具
Basic from https://github.com/zhulinyv/Semi-Auto-NovelAI-to-Pixiv.git
"""

format_ = (
    f"<m>xCensorNing</m>"
    "| <c>{time:YY-MM-DD HH:mm:ss}</c> "
    "| <c>{module}:{line}</c> "
    "| <level>{level}</level> "
    "| <level>{message}</level>"
)

logger.remove()
logger.add(sys.stdout, format=format_, colorize=True)

def file_path2name(path) -> str:
    """文件路径返还文件名

    Args:
        path (str|WindowsPath): 路径

    Returns:
        (str): 文件名
    """
    return os.path.basename(path)


def file_path2list(path) -> list[str]:
    """文件目录返还文件名列表

    Args:
        path (str|WindowsPath): 目录

    Returns:
        (list[str]): 文件名列表
    """
    return os.listdir(path)


def file_namel2pathl(file_list: list, file_path):
    """文件名列表返还文件路径列表

    Args:
        file_list (list): 文件名列表
        file_path (_type_): 文件路径列表

    Returns:
        (list[str]): 文件路径列表
    """
    empty_list = []
    for file in file_list:
        empty_list.append(Path(file_path) / file)
    file_list = empty_list[:]
    return file_list


NEIGHBOR = 0.1

try:
    from ultralytics import YOLO

    logger.debug("使用 YOLO 进行图像预测")

    def detector(image):
        model = YOLO("./models/censor.pt")
        box_list = []
        results = model(image, verbose=False)
        result = json.loads((results[0]).tojson())
        for part in result:
            if part["name"] in ["penis", "pussy"]:
                logger.debug("检测到: {}".format(part["name"]))
                x = round(part["box"]["x1"])
                y = round(part["box"]["y1"])
                w = round(part["box"]["x2"] - part["box"]["x1"])
                h = round(part["box"]["y2"] - part["box"]["y1"])
                box_list.append([x, y, w, h])
        return box_list

except ModuleNotFoundError:
    from nudenet import NudeDetector

    logger.debug("使用 nudenet 进行图像检测")

    def detector(image):
        nude_detector = NudeDetector()
        # 这个库不能使用中文文件名
        # 写重复了, batch_mosaic 里已经写过了
        box_list = []
        body = nude_detector.detect("./output/temp.png")
        for part in body:
            if part["class"] in ["FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"]:
                logger.debug("检测到: {}".format(part["class"]))
                x = part["box"][0]
                y = part["box"][1]
                w = part["box"][2]
                h = part["box"][3]
                box_list.append([x, y, w, h])
        return box_list



# -------------------- #


def __mosaic_blurry(img, length):
    s = img.size
    img = img.resize((int(length * 0.01), int(length * 0.01)))
    img = img.resize(s)
    return img


def _mosaic_blurry(img, fx, fy, tx, ty):
    length = img.width if img.width > img.height else img.height
    c = img.crop((fx, fy, tx, ty))
    c = __mosaic_blurry(c, length)
    img.paste(c, (fx, fy, tx, ty))
    return img


def mosaic_blurry(img):
    img = str(img)
    with Image.open(img) as image:
        box_list = detector(img)
        for box in box_list:
            image = _mosaic_blurry(
                image,
                box[0],
                box[1],
                box[0] + box[2],
                box[1] + box[3],
            )
            image.save(img)
        # revert_img_info(None, img, image.info)


# -------------------- #


def _mosaic_pixel(image, region, block_size):
    left, upper, right, lower = region

    cropped_image = image.crop(region)

    small = cropped_image.resize(
        (int((right - left) / block_size), int((lower - upper) / block_size)), resample=Image.Resampling.NEAREST
    )
    mosaic_image = small.resize(cropped_image.size, Image.Resampling.NEAREST)

    image.paste(mosaic_image, region)
    return image


def mosaic_pixel(img_path):
    img_path = str(img_path)
    box_list = detector(img_path)

    for box in box_list:
        with Image.open(img_path) as pil_img:
            neighbor = int(
                pil_img.width * NEIGHBOR if pil_img.width > pil_img.height else pil_img.height * NEIGHBOR
            )
            image = _mosaic_pixel(pil_img, (box[0], box[1], box[0] + box[2], box[1] + box[3]), neighbor)
            image.save(img_path)
            # revert_img_info(None, img_path, pil_img.info)


# -------------------- #


def mosaic_lines(img_path):
    img_path = str(img_path)
    box_list = detector(img_path)
    with Image.open(img_path) as image:
        draw = ImageDraw.Draw(image)
        for box in box_list:
            x, y, w, h = box

            while y <= box[1] + box[3]:
                xy = [(x, y), (x + w, y)]
                draw.line(
                    xy,
                    fill="black",
                    width=int(10 * 0.35),
                )
                y += int(box[3] * 0.15)
        image.save(img_path)
        # revert_img_info(None, img_path, image.info)


# Gradio Interface Code
def process_images_gradio(input_folder_path, mosaic_type, neighbor_value_ui):
    global NEIGHBOR
    NEIGHBOR = float(neighbor_value_ui)
    logger.info(f"NEIGHBOR value set to: {NEIGHBOR}")

    if not input_folder_path:
        logger.warning("输入文件夹路径为空。")
        return "错误：请输入文件夹路径。"
    
    input_path = Path(input_folder_path)
    if not input_path.is_dir():
        logger.warning(f"提供的路径 '{input_folder_path}' 不是一个有效的目录。")
        return f"错误：文件夹路径 '{input_folder_path}' 不存在或不是一个目录。"

    output_folder = Path("./output")
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录 '{output_folder.resolve()}' 已确保存在。")

    processed_files_count = 0
    error_messages = []

    try:
        # Add .webp to supported formats
        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
        if not image_files:
            logger.info(f"在目录 '{input_folder_path}' 中没有找到支持的图片文件。")
            return "信息：指定文件夹中没有找到支持的图片文件（.png, .jpg, .jpeg, .bmp, .gif, .webp）。"

        logger.info(f"找到 {len(image_files)} 个图片文件在目录: {input_folder_path}")

        for filename in image_files:
            original_img_path = input_path / filename
            output_img_path = output_folder / filename

            try:
                # Copy original image to output directory before processing
                shutil.copy2(original_img_path, output_img_path)
                logger.debug(f"已复制 '{original_img_path}' 到 '{output_img_path}'")

                logger.info(f"正在处理: {output_img_path} 使用模式: {mosaic_type}")
                if mosaic_type == "模糊 (Blurry)":
                    mosaic_blurry(str(output_img_path))
                elif mosaic_type == "像素化 (Pixelated)":
                    mosaic_pixel(str(output_img_path)) # mosaic_pixel will use the global NEIGHBOR
                elif mosaic_type == "线条 (Lines)":
                    mosaic_lines(str(output_img_path))
                logger.info(f"处理完成: {output_img_path}")
                processed_files_count += 1
            except Exception as e:
                logger.error(f"处理文件 '{filename}' 时发生错误: {e}")
                error_messages.append(f"文件 '{filename}': {str(e)}")
        
        status_message = f"处理完成！ {processed_files_count} / {len(image_files)} 张图片已成功处理并保存到 {output_folder.resolve()}."
        
        if error_messages:
            status_message += "\n处理部分文件时发生以下错误:\n" + "\n".join(error_messages)
        return status_message

    except Exception as e:
        logger.error(f"处理过程中发生严重错误: {e}")
        return f"处理过程中发生严重错误: {str(e)}"

if __name__ == "__main__":
    # Ensure NEIGHBOR is defined globally if not already
    # This handles cases where the script might be structured differently or NEIGHBOR isn't at the top level.
    if 'NEIGHBOR' not in globals():
        NEIGHBOR = 0.1 # Default value, matching the original script

    iface = gr.Interface(
        fn=process_images_gradio,
        inputs=[
            gr.Textbox(label="输入图片文件夹路径 (Input Image Folder Path)", placeholder="例如: C:\\Users\\YourName\\Pictures" ),
            gr.Radio(
                choices=["模糊 (Blurry)", "像素化 (Pixelated)", "线条 (Lines)"],
                label="选择打码模式 (Mosaic Mode)",
                value="像素化 (Pixelated)" # Default selection
            ),
            gr.Number(label="NEIGHBOR 值 (用于像素化模式)", value=NEIGHBOR, minimum=0.0001, maximum=0.1, step=0.0001,
                      info="调整像素化马赛克的强度。值越小，马赛克格子越大。建议范围 0.001 - 0.05。默认值基于脚本初始设定。")
        ],
        outputs=gr.Textbox(label="状态 (Status)", lines=7, interactive=False),
        title="xCensorNing 图片打码工具 (Image Censoring Tool)",
        description=(
            "1. 输入包含待处理图片的文件夹的 **完整路径**。\n"
            "2. 选择一种 **打码模式**。\n"
            "3. 如果选择 **像素化模式**，可以调整 **NEIGHBOR 值** (值越小，马赛克格子越大)。\n"
            "4. 点击 **Submit** 按钮开始处理。\n"
            "处理后的图片将保存到脚本所在目录下的 `output` 文件夹中。请确保目标文件夹可写。"
        ),
        allow_flagging="never",
        theme=gr.themes.Soft(),
        live=False # Process only on submit
    )
    logger.info("启动 Gradio 界面，请在浏览器中打开 http://127.0.0.1:2333")
    iface.launch(server_name="127.0.0.1", server_port=2333)

