import io
import img2pdf
from pathlib import Path
from pypinyin import lazy_pinyin
from rich.progress import Progress
from typing import Optional, List
from typing import Tuple, Union
from PIL import Image, ImageEnhance


def image_forms_classify(folder_path: str, extensions: Optional[List[str]] = None) -> List[Path]:
    """获取指定文件夹中的特定格式图像文件列表（自然排序）"""

    if extensions is None:
        extensions = ["jpg", "jpeg", "bmp", "png", "webp"]

    extensions = [ext.lower().lstrip(".") for ext in extensions]

    folder = Path(folder_path)

    files = [f for f in folder.iterdir() if f.suffix.lower().lstrip(".") in extensions]

    files.sort(key=lambda f: lazy_pinyin(f.name))
    return files


def save_pillow_image(image: Image.Image,
                      original_path: str | Path,
                      suffix: str = "_edited",
                      keep_original: bool = True,
                      save_quality: int = None) -> Path:
    """
    保存图像并根据需要删除原始文件。

    Args:
        image (Image.Image): 要保存的 PIL 图像对象。
        original_path (str | Path): 原始图像文件路径。
        suffix (str): 新图像文件名中添加的后缀，默认为 "_edited"。
        keep_original (bool): 是否保留原始文件。如果为 False，将删除原始文件并重命名新文件。
        save_quality : 质量参数

    Returns:
        Path: 最终保存的图像路径。

    Raises:
        ValueError: 如果 `image` 不是有效的图像对象或 `original_path` 无效。
        OSError: 如果在保存或删除文件时出现问题。
    """
    if not isinstance(image, Image.Image):
        raise ValueError("格式传入错误，图片格式应为PIL")

    original_path = Path(original_path)
    if not original_path.is_file():
        raise ValueError(f"路径无效: {original_path}")

    # 生成新文件路径
    new_image_path = original_path.with_stem(original_path.stem + suffix)

    try:
        suffix_lower = original_path.suffix.lower()
        if suffix_lower in (".jpg", ".jpeg"):
            if save_quality is None:
                save_quality = 90
            image.save(new_image_path, quality=save_quality)
        elif suffix_lower == ".png":
            if save_quality is None:
                save_quality = 3
            image.save(new_image_path, compress_level=save_quality)
        else:
            image.save(new_image_path)
        image.close()

        # 是否保留原始文件
        if not keep_original:
            original_path.unlink()
            new_image_path.replace(original_path)
            return original_path

        return new_image_path

    except OSError as error:
        raise OSError(f"保存图片失败: {error}")


def split_image_segments(image: Image.Image, direction: str, segments: int) -> List[Image.Image]:
    """按指定方向和片段数量将图像分割为多个部分。

    Args:
        image (Image.Image): 要分割的 PIL 图像对象。
        direction (str): 分割方向，'h' 表示水平分割，'v' 表示垂直分割。
        segments (int): 将图像分割的片段数量。

    Returns:
        List[Image.Image]: 分割后的图像片段列表。

    Raises:
        ValueError: 如果传入的方向无效。
    """
    width, height = image.size

    if direction == "h":
        segment_width = width // segments
        bounding_boxes = [(i * segment_width, 0, (i + 1) * segment_width, height) for i in range(segments)]
        cropped_images = [image.crop(box) for box in bounding_boxes]
    elif direction == "v":
        segment_height = height // segments
        bounding_boxes = [(0, i * segment_height, width, (i + 1) * segment_height) for i in range(segments)]
        cropped_images = [image.crop(box) for box in bounding_boxes]
    else:
        raise ValueError("无效的方向参数，方向参数应为'h'（横向）或者'v'（纵向）。")

    return cropped_images


def concatenate_images(images: List[Image.Image],
                       direction: str,
                       pieces_per_group: Optional[int] = None) -> Union[Image.Image, List[Image.Image]]:
    """
    按指定的方向将多个图片拼接在一起，支持水平（h）和垂直（v）拼接。

    Args:
        images (List[Image.Image]): 要拼接的图像列表。
        direction (str): 拼接方向，'h'表示水平拼接，'v'表示垂直拼接。
        pieces_per_group (Optional[int]): 每组拼接的图像数量。如果为None，则拼接所有图像。

    Returns:
        Union[Image.Image, List[Image.Image]]: 拼接后的图像，如果只有一张拼接结果则返回单一图像，否则返回拼接后的图像列表。

    Raises:
        ValueError: 如果指定的拼接方向不是 'h' 或 'v'。
    """
    if pieces_per_group is None:
        pieces_per_group = len(images)

    image_groups = [images[i: i + pieces_per_group] for i in range(0, len(images), pieces_per_group)]

    concatenated_images = []

    for group in image_groups:
        if direction == "h":
            total_width = sum(img.width for img in group)  # 水平方向拼接需要计算总宽度
            max_height = max(img.height for img in group)  # 高度取最高的一张图片
            concatenated_image = Image.new("RGB", (total_width, max_height), color="white")  # 创建一个新画布
            x_offset = 0
            for img in group:
                concatenated_image.paste(img, (x_offset, 0))  # 按照顺序拼接图片
                x_offset += img.width  # 更新下一个图片的X坐标
        elif direction == "v":
            max_width = max(img.width for img in group)  # 垂直方向拼接需要计算最大宽度
            total_height = sum(img.height for img in group)  # 高度取所有图片的总和
            concatenated_image = Image.new("RGB", (max_width, total_height), color="white")  # 创建一个新画布
            y_offset = 0
            for img in group:
                concatenated_image.paste(img, (0, y_offset))  # 按照顺序拼接图片
                y_offset += img.height  # 更新下一个图片的Y坐标
        else:
            raise ValueError("无效的方向参数，方向参数应为'h'（横向）或者'v'（纵向）。")  # 如果方向不合法，抛出异常

        concatenated_images.append(concatenated_image)

    # 如果只有一张拼接结果，则返回该结果，否则返回拼接后的所有结果
    if len(concatenated_images) == 1:
        return concatenated_images[0]
    else:
        return concatenated_images


def fill_image_with_margin(input_image: Image.Image,
                           margin_color: Optional[tuple] = None,
                           target_size: Optional[tuple] = None) -> Image.Image:
    """
    给定图像添加边距并调整大小。如果指定了边距颜色，则用该颜色填充，如果没有指定，则使用透明背景填充。

    Args:
        input_image (Image.Image): 输入的原始图像。
        margin_color (Optional[tuple]): 填充的边距颜色，默认为透明。格式为 (R, G, B) 或 (R, G, B, A)。
        target_size (Optional[tuple]): 目标图像大小。若未提供，则调整为足够容纳原图的大小。

    Returns:
        Image.Image: 添加边距并调整大小后的图像。
    """
    original_width, original_height = input_image.size

    # 如果没有指定目标大小，则使用最大边作为新的尺寸
    if target_size is None:
        new_width = new_height = max(original_width, original_height)
        target_size = (new_width, new_height)

    # 创建一个新的图像，背景填充为 margin_color，如果没有提供则默认为透明背景
    if margin_color is None:
        filled_image = Image.new("RGBA", target_size)  # 透明背景
    else:
        filled_image = Image.new("RGB", target_size, color=margin_color)  # 填充颜色

    # 计算原图放置的位置，确保它居中
    x_offset = (target_size[0] - original_width) // 2
    y_offset = (target_size[1] - original_height) // 2

    # 将原图粘贴到新的背景上
    filled_image.paste(input_image, (x_offset, y_offset))

    return filled_image


def detect_image_margins(input_image: Image.Image,
                         left: bool, top: bool, right: bool, bottom: bool,
                         margin_color: tuple,
                         threshold: int = 64) -> Tuple[int, int, int, int]:
    """
    检测图像边缘（空白区域），并返回非空白区域的边界坐标。

    Args:
        input_image (Image.Image): 输入的图像对象。
        left (bool): 是否检测左边缘。
        top (bool): 是否检测顶部边缘。
        right (bool): 是否检测右边缘。
        bottom (bool): 是否检测底部边缘。
        margin_color (tuple): 用于识别边缘空白区域的颜色，通常是背景色，例如 (255, 255, 255)。
        threshold (int): 用于判断颜色差异的阈值，默认值为 64。

    Returns:
        tuple: 返回四个边界坐标（left_x, top_y, right_x, bottom_y）。
    """

    def color_difference(color1, color2):
        """
        计算两个颜色之间的差异，使用欧氏距离方法。
        """
        return sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)) ** 0.5

    width, height = input_image.size
    left_x = width if left else 0
    top_y = height if top else 0
    right_x = 0 if right else width
    bottom_y = 0 if bottom else height

    if left:
        for i in range(height):
            for j in range(width):
                pixel = input_image.getpixel((j, i))
                if color_difference(pixel, margin_color) > threshold:
                    if j < left_x:
                        left_x = j
                    break

    if top:
        for j in range(width):
            for i in range(height):
                pixel = input_image.getpixel((j, i))
                if color_difference(pixel, margin_color) > threshold:
                    if i < top_y:
                        top_y = i
                    break

    if right:
        for i in range(height):
            for j in range(width - 1, -1, -1):
                pixel = input_image.getpixel((j, i))
                if color_difference(pixel, margin_color) > threshold:
                    if j > right_x:
                        right_x = j
                    break

    if bottom:
        for j in range(width):
            for i in range(height - 1, -1, -1):
                pixel = input_image.getpixel((j, i))
                if color_difference(pixel, margin_color) > threshold:
                    if i > bottom_y:
                        bottom_y = i
                    break

    return left_x, top_y, right_x, bottom_y


def remove_image_margins(input_image: Image.Image,
                         left: bool, top: bool, right: bool, bottom: bool,
                         margin_color: tuple = (255, 255, 255),
                         threshold: int = 64) -> Image.Image:
    """
    去除图像的空白边缘区域，返回裁剪后的图像。

    Args:
        input_image (Image.Image): 输入的图像对象。
        left (bool): 是否去除左边缘。
        top (bool): 是否去除顶部边缘。
        right (bool): 是否去除右边缘。
        bottom (bool): 是否去除底部边缘。
        margin_color (tuple): 用于识别空白区域的颜色，默认是白色 (255, 255, 255)。
        threshold (int): 用于判断颜色差异的阈值，默认值为 64。

    Returns:
        Image.Image: 裁剪后的图像。
    """
    margins = detect_image_margins(input_image, left, top, right, bottom, margin_color, threshold)
    cropped_image = input_image.crop(margins)
    return cropped_image


def image_margins_fill(image: Image.Image,
                       margin: Optional[tuple] = None,
                       size: Optional[tuple] = None) -> Image.Image:
    """为图像添加边框填充。

    Args:
        image (Image.Image): 要填充的PIL图像对象。
        margin (Optional[tuple]): 填充的颜色，默认为透明。
        size (Optional[tuple]): 目标大小，默认为方形。

    Returns:
        Image.Image: 添加填充后的图像。
    """
    width, height = image.size
    if size is None:
        max_length = max(width, height)
        size = (max_length, max_length)
    if margin is None:
        filled_img = Image.new("RGBA", size)
    else:
        filled_img = Image.new("RGB", size, color=margin)

    anchor = ((size[0] - width) // 2, (size[1] - height) // 2)
    filled_img.paste(image, anchor)
    return filled_img


def enhance_image_contrast(input_image: Image.Image, contrast_factor: float = 1.25) -> Image.Image:
    """
    增强图像的对比度。

    Args:
        input_image (Image.Image): 输入的原始图像。
        contrast_factor (float): 对比度增强因子，默认为 1.25。值大于 1 增强对比度，值小于 1 减弱对比度。

    Returns:
        Image.Image: 增强对比度后的图像。
    """
    contrast_enhancer = ImageEnhance.Contrast(input_image)
    enhanced_image = contrast_enhancer.enhance(contrast_factor)
    return enhanced_image


def FormatConversion(image_path: str, target_format: str, quality_jpg: int = 90, quality_png: int = 3,
                     log: bool = True) -> None:
    """
    将图像文件转换为指定格式，可选择删除原始文件。

    Args:
        image_path (str): 输入图像文件路径。
        target_format (str): 目标图像格式，如 'jpg' 或 'png'。
        quality_jpg (int): JPEG 保存质量，默认 90。
        quality_png (int): PNG 压缩等级，默认 3。
        log (bool): 是否打印日志。

    Raises:
        ValueError: 输入文件或目标格式不支持。
        OSError: 文件操作失败。
    """
    images = Path(image_path)
    if not images.is_file():
        raise ValueError(f"输入路径无效: {images}")

    supported_formats = [".jpg", ".jpeg", ".bmp", ".png", ".webp"]
    if images.suffix.lower() not in supported_formats:
        raise ValueError(f"输入文件格式不支持: {images.suffix}")

    target_format = target_format.lower()
    if target_format not in ["jpg", "png"]:
        raise ValueError(f"输出格式不支持: {target_format}")

    new_image_path = images.with_suffix(f".{target_format}")

    try:
        with Image.open(images) as image:
            # JPEG 不支持透明通道
            if target_format == "jpg" and image.mode != "RGB":
                image = image.convert("RGB")
            if target_format == "jpg":
                image.save(new_image_path, quality=quality_jpg)
            elif target_format == "png":
                image.save(new_image_path, compress_level=quality_png)

        # 如果原始文件格式与目标不同，则删除原文件
        if images.suffix.lower() != f".{target_format}":
            images.unlink()

        if log:
            print(f"图像成功转换 {target_format}: {new_image_path}")

    except OSError as error:
        raise OSError(f"转换过程中出现错误: {error}")


def ContrastEnhancement(img_path: str, keep_original: bool = False) -> None:
    """
    增强图像对比度并保存结果。

    Args:
        img_path (str): 输入图像文件路径。
        keep_original (bool): 是否保留原始图像。默认值为 False。
    """
    image = Image.open(img_path)
    image = enhance_image_contrast(image)
    save_pillow_image(image, img_path, "_contrast_enhanced", keep_original)


def MarginRemoval(img_path: str,
                  left: bool, top: bool, right: bool, bottom: bool,
                  margin: tuple = (255, 255, 255),
                  threshold: int = 64,
                  keep_original: bool = False) -> None:
    """
    移除图像的指定边缘并保存结果。

    Args:
        img_path (str): 输入图像文件路径。
        left (int): 左边缘像素数。
        top (int): 上边缘像素数。
        right (int): 右边缘像素数。
        bottom (int): 下边缘像素数。
        margin (tuple): 用于比较的边缘颜色，默认为 (255, 255, 255)（白色）。
        threshold (int): 色差阈值，默认为 64。
        keep_original (bool): 是否保留原始图像。默认值为 False。
    """
    image = Image.open(img_path)
    image = remove_image_margins(image, left, top, right, bottom, margin_color=margin, threshold=threshold)
    save_pillow_image(image, img_path, "_margin_removed", keep_original)


def PackageImagesToPDF(
        folder_path: str,
        output_pdf_path: Optional[str] = None,
        dpi: int = 300,
        compress: Optional[int] = None,
        prefix_index=None,
        checkpoint=None
) -> None:
    """
    将指定文件夹内的所有图片打包为一个 PDF 文件。
    如果指定了 compress 参数，会在打包前压缩图片。

    Args:
        folder_path (str): 包含图像文件的目标文件夹路径。
        output_pdf_path (Optional[str]): 输出 PDF 文件的路径。如果未指定，默认保存到目标文件夹内。
        dpi (int): 输出 PDF 文件的 DPI。默认值为 300。
        compress (Optional[int]): 压缩质量参数，取值范围是 0-100。如果未指定或为 None，则不压缩图像。
        prefix_index: 批处理序列
        checkpoint: feedback
    Returns:
        None
    """
    if prefix_index and len(prefix_index) == 2 and prefix_index[0] <= prefix_index[1]:
        index_head = f"({prefix_index[0]}/{prefix_index[1]})"
    else:
        index_head = ""

    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"⚠️ 文件夹不存在: {folder}")
        return

    image_files = image_forms_classify(folder_path)
    if not image_files:
        print("⚠️ 未找到任何图像文件。")
        return

    output = Path(output_pdf_path) if output_pdf_path else folder
    if output.is_dir():
        output_pdf_folder = output
        output_pdf_folder.mkdir(parents=True, exist_ok=True)
        output_pdf_path = output_pdf_folder / f"{folder.name}.pdf"
    elif output.suffix.lower() == ".pdf":
        output.parent.mkdir(parents=True, exist_ok=True)
        output_pdf_path = str(output)
    else:
        print(f"⚠️ output_pdf_path 非法: {output}")
        return

    if compress:
        images = []
        CYAN = "\033[96m"
        RESET = "\033[0m"
        print(f"{CYAN}⚙️ {folder_path}{RESET}")
        with Progress() as progress:
            task = progress.add_task(f"[cyan]{' ' * 3}质量 = {compress} 压缩图片中...",
                                     total=len(image_files))

            n = 1
            for img_path in image_files:
                with Image.open(str(img_path)) as img:
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    with io.BytesIO() as img_byte_arr:
                        img.save(img_byte_arr, format="JPEG", quality=compress, optimize=True)
                        images.append(img_byte_arr.getvalue())
                if checkpoint:
                    checkpoint(n, len(image_files))
                n += 1
                progress.update(task, advance=1)

    else:
        images = [str(p) for p in image_files]

    try:
        with open(output_pdf_path, "wb") as f:
            f.write(img2pdf.convert(images, dpi=dpi))
        print(f"✅ {index_head} PDF文件已保存到 {output_pdf_path}")

    except Exception as e:
        print(f"❌ {index_head} 打包图片为 PDF 出错：{e}")


# def CoverResizeToPNG(image_path: str, target_size: Tuple[int, int] = (707, 1000)) -> None:
#     """
#     调整给定图像的大小并填充边距，以便用于图标生成。
#
#     参数:
#     - image_path (str): 输入图像的路径。
#     - target_size (Tuple[int, int]): 调整后的图像尺寸，默认为 (707, 1000)。
#
#     返回:
#     - None: 处理后的图像将保存到与输入图像相同位置的输出文件中。
#     """
#     try:
#         with Image.open(image_path) as image:
#             resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
#             filled_image = fill_image_with_margin(resized_image)
#             output_path = f"{os.path.splitext(image_path)[0]}_pre-icon.png"
#             filled_image.save(output_path)
#             print(f"图像已保存到: {output_path}")
#     except Exception as e:
#         print(f"处理图像 {image_path} 时出错: {e}")


def PageResizeRatio(image_path: str, w_h_ratio: float, size: Tuple):
    ...
