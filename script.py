from .core import *


def list_pinyin(path: Path):
    """按拼音排序返回目录下所有文件和文件夹（Path对象）"""
    return sorted(path.iterdir(), key=lambda f: lazy_pinyin(f.name))


def Script_FormatConversion(root_folder: str, target_format: str, feedback=None) -> None:
    """
    批量转换文件夹中所有图像的格式。

    Args:
        root_folder (str): 包含图像文件的文件夹路径。
        target_format (str): 要转换的目标格式，如 'jpg' 或 'png'。
        feedback: feedback
    Returns:
        None
    """
    image_files = image_forms_classify(root_folder)
    _all = len(image_files)
    with Progress() as progress:
        task = progress.add_task(f"[lime](0/{_all}){root_folder}格式转换中...", total=_all)
        n = 1
        for image_file in image_files:
            progress.update(task, description=f"[lime]({n}/{_all}){image_file.name}格式转换中...")
            try:
                FormatConversion(str(image_file), target_format, log=False)
            except (ValueError, OSError) as e:
                print(f"转换失败: {image_file.name}: {e}")
            if feedback:
                feedback(n, len(image_files))
            n += 1
            progress.update(task, advance=1)


def Script_PackageImages2PDF(root_folder: str,
                             output_folder: str,
                             dpi: int = 300,
                             compress: Optional[int] = None,
                             checkpoint0=None,
                             checkpoint1=None) -> None:
    """
    遍历指定文件夹及其子文件夹，将每个子文件夹内的图像打包为 PDF 并保存。

    Args:
        root_folder (str): 需要遍历的根文件夹路径。
        output_folder (str): 输出 PDF 文件的根目录。
        dpi (int): 输出 PDF 文件的 DPI。默认值为 300。
        compress (Optional[int]): 压缩质量参数，取值范围是 0-100。如果未指定或为 None，则不压缩图像。
        checkpoint0: feedback
        checkpoint1: feedback
    Returns:
        None
    """
    valid_extensions = [".jpg", ".png"]

    def process_directory(input_path, output_path, _dpi, _compress, n=1):
        items = list_pinyin(input_path)
        for item in items:
            if item.is_dir():
                # 检查目录中是否有符合格式的图像文件
                has_images = any(f.is_file() and f.suffix.lower() in valid_extensions for f in item.iterdir())

                if has_images:
                    # 如果目录中包含图像，将其打包为 PDF
                    PackageImagesToPDF(str(item), str(output_path), dpi=_dpi, compress=_compress,
                                       prefix_index=(n, all_num), checkpoint=checkpoint0)
                    if checkpoint1:
                        checkpoint1(n, all_num, str(item))
                    print("\n")
                    n += 1
                else:
                    # 如果目录中不包含图像，递归处理子目录
                    subdir_output_path = output_path / item.name
                    subdir_output_path.mkdir(parents=True, exist_ok=True)
                    n = process_directory(item, subdir_output_path, _dpi, _compress, n)
        return n

    if root_folder == output_folder:
        print("输入路径不能与输出路径相同。")
        return

    def calculate_directory(input_path):
        num = 0
        for item in input_path.iterdir():
            if item.is_dir():
                has_images = any(
                    f.is_file() and f.suffix.lower() in valid_extensions
                    for f in item.iterdir()
                )
                if has_images:
                    num += 1
                else:
                    num += calculate_directory(item)
        return num

    # 创建最终输出路径
    final_output_path = Path(output_folder) / Path(root_folder).name
    final_output_path.mkdir(parents=True, exist_ok=True)
    all_num = calculate_directory(Path(root_folder))
    process_directory(Path(root_folder), Path(final_output_path), dpi, compress)
