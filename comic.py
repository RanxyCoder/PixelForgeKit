import shutil
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import zipfile
from ebooklib import epub
from lxml import etree
from .core import *
from PIL import Image


def Comic_SizeAlign(comic_path: str, keep_original: bool = False) -> None:
    if keep_original:
        original_comics_folder = os.path.join(comic_path, "original_comics")
        os.makedirs(original_comics_folder, exist_ok=True)

    comics = image_forms_classify(comic_path, [".jpg"])
    comics_path = [
        (os.path.join(comic_path, _), os.path.join(original_comics_folder, _)) for _ in comics
    ]
    sizes = []


def Kmoe_Epub2Img(epub_path, output_dir):
    def get_info(input_1, input_2):
        book = epub.read_epub(input_1)
        title = book.get_metadata('DC', 'title')[0][0]
        book_path = str(input_2 + os.sep + title)
        return book_path, title

    extract_path = get_info(epub_path, output_dir)[0]
    # prepare to read from .epub file
    with zipfile.ZipFile(epub_path, mode='r') as _zip:
        # 读取html文件
        for _name in _zip.namelist():
            if _name[-5:] == '.html':
                text = _zip.read(_name)
                xml = etree.HTML(text)
                # 读取 img 对应的图片路径
                img_path = xml.xpath('//img/@src')[0][3:]
                img_ext = xml.xpath('//img/@src')[0][-4:]
                # 读取页码信息
                page_info = xml.xpath('/html/head/title/text()')[0]
                if img_ext == '.jpg':
                    try:
                        # 解压缩图片
                        _zip.extract(img_path, extract_path)
                        # 按编号顺序改名
                        os.rename(extract_path + '/' + img_path, extract_path + '/' + page_info + '.jpg')
                    except Exception as e:
                        print(e)
                elif img_ext == '.png':
                    try:
                        # 解压缩图片
                        _zip.extract(img_path, extract_path)
                        # 按编号顺序改名
                        os.rename(extract_path + '/' + img_path, extract_path + '/' + page_info + '.png')
                    except Exception as e:
                        print(e)
                elif '.' not in img_ext:
                    pass
                else:
                    print('不支持的图片格式！！')
        # 删除已经为空的image文件夹
        shutil.rmtree(extract_path + '/' + 'image')


class Comic_Editor(QMainWindow):
    def __init__(self, folder_path, vol):
        super().__init__()

        self.folder = folder_path
        self.vol = vol
        if type(vol) is int:
            self.prefix = f"Volume{vol:02d}_Cover"
        elif vol is None:
            self.prefix = "Page"

        # 加载图片列表
        self.images = image_forms_classify(self.folder, abs_path=True)
        self.page_index = 0  # 当前页索引
        self.page_number = 1  # 编号计数
        self.label_filename = None
        self.label_right = None
        self.label_left = None
        self.btn_switch = None

        # 初始化UI
        self.init_ui()
        self.show_images()

    def init_ui(self):
        self.setWindowTitle("Comic Editor")
        self.resize(1500, 1000)

        # 设置背景色
        self.setStyleSheet("background-color: #d0f0c0;")

        # 主部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 文件名显示区（在图片区上方）
        self.label_filename = QLabel("当前文件：")
        self.label_filename.setAlignment(Qt.AlignCenter)
        self.label_filename.setStyleSheet("color: black; font-size: 18px; font-weight: bold;")

        # 两个图片显示区
        self.label_right = QLabel("右页")
        self.label_left = QLabel("左页")
        self.label_right.setAlignment(Qt.AlignCenter)
        self.label_left.setAlignment(Qt.AlignCenter)

        # 标签颜色
        self.label_right.setStyleSheet("color: yellow;")
        self.label_left.setStyleSheet("color: yellow;")

        # 垂直布局：文件名在上，图片区在下
        img_with_label_layout = QVBoxLayout()
        img_with_label_layout.addWidget(self.label_filename)
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.label_left, 1)
        img_layout.addWidget(self.label_right, 1)
        img_with_label_layout.addLayout(img_layout)

        # 按钮样式
        btn_style = """
            QPushButton {
                background-color: #4682b4;
                color: yellow;
                font-size: 16px;
                height: 50px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #5a9bd3;
            }
            QPushButton:disabled {
                background-color: #7a7a7a;
                color: #cccccc;
            }
        """

        # 按钮区
        btn_rename = QPushButton("单页编号")
        btn_merge = QPushButton("合页编号")
        btn_next = QPushButton("NEXT")

        for btn in (btn_rename, btn_merge, btn_next):
            btn.setStyleSheet(btn_style)

        btn_rename.clicked.connect(self.rename_current_right)
        btn_merge.clicked.connect(self.merge_and_rename)
        btn_next.clicked.connect(self.next_group)

        btn_layout = QVBoxLayout()

        if type(self.vol) is int:
            self.btn_switch = QPushButton("C to P")
            self.btn_switch.setStyleSheet(btn_style)
            self.btn_switch.clicked.connect(self.switch)
            btn_layout.addWidget(self.btn_switch)

        btn_layout.addWidget(btn_rename)
        btn_layout.addWidget(btn_merge)
        btn_layout.addWidget(btn_next)
        btn_layout.addStretch(1)
        btn_layout.setSpacing(50)

        # 总布局
        main_layout = QHBoxLayout()
        main_layout.addLayout(img_with_label_layout, 4)
        main_layout.addLayout(btn_layout, 1)

        central_widget.setLayout(main_layout)

    def switch(self):
        self.page_number = 1
        self.prefix = f"Volume{self.vol:02d}_Page"
        self.btn_switch.setDisabled(True)

    def show_images(self):
        """显示当前两张图：右=当前，左=下一张"""
        if self.page_index < len(self.images):
            pixmap = QPixmap(self.images[self.page_index]).scaled(
                666, 1000, Qt.KeepAspectRatio)
            self.label_right.setPixmap(pixmap)
            # 显示当前右图的文件名
            filename = os.path.basename(self.images[self.page_index])
            self.label_filename.setText(f"下一张处理文件：↓↓ ===================================== 当前文件：{filename}")
        else:
            self.label_right.clear()
            self.label_filename.setText("当前文件：无")

        if self.page_index + 1 < len(self.images):
            pixmap = QPixmap(self.images[self.page_index + 1]).scaled(
                666, 1000, Qt.KeepAspectRatio)
            self.label_left.setPixmap(pixmap)
        else:
            self.label_left.clear()

    def rename_current_right(self):
        if self.page_index >= len(self.images):
            return

        img_path = self.images[self.page_index]
        # 转码为 jpg
        FormatConversion(img_path, "jpg")
        base_dir = os.path.dirname(img_path)
        new_name = None
        if "Cover" in self.prefix:
            new_name = f"{self.prefix}{self.page_number:02d}.jpg"
        elif "Page" in self.prefix:
            new_name = f"{self.prefix}{self.page_number:03d}.jpg"
        new_path = os.path.join(base_dir, new_name)

        # 原始转码文件可能叫 xxx.jpg，把它重命名
        converted_path = os.path.splitext(img_path)[0] + ".jpg"
        if os.path.exists(converted_path):
            os.rename(converted_path, new_path)
            self.images[self.page_index] = new_path

        self.page_number += 1
        self.page_index += 1
        self.show_images()

    def merge_and_rename(self):
        if self.page_index + 1 >= len(self.images):
            return

        img_right = Image.open(self.images[self.page_index])
        img_left = Image.open(self.images[self.page_index + 1])

        merged = concatenate_images([img_left, img_right], direction="h")

        base_dir = os.path.dirname(self.images[self.page_index])
        new_name = None
        if "Cover" in self.prefix:
            new_name = f"{self.prefix}{self.page_number:02d}~{self.page_number + 1:02d}.jpg"
        elif "Page" in self.prefix:
            new_name = f"{self.prefix}{self.page_number:03d}~{self.page_number + 1:03d}.jpg"
        new_path = os.path.join(base_dir, new_name)

        merged.save(new_path, quality=95)
        os.remove(self.images[self.page_index])
        os.remove(self.images[self.page_index + 1])
        self.page_number += 2
        self.page_index += 2
        self.show_images()

    def next_group(self):
        self.page_index += 1
        self.show_images()

    @staticmethod
    def run(folder_path, vol=None):
        app = QApplication(sys.argv)
        editor = Comic_Editor(folder_path, vol=vol)
        editor.show()
        sys.exit(app.exec_())
