import glob
import os
from utils.image_page import ImagePage  # 假设你把原始代码保存为image_page.py

# 配置路径
image_dir = "images/*.png"
mask_dir = "masks/"
output_html = "visualization.html"

# 获取所有原始图像路径
images = sorted(glob.glob(image_dir))

# 创建HTML页面
html = ImagePage("图像与掩码展示", output_html)

# 添加图像对
for img_path in images:
    # 获取不带扩展名的文件名
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # 构建对应的掩码路径（假设文件名相同）
    mask_path = os.path.join(mask_dir, base_name + ".png")

    # 检查掩码文件是否存在
    if not os.path.exists(mask_path):
        print(f"警告: 找不到 {mask_path} 的掩码文件")
        continue

    # 添加到HTML表格
    html.add_table((
        (img_path, "原始图像"),
        (mask_path, "对应掩码")
    ))

# 生成HTML文件
html.write_page()
print(f"结果已保存到 {output_html}")