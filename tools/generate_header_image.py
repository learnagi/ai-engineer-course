#!/usr/bin/env python3
"""
教程题图生成工具

使用方法:
python generate_header_image.py --title "机器学习基础" --output "foundations/images/header.png"
"""

import argparse
import os
import sys
import time
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

class HeaderImageGenerator:
    """教程题图生成器"""
    
    def __init__(self, api_base: str, api_key: str):
        """初始化生成器
        
        Args:
            api_base: OpenAI API代理基础URL
            api_key: OpenAI API密钥
        """
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def _get_english_title(self, title: str) -> str:
        """获取英文标题
        
        Args:
            title: 中文标题
            
        Returns:
            str: 英文标题
        """
        title_map = {
            "机器学习基础": "Machine Learning Foundations",
            "神经网络": "Neural Networks",
            "深度学习": "Deep Learning",
            "计算机视觉": "Computer Vision",
            "自然语言处理": "Natural Language Processing",
            "强化学习": "Reinforcement Learning",
        }
        return title_map.get(title, title)

    def _generate_prompt(self, title: str) -> str:
        """生成DALL-E提示词
        
        Args:
            title: 教程标题
            
        Returns:
            str: 完整的提示词
        """
        english_title = self._get_english_title(title)
        
        branding_desc = (
            "In the bottom right corner, create a distinct brand element: "
            "a modern tech logo consisting of 'AGI01' text in a bold, geometric font "
            "inside a bright blue (#0066FF) hexagonal container with white text. "
            "The hexagon should be solid-filled and approximately 60x60 pixels. "
            "Make this brand element stand out clearly against the background "
            "like a professional tech company logo or watermark."
        )
        
        character_desc = (
            "A tiny programmer character (taking up only 10% of the image height) "
            "positioned in the bottom-left portion of the image. "
            "The character should be in modern flat illustration style without gradients, "
            "wearing a simple blue hoodie and jeans. Use clean, solid colors "
            "with minimal details, like illustrations from tech documentation. "
            "The character should be a simple, cheerful stick figure style "
            "that fits the minimal aesthetic."
        )
        
        style_desc = (
            "Create in a modern flat illustration style with absolutely no gradients, "
            "using clean solid colors and simple geometric shapes. "
            "Use a fresh, vibrant color palette: "
            "light blue (#61DAFB), coral pink (#FF6B6B), "
            "and crisp off-white (#F8F9FA) background. "
            f"Include the text '{english_title}' in a bold, modern sans-serif font "
            "positioned prominently in the upper right portion."
        )
        
        # 根据不同主题调整场景描述
        if "机器学习" in title or "ML" in title:
            scene_desc = (
                "The main focus should be on large, simple data visualization elements "
                "in the upper 70% of the image - clean bar charts, scatter plots, and "
                "decision trees in solid colors. The tiny programmer character stands "
                "at the bottom left, looking up at the floating data elements. "
                "Keep ample whitespace around the AGI01 brand mark in bottom right."
            )
        elif "神经网络" in title or "Neural" in title:
            scene_desc = (
                "Fill the upper 70% of the space with a simplified neural network diagram "
                "using solid-colored circles and clean connecting lines. No gradients. "
                "The tiny programmer character stands at the bottom left, observing the network. "
                "Keep ample whitespace around the AGI01 brand mark in bottom right."
            )
        elif "深度学习" in title or "Deep" in title:
            scene_desc = (
                "Show clean, layered network architecture in the upper portion using "
                "simple geometric shapes and solid colors. The tiny programmer character "
                "stands at the bottom left, dwarfed by the network structure. "
                "Keep ample whitespace around the AGI01 brand mark in bottom right."
            )
        else:
            scene_desc = (
                "Arrange clean, geometric shapes and simple icons related to the topic "
                "in the upper 70% of the image. The tiny programmer character stands "
                "at the bottom left, integrated into the scene but clearly visible. "
                "Keep ample whitespace around the AGI01 brand mark in bottom right."
            )
        
        prompt = (
            f"Create a wide-format educational illustration for '{english_title}' "
            "in a modern flat design style like Stripe's documentation illustrations. "
            f"{character_desc} {scene_desc} {style_desc} {branding_desc} "
            "The composition should be balanced for a banner format (1024x400). "
            "Use absolutely no gradients or shadows - only clean, solid colors. "
            "Keep the design minimal and professional. "
            "The AGI01 brand mark in the bottom right MUST be clearly visible "
            "as a distinct hexagonal logo element. Make it stand out like "
            "a professional company watermark or badge."
        )
        
        return prompt
    
    def generate_image(self, title: str) -> bytes:
        """生成图片
        
        Args:
            title: 教程标题
            
        Returns:
            bytes: 图片数据
            
        Raises:
            Exception: API调用失败时抛出异常
        """
        prompt = self._generate_prompt(title)
        
        # 调用DALL-E API
        response = requests.post(
            f"{self.api_base}/v1/images/generations",
            headers=self.headers,
            json={
                "model": "dall-e-3",
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
                "quality": "standard",
                "response_format": "url"
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"API调用失败: {response.text}")
        
        # 获取图片URL
        image_url = response.json()['data'][0]['url']
        
        # 下载图片
        image_response = requests.get(image_url)
        if image_response.status_code != 200:
            raise Exception("图片下载失败")
        
        return image_response.content
    
    def _add_brand_mark(self, image: Image.Image) -> Image.Image:
        """添加AGI01品牌标识
        
        Args:
            image: 原始图片
            
        Returns:
            Image.Image: 添加了品牌标识的图片
        """
        from PIL import ImageDraw, ImageFont, ImageFilter, ImageEnhance
        import os

        # 创建绘图对象
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体，如果失败则使用默认字体
        try:
            # 对于macOS，尝试使用系统字体
            font = ImageFont.truetype("/System/Library/Fonts/SFPro-Bold.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
            except:
                # 如果找不到特定字体，使用默认字体
                font = ImageFont.load_default()

        # 品牌标识文本
        brand_text = "AGI01"
        
        # 计算文本大小
        bbox = draw.textbbox((0, 0), brand_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 计算矩形的大小和位置
        padding_x = 24
        padding_y = 12
        rect_width = text_width + padding_x * 2
        rect_height = text_height + padding_y * 2
        
        # 计算右下角位置，留出边距
        margin = 20
        x = image.width - rect_width - margin
        y = image.height - rect_height - margin
        
        # 创建一个新的透明图层用于绘制玻璃效果
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # 绘制主要矩形（半透明白色背景）
        rect_coords = [
            (x, y),
            (x + rect_width, y + rect_height)
        ]
        overlay_draw.rectangle(rect_coords, fill=(255, 255, 255, 80))
        
        # 添加模糊效果
        overlay = overlay.filter(ImageFilter.GaussianBlur(3))
        
        # 绘制边框（更亮的线条）
        border_coords = [
            (x, y),
            (x + rect_width, y),  # 上边
            (x + rect_width, y),
            (x + rect_width, y + rect_height),  # 右边
            (x + rect_width, y + rect_height),
            (x, y + rect_height),  # 下边
            (x, y + rect_height),
            (x, y)  # 左边
        ]
        overlay_draw.line(border_coords, fill=(255, 255, 255, 120), width=1)
        
        # 将玻璃效果图层合并到原图
        image = Image.alpha_composite(image.convert('RGBA'), overlay)
        
        # 计算文本位置使其居中
        text_x = x + (rect_width - text_width) / 2
        text_y = y + (rect_height - text_height) / 2
        
        # 绘制文本（带有轻微发光效果）
        draw = ImageDraw.Draw(image)
        # 发光效果
        for offset in range(2):
            draw.text((text_x - offset, text_y - offset), brand_text, 
                     fill=(255, 255, 255, 50), font=font)
        # 主要文本
        draw.text((text_x, text_y), brand_text, fill=(255, 255, 255, 230), font=font)
        
        return image

    def save_image(self, image_data: bytes, output_path: str):
        """保存图片
        
        Args:
            image_data: 图片数据
            output_path: 输出路径
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 处理图片
        image = Image.open(BytesIO(image_data))
        
        # 调整尺寸为1024x400（保持原始比例并裁剪）
        target_ratio = 1024 / 400
        current_ratio = image.width / image.height
        
        if current_ratio > target_ratio:
            # 图片太宽，需要裁剪宽度
            new_width = int(image.height * target_ratio)
            left = (image.width - new_width) // 2
            image = image.crop((left, 0, left + new_width, image.height))
        else:
            # 图片太高，需要裁剪高度
            new_height = int(image.width / target_ratio)
            top = (image.height - new_height) // 2
            image = image.crop((0, top, image.width, top + new_height))
        
        # 调整到最终尺寸
        image = image.resize((1024, 400), Image.Resampling.LANCZOS)
        
        # 添加品牌标识
        image = self._add_brand_mark(image)
        
        # 保存图片
        image.save(output_path, "PNG", optimize=True)

def main():
    parser = argparse.ArgumentParser(description="生成教程题图")
    parser.add_argument("--title", required=True, help="教程标题")
    parser.add_argument("--output", required=True, help="输出文件路径")
    parser.add_argument("--api-base", default="https://api.agicto.cn", help="API基础URL")
    parser.add_argument("--api-key", help="OpenAI API密钥")
    
    args = parser.parse_args()
    
    # 获取API密钥（优先使用命令行参数，其次使用环境变量）
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未提供API密钥。请通过--api-key参数或OPENAI_API_KEY环境变量提供。")
        sys.exit(1)
    
    try:
        generator = HeaderImageGenerator(args.api_base, api_key)
        print(f"正在为《{args.title}》生成题图...")
        image_data = generator.generate_image(args.title)
        generator.save_image(image_data, args.output)
        print(f"题图已保存到: {args.output}")
    
    except Exception as e:
        print(f"错误：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
