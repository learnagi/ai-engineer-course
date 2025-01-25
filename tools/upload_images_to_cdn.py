#!/usr/bin/env python3
import os
import re
import sys
import hashlib
from datetime import datetime
from qiniu import Auth, put_file, etag
import argparse
import glob
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(env_path)

class ImageUploader:
    def __init__(self, access_key=None, secret_key=None, bucket_name=None, domain=None):
        """初始化七牛云配置"""
        self.access_key = access_key or os.getenv('QINIU_ACCESS_KEY')
        self.secret_key = secret_key or os.getenv('QINIU_SECRET_KEY')
        self.bucket_name = bucket_name or os.getenv('QINIU_BUCKET')
        self.domain = domain or os.getenv('QINIU_DOMAIN')
        
        if not all([self.access_key, self.secret_key, self.bucket_name, self.domain]):
            print("错误: 缺少必要的配置信息。请确保以下环境变量已设置：")
            print("- QINIU_ACCESS_KEY")
            print("- QINIU_SECRET_KEY")
            print("- QINIU_BUCKET")
            print("- QINIU_DOMAIN")
            print("\n或者通过命令行参数提供这些值。")
            sys.exit(1)
            
        self.q = Auth(self.access_key, self.secret_key)
        
    def get_file_hash(self, filepath):
        """获取文件的MD5哈希值"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    
    def upload_file(self, local_file):
        """上传文件到七牛云"""
        # 生成上传凭证
        file_hash = self.get_file_hash(local_file)
        file_ext = os.path.splitext(local_file)[1]
        key = f"tutorial/images/{file_hash}{file_ext}"  # 使用hash作为文件名，避免重复
        token = self.q.upload_token(self.bucket_name, key, 3600)
        
        try:
            ret, info = put_file(token, key, local_file)
            if ret and ret['key'] == key:
                return f"https://{self.domain}/{key}"
            else:
                print(f"上传失败: {info}")
                return None
        except Exception as e:
            print(f"上传出错: {e}")
            return None

def process_markdown_file(file_path, uploader):
    """处理单个markdown文件"""
    print(f"\n处理文件: {file_path}")
    
    # 读取markdown内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找所有本地图片引用
    img_pattern = r'!\[([^\]]*)\]\(([^http][^)]+)\)'
    matches = re.finditer(img_pattern, content)
    
    # 记录需要替换的内容
    replacements = []
    base_dir = os.path.dirname(file_path)
    
    for match in matches:
        alt_text = match.group(1)
        local_path = match.group(2)
        
        # 处理相对路径
        if not os.path.isabs(local_path):
            local_path = os.path.join(base_dir, local_path)
        
        # 规范化路径
        local_path = os.path.normpath(local_path)
        
        if os.path.exists(local_path):
            print(f"上传图片: {local_path}")
            cdn_url = uploader.upload_file(local_path)
            if cdn_url:
                old_text = match.group(0)
                new_text = f"![{alt_text}]({cdn_url})"
                replacements.append((old_text, new_text))
                print(f"✓ 成功上传: {cdn_url}")
            else:
                print(f"✗ 上传失败: {local_path}")
        else:
            print(f"✗ 文件不存在: {local_path}")
    
    # 替换内容
    new_content = content
    for old_text, new_text in replacements:
        new_content = new_content.replace(old_text, new_text)
    
    # 如果有改动，写回文件
    if new_content != content:
        backup_path = f"{file_path}.bak"
        print(f"创建备份: {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"更新文件: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    
    return len(replacements)

def main():
    parser = argparse.ArgumentParser(description='上传Markdown文件中的图片到七牛云CDN')
    parser.add_argument('path', help='Markdown文件路径或目录路径')
    parser.add_argument('--ak', help='七牛云 Access Key')
    parser.add_argument('--sk', help='七牛云 Secret Key')
    parser.add_argument('--bucket', help='七牛云 Bucket 名称')
    parser.add_argument('--domain', help='七牛云域名')
    
    args = parser.parse_args()
    
    # 初始化上传器
    uploader = ImageUploader(args.ak, args.sk, args.bucket, args.domain)
    
    # 处理文件或目录
    if os.path.isfile(args.path):
        if args.path.endswith('.md'):
            count = process_markdown_file(args.path, uploader)
            print(f"\n完成! 共处理 {count} 张图片")
    elif os.path.isdir(args.path):
        total_count = 0
        for md_file in glob.glob(os.path.join(args.path, '**/*.md'), recursive=True):
            count = process_markdown_file(md_file, uploader)
            total_count += count
        print(f"\n完成! 共处理 {total_count} 张图片")
    else:
        print("错误: 指定的路径不存在")
        sys.exit(1)

if __name__ == '__main__':
    main()
