import importlib.metadata
from importlib.metadata import PackageNotFoundError
import sys

# 你提供的 requirements 内容
REQUIREMENTS_TEXT = """
opencv-python-headless==4.8.1.78
certifi==2022.12.7
contourpy==1.3.0
huggingface-hub==0.17.3
jinja2==3.1.3
matplotlib==3.8.2
pyparsing==3.2.3
transformers==4.34.1
triton==2.1.0
tqdm==4.65.2
zipp==3.22.0
"""


def check_requirements(req_text):
    print(f"{'包名 (Package)':<30} | {'状态 (Status)':<10} | {'详细信息 (Details)':<20}")
    print("-" * 70)

    lines = req_text.strip().split('\n')
    missing_packages = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # 解析包名和版本
        # 这里简单处理 '==' 分割，如果格式更复杂可以使用正则表达式
        if '==' in line:
            pkg_name, required_version = line.split('==', 1)
        else:
            pkg_name = line
            required_version = None

        pkg_name = pkg_name.strip()

        try:
            # 尝试获取已安装包的版本
            installed_version = importlib.metadata.version(pkg_name)

            # 检查版本是否匹配
            if required_version:
                if installed_version == required_version.strip():
                    status = "✅ 已安装"
                    details = f"版本一致: {installed_version}"
                    print(f"\033[92m{pkg_name:<30} | {status:<10} | {details}\033[0m")  # 绿色
                else:
                    status = "⚠️ 版本差异"
                    details = f"当前: {installed_version} (需: {required_version})"
                    print(f"\033[93m{pkg_name:<30} | {status:<10} | {details}\033[0m")  # 黄色
            else:
                status = "✅ 已安装"
                details = f"当前: {installed_version}"
                print(f"\033[92m{pkg_name:<30} | {status:<10} | {details}\033[0m")  # 绿色

        except PackageNotFoundError:
            status = "❌ 未安装"
            details = f"缺失"
            print(f"\033[91m{pkg_name:<30} | {status:<10} | {details}\033[0m")  # 红色
            missing_packages.append(line)

    print("-" * 70)

    # 总结输出
    if missing_packages:
        print(f"\n发现 {len(missing_packages)} 个缺失的包。")
        print("你可以使用以下命令安装缺失项：")
        print(f"\npip install {' '.join(missing_packages)}")
    else:
        print("\n完美！所有包都已安装。")


if __name__ == '__main__':
    check_requirements(REQUIREMENTS_TEXT)