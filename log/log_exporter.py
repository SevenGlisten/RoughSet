"""
日志导出模块
功能：将文本内容保存为UTF-8编码的TXT文件
"""

import os
from datetime import datetime


class LogExporter:
    @staticmethod
    def export_to_txt(content: str, folder: str = "reports") -> str:
        """
        导出文本内容到TXT文件
        :param content: 要导出的文本内容
        :param folder: 保存目录（默认reports）
        :return: 生成的文件绝对路径
        """
        try:
            # 创建目录
            os.makedirs(folder, exist_ok=True)

            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_report_{timestamp}.txt"
            filepath = os.path.abspath(os.path.join(folder, filename))

            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            return filepath
        except Exception as e:
            raise RuntimeError(f"导出失败: {str(e)}")