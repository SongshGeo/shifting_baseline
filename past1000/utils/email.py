#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "Warning: python-dotenv not installed, using system environment variables only"
    )


def send_notification_email(
    success: bool,
    error_msg: str = "",
    start_time: Optional[datetime] = None,
    sender_email: Optional[str] = None,
    sender_password: Optional[str] = None,
):
    """Send email notification after model run completion.

    Args:
        success: Whether the run was successful
        error_msg: Error message if failed
        start_time: When the run started
    """
    # 从环境变量获取邮箱配置
    sender_email_final = sender_email or os.getenv("EMAIL_SENDER")
    sender_password_final = sender_password or os.getenv("EMAIL_PASSWORD")
    receiver_email = os.getenv("EMAIL_RECEIVER", "SongshGeo@gmail.com")
    smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))

    # 检查必要的配置是否存在
    if not all([sender_email_final, sender_password_final, receiver_email]):
        missing = []
        if not sender_email_final:
            missing.append("EMAIL_SENDER")
        if not sender_password_final:
            missing.append("EMAIL_PASSWORD")
        if not receiver_email:
            missing.append("EMAIL_RECEIVER")
        print(f"跳过邮件通知：缺少环境变量 {', '.join(missing)}")
        return

    try:
        # Create message
        msg = MIMEMultipart()
        msg["From"] = sender_email_final
        msg["To"] = receiver_email

        if success:
            subject = "✅ ABM 模型运行成功完成"
            duration = (
                (datetime.now() - start_time).total_seconds() if start_time else 0
            )
            body = f"""
ABM 模型运行已成功完成！

运行详情：
- 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else '未记录'}
- 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 运行时长: {duration:.1f} 秒 ({duration / 60:.1f} 分钟)

结果文件已保存，可以开始分析数据。
            """
        else:
            subject = "❌ ABM 模型运行失败"
            body = f"""
ABM 模型运行过程中发生错误！

错误详情：
- 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else '未记录'}
- 失败时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 错误信息:
{error_msg}

请检查日志文件并修复问题。
            """
        assert sender_email_final is not None, "sender_email_final is None"
        assert smtp_port is not None, "smtp_port is None"
        assert smtp_server is not None, "smtp_server is None"
        assert sender_password_final is not None, "sender_password_final is None"
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email_final, sender_password_final)
        text = msg.as_string()
        server.sendmail(sender_email_final, receiver_email, text)
        server.quit()

        print(f"邮件通知已发送: {subject}")

    except Exception as e:  # pylint: disable=broad-except # 邮件发送可能遇到多种异常类型
        print(f"发送邮件通知失败: {e}")
