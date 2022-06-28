# -*- coding: utf-8 -*-
import os
import traceback
import zipfile
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import datetime
from utils.utils import logger


class _Email(object):
    def __init__(self, from_address, password, smtp_server, to_address):
        self.from_address = from_address
        self.password = password
        self.to_address = to_address
        self.smtp_server = smtp_server
        self.logger = logger()

    def zip_folder(self, folder_path, stock_name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        stock_name = "港股" if stock_name is 'gang' else 'a股'
        zip_file_path = "{}-{}背离预测.zip".format(current_date, stock_name)

        with zipfile.ZipFile(zip_file_path, 'w', ) as target:
            for i in os.walk(folder_path):
                for n in i[2]:
                    target.write(''.join((i[0], '\\', n)))

        return zip_file_path

    # 中文处理
    def _format_addr(self, s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    def send_email(self, stock_type, file_paths):
        stock_chinese_name = "港股" if stock_type == "gang" else "A股"

        # 邮件发送和接收人配置
       
        msg = MIMEMultipart()
        msg['From'] = self._format_addr('四叶草 <%s>' % self.from_address)  # 显示的发件人
        # msg['To'] = _format_addr('管理员 <%s>' % to_addr)                # 单个显示的收件人
        msg['To'] = ",".join(self.to_address)  # 多个显示的收件人
        header_str = '今日{}预测'.format(stock_chinese_name)
        msg['Subject'] = Header(header_str, 'utf-8').encode()  # 显示的邮件标题


        # 邮件正文是MIMEText:
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        body = "{}-{}背离预测".format(current_date, stock_chinese_name)
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        for path in file_paths:
            # 添加附件就是加上一个MIMEBase，从本地读取一个文件
            with open(path, 'rb') as f:
                # 设置附件的MIME和文件名，这里是txt类型:
                mime = MIMEBase('file', 'txt', filename=path)
                # 加上必要的头信息:
                mime.add_header('Content-Disposition', 'attachment', filename=path)
                mime.add_header('Content-ID', '<0>')
                mime.add_header('X-Attachment-Id', '0')
                # 把附件的内容读进来:
                mime.set_payload(f.read())
                # 用Base64编码:
                encoders.encode_base64(mime)
                # 添加到MIMEMultipart:
                msg.attach(mime)
        try:
            server = smtplib.SMTP(self.smtp_server, 25)
            # server.starttls()
            # server.set_debuglevel(1)  # 用于显示邮件发送的执行步骤
            server.login(self.from_address, self.password)
            server.sendmail(self.from_address, self.to_address, msg.as_string())
            server.quit()
        except RuntimeError:
            message = traceback.format_exc()
            self.logger.error(message)


if __name__ == '__main__':
    pass
