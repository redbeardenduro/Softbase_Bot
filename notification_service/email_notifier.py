import smtplib
from email.mime.text import MIMEText

class EmailNotifier:
    def __init__(self, from_email, from_password, smtp_server, smtp_port):
        self.from_email = from_email
        self.from_password = from_password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def send_email(self, to_email, subject, body):
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.from_email
        msg['To'] = to_email

        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()  # Start TLS for security
            server.login(self.from_email, self.from_password)
            server.sendmail(self.from_email, to_email, msg.as_string())
