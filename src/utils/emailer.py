"""
SMTP email helpers for account verification.
"""

import os
import smtplib
from email.message import EmailMessage


class EmailConfigurationError(RuntimeError):
    """Raised when SMTP settings are incomplete."""


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise EmailConfigurationError(f"Missing required email setting: {name}")
    return value


def validate_email_settings() -> None:
    """Fail fast when required SMTP settings are missing."""
    _required_env("SMTP_HOST")
    _required_env("SMTP_USERNAME")
    _required_env("SMTP_PASSWORD")


def send_verification_email(to_email: str, code: str) -> None:
    """Send the account verification code using configured SMTP settings."""
    validate_email_settings()
    host = _required_env("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = _required_env("SMTP_USERNAME")
    password = _required_env("SMTP_PASSWORD")
    from_email = os.getenv("SMTP_FROM_EMAIL", username).strip()
    from_name = os.getenv("SMTP_FROM_NAME", "Florida Policy Navigator").strip()
    use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() != "false"

    message = EmailMessage()
    message["Subject"] = "Your Florida Policy Navigator verification code"
    message["From"] = f"{from_name} <{from_email}>"
    message["To"] = to_email
    message.set_content(
        "\n".join(
            [
                "Welcome to Florida Policy Navigator.",
                "",
                f"Your verification code is: {code}",
                "",
                "This code verifies access to your account.",
            ]
        )
    )

    with smtplib.SMTP(host, port, timeout=20) as smtp:
        if use_tls:
            smtp.starttls()
        smtp.login(username, password)
        smtp.send_message(message)
