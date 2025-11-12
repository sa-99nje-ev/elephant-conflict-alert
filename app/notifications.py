import os
from twilio.rest import Client
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv

# --- Load Environment Variables ---
# This loads the keys from your .env file
load_dotenv()

# --- Message Templates (English) ---
HIGH_RISK_WARNING_SUBJECT = "Elephant Conflict Alert"
HIGH_RISK_WARNING_BODY = ("WARNING: A high risk of elephant conflict has been predicted in your area. "
                          "Please be cautious and avoid traveling at night.")

# --- Twilio Configuration ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# This is the number you verified in your Twilio account
YOUR_MOBILE_NUMBER = os.getenv("YOUR_MOBILE_NUMBER")


def send_sms(to_number: str, message_body: str):
    """
    Sends an SMS using Twilio.
    """
    
    # Check if all keys are present
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        print("--- Twilio credentials not set. SIMULATING SMS ---")
        print(f"To: {to_number}, Body: {message_body}")
        print("--------------------------------------------------")
        return

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )
        print(f"SMS sent successfully to {to_number}! SID: {message.sid}")
    except Exception as e:
        print(f"Error sending SMS via Twilio: {e}")
        print("NOTE: Did you add and verify your phone number in the Twilio trial account?")


def send_email(to_email: str, subject: str, body: str):
    """
    Sends an email using SendGrid.
    """
    # --- !! TODO: Get these from environment variables !! ---
    sendgrid_api_key = os.getenv('SENDGRID_API_KEY')
    from_email = "alerts@yourdomain.com" # Must be a verified sender in SendGrid
    
    if not sendgrid_api_key:
        print(f"--- SendGrid API key not set. SIMULATING Email ---")
        print(f"To: {to_email}, Subject: {subject}")
        print("--------------------------------------------------")
        return

    # --- UNCOMMENT THIS BLOCK TO SEND REAL EMAIL ---
    # message = Mail(
    #     from_email=from_email,
    #     to_emails=to_email,
    #     subject=subject,
    #     html_content=f"<strong>{body}</strong>"
    # )
    # try:
    #     sg = SendGridAPIClient(sendgrid_api_key)
    #     response = sg.send(message)
    #     print(f"Email sent successfully, Status Code: {response.status_code}")
    # except Exception as e:
    #     print(f"Error sending email: {e}")


def dispatch_alerts(location: str):
    """
    Main function to trigger alerts for a high-risk location.
    """
    print(f"*** Dispatching REAL alerts for high-risk location: {location} ***")
    
    # --- In a real app, you would fetch users from a DB ---
    # --- For testing, we'll send to YOUR mobile number ---
    
    if YOUR_MOBILE_NUMBER:
        send_sms(YOUR_MOBILE_NUMBER, f"{HIGH_RISK_WARNING_BODY} (Location: {location})")
    else:
        print("--- YOUR_MOBILE_NUMBER is not set in .env. Skipping SMS. ---")

    # --- For testing, we'll send to a dummy email ---
    dummy_email = "test@example.com"
    send_email(dummy_email, HIGH_RISK_WARNING_SUBJECT, f"{HIGH_RISK_WARNING_BODY} (Location: {location})")