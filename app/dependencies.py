# app/dependencies.py
import os
from fastapi import Security, HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

# Load your app's secret key from the .env file
load_dotenv()
APP_API_KEY = os.getenv("APP_API_KEY")

# This tells FastAPI to look for a header named "X-API-Key"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency function that checks for the API key in the request header.
    """
    if not APP_API_KEY:
        # This is a server-side configuration error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key not configured on server"
        )

    if not api_key or api_key != APP_API_KEY:
        # This is a client-side authentication error
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    
    # If the key is valid, the request can proceed
    return api_key

# We create a single "dependency" that we can reuse
PROTECTED = Depends(get_api_key)