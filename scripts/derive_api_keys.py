"""One-shot script to derive Polymarket CLOB API credentials from wallet private key.

Usage:
    PYTHONPATH=. python scripts/derive_api_keys.py

Outputs the 3 values to add to your .env file:
    POLYMARKET_API_KEY, POLYMARKET_API_SECRET, POLYMARKET_API_PASSPHRASE
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))

if not private_key:
    print("ERROR: POLYMARKET_PRIVATE_KEY not set in .env")
    sys.exit(1)

from py_clob_client.client import ClobClient

client = ClobClient(
    host="https://clob.polymarket.com",
    key=private_key,
    chain_id=chain_id,
)

print("Deriving API credentials from wallet...\n")
creds = client.create_or_derive_api_creds()

print(f"POLYMARKET_API_KEY={creds.api_key}")
print(f"POLYMARKET_API_SECRET={creds.api_secret}")
print(f"POLYMARKET_API_PASSPHRASE={creds.api_passphrase}")
print("\n→ Copy these 3 lines into your .env file")
