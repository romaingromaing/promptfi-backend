# import os
# import json
# from web3 import Web3
# from dotenv import load_dotenv

# load_dotenv()

# FUJI_RPC_URL = os.getenv("FUJI_RPC_URL")

# PRICE_FEED_ADDRESS = "0x5498BB86BC934c8D34FDA08E81D444153d0D06aD"

# ABI = json.loads('''
# [{"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"description","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint80","name":"_roundId","type":"uint80"}],"name":"getRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"latestRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"version","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]
# ''')

# # --- Main Logic ---
# def main():
#     if not FUJI_RPC_URL:
#         print("Error: FUJI_RPC_URL not set. Please add it to your .env file.")
#         return

#     # Connect to the Avalanche Fuji C-Chain
#     w3 = Web3(Web3.HTTPProvider(FUJI_RPC_URL))
#     if not w3.is_connected():
#         print("Error: Not connected to the Avalanche Fuji network. Check your RPC URL.")
#         return

#     print("Connected to Avalanche Fuji network.")

#     # Create a contract instance
#     contract = w3.eth.contract(address=Web3.to_checksum_address(PRICE_FEED_ADDRESS), abi=ABI)

#     try:
#         # Fetch the decimals for correct formatting
#         decimals = contract.functions.decimals().call()

#         # Fetch the latest price data
#         latest_data = contract.functions.latestRoundData().call()
#         (round_id, answer, started_at, updated_at, answered_in_round) = latest_data

#         # Format and print the results
#         price = answer / (10**decimals)
#         print("--- Chainlink Price Feed Data ---")
#         print(f"Token: AVAX/USD")
#         print(f"Latest Price: ${price}")
#         print(f"Round ID: {round_id}")
#         print(f"Updated At: {updated_at}")
#         print(f"Answered in Round: {answered_in_round}")

#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()

