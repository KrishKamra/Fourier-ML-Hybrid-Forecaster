import numpy as np
import json

def generate_payload(period=24):
    # Generate 48 hours of a specific cycle
    time = np.arange(48)
    signal = 10 * np.sin(2 * np.pi * time / period) + 20 + np.random.normal(0, 1, 48)
    
    payload = {"data_window": signal.tolist()}
    print(json.dumps(payload))

if __name__ == "__main__":
    # Generate a standard 24h cycle payload
    print("--- 24h Cycle Payload ---")
    generate_payload(12)