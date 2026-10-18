"""Send realistic random traffic (plus a few bad requests) to the API."""
import argparse
import random
import time

import requests


def random_patient():
    return {
        "age": random.randint(30, 75), "sex": random.randint(0, 1),
        "cp": random.randint(1, 4), "trestbps": random.randint(100, 180),
        "chol": random.randint(150, 350), "fbs": random.randint(0, 1),
        "restecg": random.randint(0, 2), "thalach": random.randint(90, 200),
        "exang": random.randint(0, 1), "oldpeak": round(random.random() * 4, 1),
        "slope": random.randint(1, 3), "ca": random.randint(0, 3),
        "thal": random.choice([3, 6, 7]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--delay", type=float, default=0.2)
    args = ap.parse_args()

    ok = bad = 0
    for _ in range(args.n):
        body = random_patient()
        if random.random() < 0.1:          # ~10% invalid -> 422
            body["age"] = 500
        r = requests.post(f"{args.url}/predict", json=body, timeout=5)
        ok, bad = (ok + 1, bad) if r.status_code == 200 else (ok, bad + 1)
        time.sleep(args.delay)
    print(f"done: {ok} OK, {bad} rejected")


if __name__ == "__main__":
    main()