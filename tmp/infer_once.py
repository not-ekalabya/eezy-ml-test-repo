"""Send a single inference request to the running server.

Usage:
    python scripts/infer_once.py
    python scripts/infer_once.py --prompt "Summarize model deployment in one sentence."
    python scripts/infer_once.py --server-url http://localhost:5000 --timeout 30
"""

import argparse
import json

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one inference call against /predict")
    parser.add_argument(
        "--server-url",
        default="http://100.24.117.137:5000",
        help="Base URL of the inference server",
    )
    parser.add_argument(
        "--prompt",
        default=r"""	
Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations:
\[\log_2\left({x \over yz}\right) = {1 \over 2}\]
\[\log_2\left({y \over xz}\right) = {1 \over 3}\]
\[\log_2\left({z \over xy}\right) = {1 \over 4}\]
Then the value of $\left|\log_2(x^4y^3z^2)\right|$ is $\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.""",
        help="Prompt text sent as features",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=6000.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional max_new_tokens override for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature override for generation",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional top_p override for generation",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable thinking mode for models that support it",
    )
    parser.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking mode for models that support it",
    )
    parser.set_defaults(enable_thinking=None)
    args = parser.parse_args()

    url = args.server_url.rstrip("/") + "/predict"
    payload = {"features": args.prompt}
    if args.max_new_tokens is not None:
        payload["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.enable_thinking is not None:
        payload["enable_thinking"] = args.enable_thinking

    response = requests.post(url, json=payload, timeout=args.timeout)
    response.raise_for_status()

    body = response.json()
    print(json.dumps(body, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
