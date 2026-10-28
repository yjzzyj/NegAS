"""Generate candidate text offline from a user-supplied prompt."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="UTF-8 prompt file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        parser.error("Set OPENAI_API_KEY in the environment")
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    result = client.chat.completions.create(model=args.model, messages=[
        {"role": "user", "content": Path(args.prompt).read_text(encoding="utf-8")}
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.choices[0].message.content or "", encoding="utf-8")


if __name__ == "__main__":
    main()
