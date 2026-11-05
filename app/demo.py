"""CLI demo for CitizenSense.

Interactive:   python demo.py          (in Colab use: %run demo.py)
Single input:  python demo.py "There is no water supply in our area for three days."
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import analyze_feedback, format_card  # noqa: E402


def main():
    if len(sys.argv) > 1:
        print(format_card(analyze_feedback(" ".join(sys.argv[1:]))))
        return
    print("CitizenSense CLI demo (type 'quit' to exit)")
    while True:
        text = input("\nEnter feedback text: ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            print("Please type some feedback.")
            continue
        print(format_card(analyze_feedback(text)))


if __name__ == "__main__":
    main()
