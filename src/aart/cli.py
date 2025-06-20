# see: https://packaging.python.org/en/latest/guides/creating-command-line-tools/
# which also mentions [typer] and [docopt] packages to make CLI's, but I'm using argparse myself

import argparse
import sys

from .image import (
    DEBUG,
    _debug_print,
    alphabets,
    convert_to_ascii_with_args,
    preview_alphabets,
)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: aart <command> [options]\nCommands:"
            "\n  text   Convert text to ASCII art"
            "\n  image  (coming soon)"
        )
        sys.exit(1)
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if command == "text":
        print("'aart text' is not implemented yet.")
        sys.exit(1)
    elif command == "image":
        image_main()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


def image_main():
    parser = argparse.ArgumentParser(description="Convert image to ASCII art.")
    parser.add_argument("image_path", help="Path to the input image file")
    # parser.add_argument(
    #     "-o", "--output", default="ascii_image.txt", help="Output ASCII text file"
    # )
    parser.add_argument(
        "--alphabet",
        "-a",
        default="variant2",
        choices=alphabets.keys(),
        help="Choose the ASCII alphabet to use (default: variant2)",
    )
    parser.add_argument(
        "--mode",
        choices=["monotone", "color"],
        default="monotone",
        help="Choose the color mode (default: monotone)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    args = parser.parse_args()

    if args.debug:
        DEBUG = True
        _debug_print("Debug mode is enabled.")

    if args.alphabet == "__all__":
        preview_alphabets(args)
    else:
        convert_to_ascii_with_args(args)
