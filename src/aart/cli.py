# see: https://packaging.python.org/en/latest/guides/creating-command-line-tools/
# which also mentions [typer] and [docopt] packages to make CLI's, but I'm using argparse myself.

import argparse
from . import aart
from .aart import _debug_print, alphabets, convert_to_ascii_with_args, preview_alphabets


def main():
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
        "--color-mode",
        choices=["monotone", "true-color"],
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
        aart.DEBUG
        aart._debug_print("Debug mode is enabled.")

    if args.alphabet == "__all__":
        preview_alphabets(args)
    else:
        convert_to_ascii_with_args(args)
