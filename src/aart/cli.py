# see: https://packaging.python.org/en/latest/guides/creating-command-line-tools/
# which also mentions [typer] and [docopt] packages to make CLI's, but I'm using argparse myself

import argparse
import shutil
import sys
from turtle import width

from pyfiglet import Figlet

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
        text_main()
    elif command == "image":
        image_main()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


def image_main():
    # TODO: move this back into image.py?
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


def text_main():
    parser = argparse.ArgumentParser(description="Convert text to ASCII art.")
    parser.add_argument(
        "text", help="Text to convert to ASCII art", nargs="+", default="hello"
    )
    parser.add_argument(
        "--font",
        "-f",
        default="slant",
        help="Font to use for ASCII art (default: slant), or '__all__' to preview all fonts",
    )
    # unlike pyfiglet, this defaults to the terminal width
    parser.add_argument(
        "--width",
        "-w",
        type=int,
        help="Width of the output ASCII art",
    )
    parser.add_argument(
        "--list-fonts",  # style of the rest of this project
        "--list_fonts",  # compatibility with pyfiglet
        "-l",
        action="store_true",
        help="show installed fonts list",
    )
    # TODO: color
    args = parser.parse_args()

    if args.list_fonts:
        # TODO: just call pyfiglet's function?
        fonts = Figlet().getFonts()
        print("Installed fonts:")
        for font in fonts:
            print(f"  {font}")
    if args.font == "__all__":
        # like the preview_alphabets function for images
        # TODO: this might be too many for the kids (how many are there?)
        for font in Figlet().getFonts():
            print("\033[2J\033[H", end="")  # clear the screen
            width = args.width
            if not width:
                width = shutil.get_terminal_size().columns
            f = Figlet(font=font, width=width)
            print(f.renderText(" ".join(args.text)))
            inp = input(
                f"Showing '{font}' font. Press 'q' then ENTER to quit, or ENTER to continue..."
            )
            if inp == "q":
                break
    else:
        width = args.width
        if not width:
            width = shutil.get_terminal_size().columns
        f = Figlet(font=args.font, width=width)
        print(f.renderText(" ".join(args.text)))
