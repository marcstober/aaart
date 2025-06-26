# see: https://packaging.python.org/en/latest/guides/creating-command-line-tools/
# which also mentions [typer] and [docopt] packages to make CLI's, but I'm using argparse myself

import argparse
import shutil

import pyfiglet

from . import image
from .image import (
    _debug_print,
    alphabets,
    convert_to_ascii_with_args,
    preview_alphabets,
)


def main():
    # TODO: Also use argparse for this?
    parser = argparse.ArgumentParser(description="Another ASCII Art program (aaart)")
    parser.add_argument("command", choices=["text", "image"], help="Command to run")
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments for the command"
    )
    args = parser.parse_args()
    if args.command == "text":
        text_main(args.args)
    elif args.command == "image":
        image_main(args.args)


def image_main(raw_args):
    # TODO: move this back into image.py?
    # TODO: consider using argparse's subparsers: https://docs.python.org/3/library/argparse.html#other-utilities
    #   (But actually, I like it this way, because I could move subcommands with their argument parses to their own files.)
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
    # the idea is to make this work like pyfiglet's --color argument
    parser.add_argument(
        "--color",
        "-c",
        default=":",
        help="color to use in monotone mode",
    )
    parser.add_argument(
        "--width",
        "-w",
        type=int,
        help="Width of the output ASCII art (default: terminal width)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    args = parser.parse_args(raw_args)

    if args.debug:
        image.DEBUG = True
        _debug_print("Debug mode is enabled.")

    if args.alphabet == "__all__":
        preview_alphabets(args)
    else:
        convert_to_ascii_with_args(args)


def text_main(raw_args):
    # the arguments are *mostly* a subset of pyfiglet's
    parser = argparse.ArgumentParser(description="Convert text to ASCII art.")
    parser.add_argument("text", help="Text to convert to ASCII art", nargs="+")
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
    parser.add_argument(
        "--color",
        "-c",
        default=":",  # default needed for pyfiglet
        help="prints text with color",
    )
    args = parser.parse_args(raw_args)

    def render_text(args):
        width = args.width
        if not width:
            width = shutil.get_terminal_size().columns
        pyfiglet.print_figlet(
            " ".join(args.text), font=args.font, colors=args.color, width=width
        )

    if args.list_fonts:
        # TODO: just call pyfiglet's function?
        fonts = pyfiglet.Figlet().getFonts()
        print("Installed fonts:")
        for font in fonts:
            print(f"  {font}")
    if args.font == "__all__":
        # like the preview_alphabets function for images
        # TODO: this might be too many for the kids (how many are there?)
        for font in pyfiglet.Figlet().getFonts():
            print("\033[2J\033[H", end="")  # clear the screen
            args.font = font
            render_text(args)
            inp = input(
                f"Showing '{font}' font. Press 'q' then ENTER to quit, or ENTER to continue..."
            )
            if inp == "q":
                break
    else:
        render_text(args)
