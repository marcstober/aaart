import argparse
import numbers
import shutil
import sys
from collections import Counter
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# TODO: package (open-source/pip installable?)
# TODO: figment text
# TODO: mode not color-mode
# TODO: how does the JS version get the grayscale value of a colored pixel? (I think I did this)
# TODO: colored monotone and other duotone modes
# TODO: cf. other Python options?
# TODO? "art" (not image or text?) - but some is inappropriate for kids
# TODO: other options like posterize, stipple, lineart from Khrome's ascii-art
# TODO: invert option
# TODO: more color modes (8 bit, 4 bit, "standard" terminal colors, etc.)
# TODO: Despite a lot of work, some details are different than the JS version.
#   See downsampled-with-node-js.png. In particular, the first line of text.
#   Maybe we just got lucky with how it aligns to the pixel grid or something?
#   But we're close enough.
# TODO: A conceptual issue is that in true color mode, we both change the letter
#   when the color is less bright, and the color itself, so the range of brightness is more than it should be.
#   Maybe this is just a quirk of making ASCII art, the point is to be "lo-fi."
# TODO: Faster performance with RGB-mode images. (Maybe scale down first,
#   to something like 4x size?)

DEBUG = False


def _debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


# Predefined value scales (ASCII alphabets)
# adapted from [ascii-art] https://github.com/khrome/ascii-art),
# originally licensed under the MIT License:
# Copyright (c) Abbey Hawk Sparrow.
#
# See ascii-art-LICENSE for full license text.
alphabets: dict[str, str] = {
    "grayscale-chars": " .:-=+*#%@",
    # this one only makes sense with color
    "solid": "█",
    "standard": "'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "variant1": " .,:;i1tfLCG08@",
    "variant2": " .:-=+*#%@",
    "variant3": "    .,-=++°oo0ø$$ØØ®®¥¥#",
    "variant4": " .,:;=+itIYVXRBMW#",
    "ultra-wide": "    .........,,,,,,:,:::::::iiiiiiiii;;;;;;;;rrrrrrr7777777XXXXXXXXXXXSSSSSSS2222222aaaaaaZaZZZZZZZZZ888888800000000BBBBBBBBWWWWWWWWW@@@@@@@MMMMMMM",
    "wide": "        ........,,,,,,,:::::::;;;;;;;;rrrrrrrssssiiiiSSS552222XXX3399hhGG&&AAAAHHHBBMMM######@@@@@@@",
    "hatching": "    ...,,;;---===+++xxxXX##",
    "bits": " #",
    "binary": " 10",
    "binary-log": " 11111000",  # higher contrast, works better with logo.gif
    "grayscale": " ░░░░▒▒▒▒▓▓▓▓█",
    "blocks": " ▖▚▜█",
    # special
    "__all__": "",
}


def enhance_gamma_value(value, max_value=255, gamma=0.2):
    # TODO? Remove this? It could be useful, but we don't actually use it currently.
    value = max(0, min(255, value))
    if value == 255:
        return max_value
    scaled = (value / 255) ** gamma * max_value
    return int(round(scaled))


def map_rgb_value(value: Iterable[numbers.Number], alphabet_name="grayscale-chars"):
    v = _map_value(value, alphabet_name=alphabet_name, color_mode="true-color")
    return v


def map_value(value, alphabet_name="grayscale-chars"):
    return _map_value(value, alphabet_name=alphabet_name, color_mode="monotone")


def _map_value(value, alphabet_name, color_mode):
    """
    Map a pixel value (0-255) to a character in the grayscale_chars string.
    """
    grayscale_value: int = 0
    alphabet: str = alphabets[alphabet_name]
    scale = 256 / len(alphabet)

    if color_mode == "monotone":
        grayscale_value = int(value)
        # ensure value is integer (calling code may pass numpy.uint8)
        grayscale_value = int(value)
        return alphabet[round(min(grayscale_value / scale, len(alphabet) - 1))]
    elif color_mode == "true-color":
        int_rgb_values = tuple(map(int, value))
        grayscale_value = int(
            (int_rgb_values[0] + int_rgb_values[1] + int_rgb_values[2]) / 3
        )
        char = alphabet[round(min(grayscale_value / scale, len(alphabet) - 1))]
        return f"\033[38;2;{int_rgb_values[0]};{int_rgb_values[1]};{int_rgb_values[2]}m{char}"
    else:
        raise ValueError(f"Unsupported color mode: {color_mode}")


def calculate_new_size(width, height, max_width, max_height, character_aspect_ratio):
    """
    Calculate new image size to fit within terminal dimensions, preserving aspect ratio.
    Returns (new_width, new_height).
    """
    # max_width = 30  # for debugging, set a fixed max width
    new_height = height * max_width // width * character_aspect_ratio

    if new_height > max_height:
        # if it's too tall to fit, shrink width
        new_width = width * max_height // height // character_aspect_ratio
        new_height = height * new_width // width * character_aspect_ratio
    else:
        new_width = max_width

    return map(int, (new_width, new_height))


def has_transparency(img):
    return img.mode == "P" and img.info.get("transparency") is not None


def set_gif_transparency_to_black(img):
    _debug_print("Image mode is ", img.mode)

    transparency_index = img.info.get("transparency")

    if img.mode == "P" and transparency_index is not None:
        _debug_print(
            f"GIF has transparency. Transparent color index: {transparency_index}"
        )

        # make it not transparent but keep in palette mode
        # by removing from img.info
        _debug_print("Removing transparency from image info.")
        img.info.pop("transparency", None)

        _debug_print("Image has transpaceny:", has_transparency(img))
        # If you want the actual RGB color of that index:
        palette = img.getpalette()
        r = palette[transparency_index * 3]
        g = palette[transparency_index * 3 + 1]
        b = palette[transparency_index * 3 + 2]
        _debug_print(f"Transparent color (RGB): ({r}, {g}, {b})")

        # Change the transparent color to black in the palette
        palette = img.getpalette()
        palette[transparency_index * 3] = 0  # R
        palette[transparency_index * 3 + 1] = 0  # G
        palette[transparency_index * 3 + 2] = 0  # B
        img.putpalette(palette)
        _debug_print(f"Transparent color at index {transparency_index} set to black.")
    else:
        _debug_print("GIF does not have transparency.")


def get_color_counts(pixels):
    color_counts = Counter(pixels)
    return color_counts


def get_all_pixels(img: Image.Image, x0: int, y0: int, x1: int, y1: int):
    cropped_image = img.crop((x0, y0, x1, y1))
    # Fast path for numpy-compatible modes
    if img.mode in ("L", "P"):
        arr = np.array(cropped_image)
        return arr.flatten().tolist()
    elif img.mode.startswith("RGB"):
        arr = np.array(cropped_image)
        return [tuple(pixel) for pixel in arr.reshape(-1, arr.shape[-1])]
    # Fallback for other modes
    pixels = []
    for x in range(0, cropped_image.width):
        for y in range(0, cropped_image.height):
            pixel = cropped_image.getpixel((x, y))
            pixels.append(pixel)
    return pixels


# ensure the profile decorator is defined, even if not profiling
try:
    profile  # type: ignore
except NameError:

    def profile(func):
        """
        Dummy profile decorator that does nothing.
        """
        return func


class NoPixelError(ValueError):
    """
    Custom exception for when a pixel is not found in the image.
    """

    def __init__(self, img, x, y):
        message = f"Image of size {img.size} does not have pixel data for coordinate ({x}, {y})."
        super().__init__(message)


def _safe_getpixel(img: Image.Image, xy):
    """
    Safely get a pixel value from an image, handling cases where the coordinates might be out of bounds.
    """
    pixel = img.getpixel(xy)
    x, y = xy
    if pixel is not None and 0 <= x < img.width and 0 <= y < img.height:
        return pixel

    raise NoPixelError(img, x, y)


class _WrappedPalette:
    # TODO? Refactor class so set/get/len work on RGB tuples, not single values?

    @staticmethod
    def from_image(img):
        """
        Safely create a WrappedPalette from a PIL image in palette mode.
        """
        if img.mode == "P":
            palette = img.getpalette()
            if not palette:
                raise ValueError("Palette mode image does not have a palette.")
            return _WrappedPalette(palette)
        else:
            raise ValueError(f"Image mode '{img.mode}' does not support palettes.")

    def __init__(self, palette):
        self._palette = palette

    def __getitem__(self, index):
        return self._palette[index]

    def __setitem__(self, index, value):
        self._palette[index] = value

    def __len__(self):
        return len(self._palette)

    def get_color(self, index):
        """
        Get the RGB color for a given palette index.
        """
        r = self._palette[index * 3]
        g = self._palette[index * 3 + 1]
        b = self._palette[index * 3 + 2]
        return (r, g, b)


@profile  # type: ignore
def convert_to_ascii(
    image_path,
    color_mode="monotone",
    alphabet_name="variant2",
    # output_file="ascii_image.txt",
    width=None,  # target width in characters
    height=None,  # target height in characters
):
    img = Image.open(image_path)

    # Make it do what we want for showing in the terminal.
    # For example, logo on a white background (which is transparent)
    # should get a black background.
    # If the image is a GIF with a palette, set transparency to black
    set_gif_transparency_to_black(img)

    _debug_print("Image has transparency:", has_transparency(img))

    if color_mode == "monotone":
        img = img.convert("L")  # convert to grayscale
    # elif img.mode != "RGB":
    #     #     # convert anything else (palette-based GIF, RGBA, etc.) to RGB
    #     #     _debug_print(f"Converting {img.mode} image to RGB mode.")
    #     img = img.convert("RGB")

    original_width, original_height = img.size

    # get the size in columns of the current terminal
    # and resize the image accordingly
    terminal_size = shutil.get_terminal_size()
    max_width = terminal_size.columns
    max_height = terminal_size.lines - 2  # -2 for prompt

    # calculation in Get-CharacterSize.ps1
    character_aspect_ratio = 0.47

    calculated_new_width = 0
    calculated_new_height = 0
    if not width or not height:
        calculated_new_width, calculated_new_height = calculate_new_size(
            original_width,
            original_height,
            max_width,
            max_height,
            character_aspect_ratio,
        )
    new_width = width if width else calculated_new_width
    new_height = height if height else calculated_new_height

    new_width = width if width else calculated_new_width
    new_height = height if height else calculated_new_height
    _debug_print("new_width is", new_width)
    _debug_print("new_height is", new_height)

    # TODO! This is super hacky make it better! Especially don't save files!
    from . import downsampler

    img.save("temp.png")
    downsampler.advanced_downsample(
        "temp.png", "temp_downsampled.png", new_width, new_height
    )
    # it's not waht I want do this
    new_image_bilinear = Image.open("temp_downsampled.png").convert(
        "P", palette=Image.Palette.ADAPTIVE, colors=256
    )

    new_image_bilinear = new_image_bilinear.convert("L")

    if DEBUG:
        new_image_bilinear.save("debug_bilinear.png")

    # Get the RGBA value of the first pixel (top-left corner)
    first_pixel_rgba = img.getpixel((0, 0))
    _debug_print("First pixel RGBA value:", first_pixel_rgba)

    box_size_x = original_width / new_width
    box_size_y = original_height / new_height
    _debug_print("Box size (x, y):", box_size_x, box_size_y)

    interesting_labeled = {
        # (0, 0): "top-left corner (SHOULD be black)",
        (90, 10): "why is it magenta",
    }
    interesting = interesting_labeled.keys()

    if img.mode == "P":
        # img = img.convert("RGBA") # actually, is this a lot slower?

        _debug_print("Image is in palette mode (P).")
        # palette = _WrappedPalette.from_image(img)
        # palette[189] = 255
        # palette[190] = 0
        # palette[191] = 255
        # img.putpalette(palette._palette)

    x_grid_size = original_width / new_width
    y_grid_size = original_height / new_height

    # offsets to the center of each grid cell
    # NOTE: I think this might be exactly the same as PIL's NEAREST resampling.
    x_offset = x_grid_size / 2
    y_offset = y_grid_size / 2

    # TODO: don't really need to rezize here, just create a new image with the same palette
    # new_image = Image.new(img.mode, (new_width, new_height))
    new_image = img.resize((new_width, new_height), Image.Resampling.BOX)
    new_image_secondmost = img.resize((new_width, new_height), Image.Resampling.BOX)
    # new_image = new_image.convert("RGB")

    GREEN = None

    # TODO: don't do this if we don't need this and aren't debugging
    # TODO: also should probably be a separate function
    if new_image.mode == "P":
        # Add green (0,255,0) and magenta (255,255,0) to the palettes (for debugging)
        for img_palette_target in (new_image, new_image_secondmost):
            palette = img_palette_target.getpalette()
            if not palette:
                raise ValueError("Palette mode image does not have a palette.")

            # add two slots to the palette
            # palette.extend([0, 255, 0, 255, 0, 255])

            GREEN = len(palette) // 3 - 2

            _debug_print("Green index in palette:", GREEN)

    # cache original palette
    original_palette = None
    if img.mode == "P":
        original_palette = _WrappedPalette.from_image(img)

    # for debugging, make a copy to draw on and load the font
    if DEBUG:
        debug_image = img.copy()
        draw = ImageDraw.Draw(debug_image)
        font = ImageFont.load_default()

    # for each pixel in the new image,
    # find the corresponding pixel in the old image
    for x in range(new_width):
        x00 = round(x * x_grid_size)
        if x00 >= img.width:
            # ensure we don't go out of bounds.
            # this is here to satisfy unit tests.
            # I'm not sure it should happen in real life - I think it means you are
            # trying this with an image that is smaller than the target size
            # which probably doesn't make sense and shouldn't be supported.
            x00 -= 1
        x01 = round((x + 1) * x_grid_size)
        if x01 == x00:
            x01 += 1  # ensure we have at least one pixel in the box
        x0 = round(x * x_grid_size + x_offset)
        if x0 >= img.width:
            # ensure we don't go out of bounds.
            # this is here to satisfy unit tests.
            # I'm not sure it should happen in real life - I think it means you are
            # trying this with an image that is smaller than the target size
            # which probably doesn't make sense and shouldn't be supported.
            x0 -= 1
        for y in range(new_height):
            y00 = round(y * y_grid_size)
            if y00 >= img.height:
                # ensure we don't go out of bounds.
                # this is here to satisfy unit tests.
                # I'm not sure it should happen in real life - I think it means you are
                # trying this with an image that is smaller than the target size
                # which probably doesn't make sense and shouldn't be supported.
                y00 -= 1
            y01 = round((y + 1) * y_grid_size)
            if y01 == y00:
                y01 += 1  # ensure we have at least one pixel in the box
            y0 = round(y * y_grid_size + y_offset)
            if y0 >= img.height:
                # ensure we don't go out of bounds.
                # this is here to satisfy unit tests.
                # I'm not sure it should happen in real life - I think it means you are
                # trying this with an image that is smaller than the target size
                # which probably doesn't make sense and shouldn't be supported.
                y0 -= 1
            # get the pixel at (x, y)
            pixel = _safe_getpixel(img, (x0, y0))

            if (x, y) in interesting:
                if img.mode == "P":
                    # keep Pylance happy (and should never have palette mode image without palette!)
                    assert original_palette is not None

                    pixel_rgb = original_palette.get_color(pixel)
                else:
                    pixel_rgb = pixel
                _debug_print(
                    f"\nPixel at ({x}, {y}): pixel (x: {x0}, y: {y0}): {pixel} ({pixel_rgb}) [{interesting_labeled[(x, y)]}]"
                )

            # average color in the box algorithm
            # does this really work? Copilot suggested it.
            # pixel = img.crop(box).convert("RGB").resize((1, 1)).getpixel((0, 0))

            # most common color in the box algorithm
            all_pixels = get_all_pixels(img, x00, y00, x01, y01)
            color_counts = get_color_counts(all_pixels)

            if (x, y) in interesting:
                _debug_print("Pixel color counts:", color_counts)

            if (x, y) in interesting:
                _debug_print("Boosted pixel color counts:", color_counts)

            # Filter color_counts to only include items with a count > 164
            # Although we're doing this to capture small details, some details are so small we don't
            # want to magnify them in the downsampled image.
            # This threshold may need to be adjusted for different images and original image sizes.
            # 164 worked for logo.gif that I was designing this to work with.
            color_counts_filtered = Counter(
                {color: count for color, count in color_counts.items() if count > 412}
            )
            # if no colors are left
            # (again, probably should adapt the threshold for different images, probably by image size)
            if len(color_counts_filtered) == 0:
                color_counts_filtered = color_counts

            if (x, y) in interesting:
                _debug_print("Filtered pixel color counts:", color_counts_filtered)

            # # see which color has the most pixels
            most_common_colors = color_counts_filtered.most_common(2)
            most_common_color = most_common_colors[0]

            if len(most_common_colors) > 1:
                secondmost_common_color = most_common_colors[1]
                if (x, y) in interesting:
                    _debug_print(
                        f"Storing second most common color: {secondmost_common_color}"
                    )
            else:
                secondmost_common_color = most_common_colors[0]

            most_common_color_pixel = most_common_color[0]

            # OR
            # brightest color in the box algorithm

            # TODO: Just convert everything to RGB(A) and not need 3 branches?
            if original_palette:
                brightest_color = max(
                    color_counts_filtered,
                    # TODO! Shouldn't there be a divided by 3 with the sum (and below)?
                    key=lambda c: sum(original_palette[c * 3 : c * 3 + 3]),
                )
            elif img.mode.startswith("RGB"):
                brightest_color = max(
                    color_counts_filtered,
                    # TODO! Shouldn't there be a divided by 3 with the sum (and above)?
                    # TODO: should it be a float?
                    key=lambda c: sum(map(int, c)),  # sum RGB values
                )
            elif img.mode == "L":  # grayscale image
                brightest_color = max(
                    color_counts_filtered, key=lambda c: c  # just the value
                )
            else:
                raise ValueError(f"Unknown image mode '{img.mode}'.")
            # brightest_color_pixel = brightest_color
            # if (x, y) in interesting:
            #     if original_palette:
            #         _debug_print(
            #             f"Brightest color: {brightest_color_pixel} ({original_palette[brightest_color_pixel*3:brightest_color_pixel*3+3]})"
            #         )
            #     else:
            #         _debug_print(
            #             f"Brightest color: {brightest_color_pixel} ({brightest_color})"
            #         )

            # HYBRID = False
            # if HYBRID:
            #     # Determine the color of the pixel to the left
            #     left_pixel = None
            #     top_pixel = None
            #     if x > 0:
            #         left_pixel = new_image.getpixel((x - 1, y))
            #     if y > 0:
            #         top_pixel = new_image.getpixel((x, y - 1))
            #     if (
            #         left_pixel == most_common_color_pixel
            #         and top_pixel == most_common_color_pixel
            #     ):
            #         pixel = brightest_color_pixel
            #     else:
            #         pixel = most_common_color_pixel

            # draw coordinates on an image we can use for debugging
            if DEBUG:
                # add text to the new image
                draw.text(  # pyright: ignore[reportPossiblyUnboundVariable]
                    (x00 + 1, y00 + 1),
                    f"{x}, {y}",
                    fill=(255, 0, 255),
                    font=font,  # pyright: ignore[reportPossiblyUnboundVariable]
                )

            # map pixel to new pixel
            new_image.putpixel((x, y), pixel)
            new_image_secondmost.putpixel((x, y), secondmost_common_color[0])
    new_image_secondmost.save("debug.secondmost.py.png")

    # TODO: "naive?" mode that ignores the second pass (what did Copilot call this - mode pooling?)
    #   and compare this to non-"naive" result with logo.gif

    # second pass: find any pixels that have a secondmost common color
    # but it's current color is BRIGHTER THAN ALL of its neighbors' most current color

    new_image_copy = new_image.copy()

    new_image_secondmost_palette = new_image_secondmost.getpalette()

    for x in range(new_width):
        for y in range(new_height):
            pixel1 = new_image.getpixel((x, y))
            pixel2 = new_image_secondmost.getpixel((x, y))
            if pixel1 == pixel2:
                continue

            # Get neighbors' most common colors (from the copy of new_image made above,
            #   not from the image we're modifying)
            neighbors_brightness = []
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < new_width and 0 <= ny < new_height:
                        neighbor_pixel = _safe_getpixel(new_image_copy, (nx, ny))
                        if new_image_secondmost_palette:
                            assert isinstance(neighbor_pixel, int)
                            neighbor_pixel_brightness = (
                                sum(
                                    new_image_secondmost_palette[
                                        neighbor_pixel * 3 : neighbor_pixel * 3 + 3
                                    ]
                                )
                                / 3
                            )
                        elif new_image_copy.mode.startswith("RGB"):
                            assert isinstance(neighbor_pixel, tuple)
                            # if the image is in RGB mode, just use the pixel value directly
                            neighbor_pixel_brightness = sum(neighbor_pixel) / 3
                        elif new_image_copy.mode == "L":  # grayscale image
                            # if the image is in grayscale mode, just use the pixel value directly
                            neighbor_pixel_brightness = neighbor_pixel
                        else:
                            raise ValueError(
                                f"Unknown image mode '{new_image_copy.mode}'."
                            )
                        neighbors_brightness.append(neighbor_pixel_brightness)

            if new_image_secondmost_palette:
                assert isinstance(pixel2, int)
                pixel2_brightness = (
                    sum(new_image_secondmost_palette[pixel2 * 3 : pixel2 * 3 + 3]) / 3
                )
            elif new_image_secondmost.mode.startswith("RGB"):
                assert isinstance(pixel2, tuple)
                # if the image is in RGB mode, just use the pixel value directly
                pixel2_brightness = sum(pixel2) / 3
            elif new_image_secondmost.mode == "L":  # grayscale image

                # if the image is in grayscale mode, just use the pixel value directly
                pixel2_brightness = pixel2
            else:
                raise ValueError(f"Unknown image mode '{new_image_secondmost.mode}'.")

            brighter_pixels_found = 0
            MAX_BRIGHTER_PIXELS = 3
            for b in neighbors_brightness:
                # this is another threshold that worked experimentally
                # and may need to be adjusted for different images
                assert pixel2_brightness is not None
                assert not isinstance(pixel2_brightness, tuple)
                if b >= pixel2_brightness - 15:
                    brighter_pixels_found += 1
                    # optimization but doesn't give us all debugging info:
                    # if brighter_pixels_found > MAX_BRIGHTER_PIXELS:
                    #     break

            # maybe even more accurate:
            # in the convolution of this pixel and the surrounding pixels (3x3 grid),
            # - does this pixel have a second-most-common color that is significantly brighter
            #   than all of its neighbors' (in the first-pass image) most common color?
            # - AND is one of the top two (in terms of how bright that box is in the original image?) of itself and its neighbors in terms of being "turned back on"?
            # another related method might be to compare to a bilinear resampling of the original image?

            if (x, y) in interesting:
                _debug_print(
                    f"Pixel ({x}, {y}) [{interesting_labeled[x, y]}]: "
                    f"\n\tcolor: {pixel1} \n\tsecondary color: {pixel2} "
                    f"\n\tneighbors' brightness: {neighbors_brightness}\n\tpixel brightness: {pixel2_brightness}"
                    f"\n\tsignificantly brighter pixels found (among neighbors): {brighter_pixels_found}",
                )

            if brighter_pixels_found > MAX_BRIGHTER_PIXELS:
                continue

            # this pixel could be "turned back on"
            # but first, see how bright it is in a bilinear resampling of the original image?
            _debug_print(x, y, "bilinear resampling")
            bilinear_pixel = new_image_bilinear.getpixel((x, y))
            # bilinear_pixel_rgb = new_image_bilinear_palette[
            #     bilinear_pixel * 3 : bilinear_pixel * 3 + 3
            # ]
            bilinear_pixel_brightness = bilinear_pixel  # sum(bilinear_pixel_rgb) / 3
            # TODO: if this is all we need bilinear for, we could make it grayscale
            if (x, y) in interesting:
                _debug_print(
                    f"\tbilinear pixel: {bilinear_pixel}, brightness: {bilinear_pixel_brightness}"
                )

            assert bilinear_pixel_brightness is not None
            assert not isinstance(bilinear_pixel_brightness, tuple)
            if bilinear_pixel_brightness < 10:
                if (x, y) in interesting:
                    _debug_print("\ttoo dark, never mind")
                continue

            assert pixel2 is not None
            new_image.putpixel((x, y), pixel2)
            # new_image.putpixel((x, y), GREEN)  # debugging

            # if pixel1 == most_common_neighbor and most_common_neighbor_count > 4:
            #     _debug_print("\t***Changing to secondmost color")

            # if pixel1 == most_common_neighbor and most_common_neighbor_count == 8:

    if DEBUG:
        draw_grid(
            debug_image,  # pyright: ignore[reportPossiblyUnboundVariable]
            original_width,
            original_height,
            new_width,
            new_height,
        )
        # save image with grid lines
        debug_image.save(  # pyright: ignore[reportPossiblyUnboundVariable]
            "debug0.py.png"
        )

    _debug_print("Image resized to:", new_image.size)
    new_image.save("debug.py.png")

    pixel_array = np.array(new_image)

    # *** Turn the scaled-down image into ASCII ***
    f = sys.stdout
    new_image_palette = None
    if new_image.mode == "P":
        new_image_palette = _WrappedPalette.from_image(new_image)

    # uncomment following line to write to a file:
    # with open(output_file, "w") as f:
    for row in pixel_array:
        line = []
        for pixel in row:
            match color_mode:
                case "monotone":
                    # _debug_print("pixel is", pixel, type(pixel))
                    line.append(map_value(pixel, alphabet_name=alphabet_name))
                case "true-color":
                    if new_image.mode == "P":
                        assert new_image_palette is not None
                        # if the image is in palette mode, get the RGB value from the palette
                        pixel = int(pixel)  # ensure pixel is int, not numpy.uint8
                        pixel = new_image_palette.get_color(pixel)
                    line.append(map_rgb_value(pixel, alphabet_name=alphabet_name))
                case _:
                    raise ValueError(f"Unsupported color mode: {color_mode}")
        line = "".join(line)
        f.write(line + "\n")
    f.write("\033[0m")  # reset color


def draw_grid(img, width, height, new_width, new_height):
    draw = ImageDraw.Draw(img)
    for x in range(new_width):
        x0 = round(x * width / new_width)
        draw.line((x0, 0, x0, height), fill=(255, 0, 255))  # magenta for debugging
    for y in range(new_height):
        y0 = round(y * height / new_height)
        draw.line((0, y0, width, y0), fill=(255, 0, 255))  # magenta for debugging


def preview_alphabets(args):
    for alphabet in alphabets.keys():
        if alphabet[:2] == "__" and alphabet[-2:] == "__":
            continue
        args.alphabet = alphabet
        print("\033[2J\033[H", end="")  # clear the screen
        convert_to_ascii_with_args(args)
        inp = input(
            f"Showing '{alphabet}' alphabet. Press 'q' then ENTER to quit, or ENTER to continue..."
        )
        if inp == "q":
            break


def convert_to_ascii_with_args(args):
    if args.debug:
        global DEBUG
        DEBUG = True
        _debug_print("Debug mode is enabled.")

    return convert_to_ascii(
        args.image_path,
        alphabet_name=args.alphabet,
        color_mode=args.color_mode,
    )
