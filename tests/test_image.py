import io
import sys

import numpy
import pytest
from PIL import Image

from aaart import image


def create_test_image(path, size=(10, 10), color=128):
    img = Image.new("L", size, color)
    img.save(path)


def test_convert_to_ascii_handles_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        image.convert_to_ascii("nonexistent.png", "out.txt")


def test_map_value_0():
    assert image.map_value(0) == " "


def test_map_value_255():
    assert image.map_value(255) == "@"


def test_map_value_solid():
    value = numpy.uint8(128)
    assert image.map_value(value, alphabet_name="solid") == "█"


def test_map_rgb_value():
    value = numpy.array([128, 128, 128], dtype=numpy.uint8)
    assert image.map_rgb_value(value).startswith(
        "\033[38;2;"  # ANSI escape code for RGB color
    )


def test_convert_to_ascii_with_unsupported_mode(tmp_path):
    # Create a test image
    img_path = tmp_path / "test_img.png"
    create_test_image(img_path)

    with pytest.raises(ValueError):
        image.convert_to_ascii(str(img_path), mode="unsupported")


def test_convert_to_ascii_calls_map_value_with_alphabet_name(monkeypatch, tmp_path):
    # Create a test image
    img_path = tmp_path / "test_img.png"
    create_test_image(img_path)

    called = {}

    def fake_map_value(value, alphabet_name="grayscale-chars", mode="monotone"):
        called["alphabet_name"] = alphabet_name
        # Return a dummy character
        return "X"

    # Patch map_value in image
    monkeypatch.setattr(image, "map_value", fake_map_value)

    # Call convert_to_ascii with a specific alphabet_name
    alphabet = "foo"
    image.convert_to_ascii(str(img_path), alphabet_name=alphabet)

    assert called["alphabet_name"] == alphabet


def test_convert_to_ascii_with_mode_color(monkeypatch, tmp_path):
    # Create a test image
    img_path = tmp_path / "test_img.png"
    create_test_image(img_path)

    was_called = False

    def fake_map_rgb_value(value, alphabet_name="grayscale-chars"):
        nonlocal was_called
        was_called = True
        # Return a dummy character
        return "X"

    # Patch map_value in image
    monkeypatch.setattr(image, "map_rgb_value", fake_map_rgb_value)

    # Call convert_to_ascii with a specific mode
    image.convert_to_ascii(str(img_path), mode="color")

    # Check that the patched function was called
    assert was_called


def test_calculate_new_size_fits_width():
    # Image fits within width, not too tall
    w, h = 100, 50
    max_w, max_h = 80, 40
    new_w, new_h = image.calculate_new_size(w, h, max_w, max_h, 0.5)
    assert new_w <= max_w
    assert new_h <= max_h
    assert new_w > 0 and new_h > 0


def test_calculate_new_size_fits_height():
    # Image would be too tall, so width is reduced
    w, h = 100, 200
    max_w, max_h = 80, 40
    new_w, new_h = image.calculate_new_size(w, h, max_w, max_h, 0.5)
    assert new_w <= max_w
    assert new_h <= max_h
    assert new_w > 0 and new_h > 0


def test_calculate_new_size_square():
    # Square image, square terminal
    w, h = 50, 50
    max_w, max_h = 50, 50
    new_w, new_h = image.calculate_new_size(w, h, max_w, max_h, 1.0)
    assert new_w <= max_w
    assert new_h <= max_h
    assert new_w > 0 and new_h > 0


def test_get_all_pixels_len():
    # Create a test image
    img = Image.new("L", (30, 30), 128)
    pixels = image.get_all_pixels(img, 10, 10, 20, 20)
    assert len(pixels) == 100  # 10x10 image should have 100 pixels


def test_enhance_gamma_value_0():
    assert image.enhance_gamma_value(0) == 0


def test_enhance_gamma_value_255():
    assert image.enhance_gamma_value(255) == 255


def test_enhance_gamma_value_midpoint():
    # I'm not sure I'm even going to use this,
    # and when I was I used a gamma of 0.2,
    # but let's get the test to pass in case we need it later.
    assert image.enhance_gamma_value(128, gamma=0.41) == 192


def test_convert_to_ascii_with_monotone_color(tmp_path):
    # Create a test image
    img_path = tmp_path / "test_img.png"
    create_test_image(img_path)

    # Redirect stdout to capture output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Call convert_to_ascii with a specific mode
    image.convert_to_ascii(str(img_path), colors="green")

    # Get the output and restore stdout
    ascii_output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    assert "\033[32m" in ascii_output


def test_logo():
    """
    This is more of an integration (end-to-end) test.
    """

    # Redirect stdout to capture output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Convert the image to ASCII art and print to stdout
    image.convert_to_ascii(
        R"tests\logo.gif",
        mode="color",
        alphabet_name="ultra-wide",
        width=148,
        height=32,
    )

    # Get the output and restore stdout
    ascii_output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # save the output to a file for debugging purposes
    # with open("tests/output_logo.txt", "w", encoding="utf-16") as f:
    #     f.write(ascii_output)

    # Optionally, check that output is not empty
    assert ascii_output.strip() != ""

    # see if it exactly matches the expected out stored in tests
    with open("tests/expected_logo_output.txt", encoding="utf-16") as f:
        expected_output = f.read()
    ascii_output = ascii_output.strip("\n")
    expected_output = expected_output.strip("\n")

    assert ascii_output == expected_output


def test_example_jpg():
    # this is just making sure this runs without error
    image.convert_to_ascii(R"tests\pexels-creationhill-1681010.jpg", mode="color")
