"""Images sent into HyperLink are stored as WebP, or sanitized SVG.

Real images, encoded by Pillow, go through the real codec: the claims
here are about bytes on disk -- that a photo's GPS is gone, that it is
the right way up, that an SVG's script is not there -- and a mock of
the encoder could not make any of them.
"""
from __future__ import annotations

import io
import struct

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

from hypernix.hyperlink import imagecodec as ic  # noqa: E402


def _picture(size=(640, 480)):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for x in range(0, size[0], 32):
        draw.line([(x, 0), (size[0] - x, size[1])], fill=(x % 255, 90, 170))
    draw.text((16, 16), "screenshot text", fill="black")
    return image


def _encode(image, fmt, **options) -> bytes:
    out = io.BytesIO()
    image.save(out, fmt, **options)
    return out.getvalue()


def _decode(data: bytes):
    return Image.open(io.BytesIO(data))


@pytest.mark.parametrize("fmt,kind", [
    ("PNG", "image/png"), ("JPEG", "image/jpeg"), ("GIF", "image/gif"),
    ("BMP", "image/bmp"), ("TIFF", "image/tiff"), ("WEBP", "image/webp"),
])
def test_rasters_are_recognised_by_their_bytes(fmt, kind):
    data = _encode(_picture(), fmt)
    assert ic.sniff_image(data, "misleading-name.txt") == kind


def test_a_png_screenshot_becomes_a_smaller_webp():
    data = _encode(_picture(), "PNG")
    result = ic.compress(data, "shot.png")
    assert result.content_type == "image/webp" and result.filename == "shot.webp"
    assert len(result.data) < len(data)
    assert _decode(result.data).format == "WEBP"


def test_a_phone_photo_loses_its_gps_and_is_the_right_way_up():
    photo = Image.effect_mandelbrot((3000, 2000), (-2, -1.5, 1, 1.5), 60).convert("RGB")
    exif = Image.Exif()
    exif[0x0112] = 6                                # rotate 90° on display
    exif[0x8825] = {2: (51.0, 30.0, 0.0)}           # GPS latitude
    data = _encode(photo, "JPEG", quality=95, exif=exif)
    result = ic.compress(data, "IMG_0001.JPG")
    out = _decode(result.data)
    assert not out.getexif()                        # GPS, and everything else, gone
    assert out.size[1] > out.size[0]                # portrait, as it was shot
    assert max(out.size) <= ic.MAX_EDGE
    assert any("resized" in n for n in result.notes)


def test_an_animated_gif_stays_animated():
    frames = []
    for i in range(3):
        frame = Image.new("RGB", (120, 90), (i * 80, 20, 20))
        ImageDraw.Draw(frame).text((8, 8), str(i), fill="white")
        frames.append(frame)
    out = io.BytesIO()
    frames[0].save(out, "GIF", save_all=True, append_images=frames[1:], duration=90, loop=0)
    result = ic.compress(out.getvalue(), "a.gif")
    assert _decode(result.data).n_frames == 3


def test_a_dng_is_converted_from_its_embedded_preview():
    preview = _encode(_picture((1600, 1200)), "JPEG", quality=90)
    ifd = struct.pack("<H", 1) + struct.pack("<HHII", 50706, 1, 4, 0x00000401) + struct.pack("<I", 0)
    dng = b"II*\x00" + struct.pack("<I", 8) + ifd + b"\x00" * 64 + preview
    assert ic.sniff_image(dng) == "image/x-adobe-dng"
    result = ic.compress(dng, "RAW_0001.dng")
    if any("rawpy" in n and "could not" in n for n in result.notes):
        pytest.skip("rawpy is installed and refused the synthetic DNG")
    assert result.content_type == "image/webp"
    assert _decode(result.data).size == (1600, 1200)


def test_an_svg_stays_svg_without_anything_that_runs():
    svg = (b'<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY a "x">]>'
           b'<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)">'
           b'<script>alert(2)</script><foreignObject><b>x</b></foreignObject>'
           b'<a href="javascript:alert(3)"><rect width="10" height="10"/></a>'
           b'<image href="https://example.com/x.png"/><use href="#ok"/></svg>')
    result = ic.compress(svg, "logo.svg")
    text = result.data.decode()
    assert result.content_type == "image/svg+xml" and result.filename == "logo.svg"
    for gone in ("script", "onload", "javascript", "foreignObject", "example.com", "ENTITY"):
        assert gone.lower() not in text.lower(), gone
    assert "<rect" in text and 'href="#ok"' in text


@pytest.mark.parametrize("data,name,kind", [
    (b"\xff\x0a" + b"\0" * 64, "x.jxl", "image/jxl"),
    (b"\0\0\0\x0cJXL \r\n\x87\n" + b"\0" * 64, "x.jxl", "image/jxl"),
    (b"\0\0\0\x18ftypheic" + b"\0" * 64, "x.heic", "image/heic"),
    (b"\0\0\0\x18ftypavif" + b"\0" * 64, "x.avif", "image/avif"),
    (b"\0\0\0\x0cjP  \r\n\x87\n" + b"\0" * 64, "x.jp2", "image/jp2"),
    (b"qoif" + b"\0" * 64, "x.qoi", "image/qoi"),
    (b"8BPS" + b"\0" * 64, "x.psd", "image/vnd.adobe.photoshop"),
])
def test_more_formats_are_recognised(data, name, kind):
    assert ic.sniff_image(data, name) == kind


def test_what_cannot_be_decoded_is_stored_as_sent_with_the_reason():
    data = b"\xff\x0a" + b"\0" * 64              # a JPEG XL header with no image
    result = ic.compress(data, "x.jxl")
    assert result.data == data and not result.converted
    assert any("stored as sent" in n for n in result.notes)


def test_a_webp_that_would_only_grow_is_kept():
    data = _encode(_picture((200, 150)), "WEBP", quality=40)
    result = ic.compress(data, "small.webp")
    assert len(result.data) <= len(data)


def test_not_an_image_is_left_alone():
    assert ic.compress(b"print('hello')\n", "x.py") is None


def test_a_decompression_bomb_is_not_decoded():
    huge = Image.new("1", (20000, 20000))
    data = _encode(huge, "PNG")
    result = ic.compress(data, "bomb.png")
    assert not result.converted and result.data == data


# ---------------------------------------------------------------------------
# Through the upload route
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402


def _client(tmp_path, **overrides):
    km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
    gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False)
    config = T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough", db_path=str(tmp_path / "t1.db"),
        module_storage_dir=str(tmp_path / "m"), hyperlink_files_dir=str(tmp_path / "f"),
        default_plan="free", web_enabled=False, **overrides,
    )
    client = TestClient(create_app(config=config, keymaster=km, gatekeeper=gk),
                        client=("127.0.0.1", 5000))
    key = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key
    return client, {"Authorization": f"Bearer {key}"}


def test_an_uploaded_photo_is_stored_as_webp(tmp_path):
    client, auth = _client(tmp_path)
    data = _encode(_picture(), "PNG")
    got = client.post("/hyperlink/files", files={"file": ("shot.png", data, "image/png")},
                      headers=auth)
    assert got.status_code == 200, got.text
    stored = got.json()["file"]
    assert stored["content_type"] == "image/webp" and stored["filename"] == "shot.webp"
    assert stored["is_image"] and stored["metadata"]["image"]["original_type"] == "image/png"
    raw = client.get(f"/hyperlink/files/{stored['file_id']}", headers=auth).content
    assert _decode(raw).format == "WEBP"


def test_an_uploaded_svg_is_text_a_model_can_read(tmp_path):
    client, auth = _client(tmp_path)
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>x()</script><circle r="4"/></svg>'
    stored = client.post("/hyperlink/files", files={"file": ("d.svg", svg, "image/svg+xml")},
                         headers=auth).json()["file"]
    assert stored["content_type"] == "image/svg+xml" and stored["is_text"]


def test_the_switch_turns_it_off(tmp_path):
    client, auth = _client(tmp_path, hyperlink_image_compress=False)
    data = _encode(_picture(), "PNG")
    stored = client.post("/hyperlink/files", files={"file": ("shot.png", data, "image/png")},
                         headers=auth).json()["file"]
    assert stored["content_type"] == "image/png" and stored["size_bytes"] == len(data)
