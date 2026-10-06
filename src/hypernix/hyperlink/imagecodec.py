"""Images sent into HyperLink: recognised by their bytes, stored as WebP.

A phone photo is 3-12 MB of JPEG or HEIC with the location it was taken
in its EXIF; a screenshot is a PNG twice the size it needs to be. Neither
is what a vision model wants (they downscale to around 1-2 megapixels
anyway) and both are what crosses a cellular link and sits in the
attachment store. So an image upload is converted once, on arrival:

* **rasters become WebP** -- lossless for the formats that are lossless
  (PNG, GIF, BMP, TIFF, QOI, ...), unless that comes out large, and
  quality 85 for photographic ones (JPEG, HEIC/HEIF, AVIF, JPEG XL, JPEG
  2000, DNG/raw). Animated GIF/APNG/WebP stay animated;
* **the long edge is capped** at :data:`MAX_EDGE` pixels, and the EXIF
  orientation applied first so the picture is the right way up;
* **metadata is dropped** -- EXIF, GPS, XMP, maker notes -- since
  re-encoding writes none of it;
* **SVG stays SVG**, minus everything that can run or fetch: scripts,
  ``on*`` handlers, ``foreignObject``, ``javascript:`` and external
  references. It is vector text, and a model reads it as such.

Formats are recognised by their magic numbers, never the filename or
the client's claim. Decoders beyond Pillow are optional and used when
installed: ``pillow-heif`` (HEIC/HEIF), ``pillow-jxl-plugin`` (JPEG XL),
``rawpy`` (DNG and camera raw). Without ``rawpy``, a DNG's own embedded
full-size JPEG preview is used. When nothing can decode an upload, the
original bytes are stored and the result says why: an upload is never
refused for being in a format this machine cannot convert.

Decoding untrusted images is attack surface. Pillow's decompression-bomb
guard is set to :data:`MAX_PIXELS` and enforced as an error, and nothing
here writes a file anywhere but into the returned bytes.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

__all__ = [
    "CompressResult",
    "MAX_EDGE",
    "MAX_PIXELS",
    "RASTER_TYPES",
    "SVG_TYPE",
    "compress",
    "sniff_image",
]

#: Long edge after conversion. Vision encoders work at 336-1568 px; 2048
#: keeps text in a screenshot legible with room to spare.
MAX_EDGE = 2048
#: Refuse to decode beyond this many pixels (a 16-bit 12000x12000 TIFF
#: is ~860 MB in memory). Larger than any camera sensor in a phone.
MAX_PIXELS = 160_000_000
#: Lossless WebP larger than this is re-encoded lossy instead.
LOSSLESS_CEILING = 1_500_000
WEBP_QUALITY = 85

SVG_TYPE = "image/svg+xml"

#: Content types this module recognises, by magic number.
RASTER_TYPES = frozenset({
    "image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp",
    "image/tiff", "image/heic", "image/heif", "image/avif", "image/jxl",
    "image/jp2", "image/x-adobe-dng", "image/x-icon", "image/qoi",
    "image/vnd.adobe.photoshop", "image/x-tga",
})

#: Already-efficient formats, kept when WebP would only make them bigger.
_ALREADY_SMALL = frozenset({"image/webp", "image/avif"})

#: Formats whose pixels were never lossy: converted losslessly first.
_LOSSLESS_SOURCES = frozenset({
    "image/png", "image/gif", "image/bmp", "image/tiff", "image/x-icon",
    "image/qoi", "image/vnd.adobe.photoshop", "image/x-tga",
})


@dataclass
class CompressResult:
    data: bytes
    content_type: str
    filename: str
    converted: bool
    original_type: str = ""
    original_bytes: int = 0
    width: int = 0
    height: int = 0
    notes: list[str] = field(default_factory=list)

    def metadata(self) -> dict:
        """What is kept beside the attachment about the conversion."""
        return {
            "original_type": self.original_type,
            "original_bytes": self.original_bytes,
            "converted": self.converted,
            **({"width": self.width, "height": self.height} if self.width else {}),
            **({"notes": self.notes} if self.notes else {}),
        }


# ---------------------------------------------------------------------------
# Recognising
# ---------------------------------------------------------------------------

_FTYP = {
    b"heic": "image/heic", b"heix": "image/heic", b"hevc": "image/heic", b"hevx": "image/heic",
    b"heim": "image/heic", b"heis": "image/heic", b"mif1": "image/heif", b"msf1": "image/heif",
    b"avif": "image/avif", b"avis": "image/avif",
}


def sniff_image(data: bytes, filename: str = "") -> str:
    """The image type of *data* by its bytes, or ``""`` if not an image.

    *filename* only separates a DNG from other TIFFs when the TIFF header
    alone cannot (the DNGVersion tag is checked first).
    """
    head = data[:64]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head.startswith(b"BM") and len(data) > 26:
        return "image/bmp"
    if head[4:8] == b"ftyp" and head[8:12] in _FTYP:
        return _FTYP[head[8:12]]
    if head.startswith(b"\xff\x0a") or head.startswith(b"\x00\x00\x00\x0cJXL \r\n\x87\n"):
        return "image/jxl"
    if head.startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n") or head.startswith(b"\xff\x4f\xff\x51"):
        return "image/jp2"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        if _is_dng(data) or filename.lower().endswith(".dng"):
            return "image/x-adobe-dng"
        return "image/tiff"
    if head.startswith(b"\x00\x00\x01\x00") and len(data) > 22:
        return "image/x-icon"
    if head.startswith(b"qoif"):
        return "image/qoi"
    if head.startswith(b"8BPS"):
        return "image/vnd.adobe.photoshop"
    if _looks_svg(data):
        return SVG_TYPE
    if filename.lower().endswith(".tga") and len(data) > 18:
        return "image/x-tga"
    return ""


def _is_dng(data: bytes) -> bool:
    """A TIFF whose first IFD carries DNGVersion (tag 50706)."""
    import struct

    if len(data) < 16:
        return False
    order = "<" if data[:2] == b"II" else ">"
    try:
        (offset,) = struct.unpack(order + "I", data[4:8])
        (count,) = struct.unpack(order + "H", data[offset:offset + 2])
        for i in range(min(count, 512)):
            at = offset + 2 + i * 12
            (tag,) = struct.unpack(order + "H", data[at:at + 2])
            if tag == 50706:
                return True
    except (struct.error, IndexError):
        return False
    return False


def _looks_svg(data: bytes) -> bool:
    sample = data[:4096].lstrip(b"\xef\xbb\xbf \t\r\n")
    if not sample.startswith(b"<"):
        return False
    try:
        text = sample.decode("utf-8", "strict").lower()
    except UnicodeDecodeError:
        return False
    return "<svg" in text


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

_SVG_DROP = re.compile(
    r"<\s*(script|foreignobject|iframe|object|embed|handler|listener)\b.*?<\s*/\s*\1\s*>"
    r"|<\s*(script|foreignobject|iframe|object|embed|handler|listener)\b[^>]*/\s*>",
    re.I | re.S,
)
_SVG_EVENT = re.compile(r"\s+on[a-z0-9_-]+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", re.I)
_SVG_HREF = re.compile(
    r"\s+(?:xlink:)?href\s*=\s*(\"|')\s*(?!#|data:image/(?:png|jpe?g|gif|webp);)[^\"']*\1", re.I)
_SVG_ENTITY = re.compile(r"<!DOCTYPE[^>\[]*(\[.*?\])?\s*>", re.I | re.S)


def sanitize_svg(data: bytes) -> bytes:
    """SVG with nothing that runs, fetches, or expands.

    Removes script-like elements, ``on*`` handlers, any ``href`` that is
    not a fragment or an inline raster, and the DOCTYPE (where entity
    expansion -- "billion laughs" -- lives). Comments go too, and the
    whitespace between tags, which is most of what an exported SVG
    wastes.
    """
    text = data.decode("utf-8", "replace")
    text = _SVG_ENTITY.sub("", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    previous = None
    while previous != text:            # nested tricks: <scr<script>ipt>
        previous = text
        text = _SVG_DROP.sub("", text)
    text = _SVG_EVENT.sub("", text)
    text = _SVG_HREF.sub("", text)
    text = re.sub(r"javascript\s*:", "", text, flags=re.I)
    text = re.sub(r">\s+<", "><", text).strip()
    return text.encode("utf-8")


# ---------------------------------------------------------------------------
# Rasters
# ---------------------------------------------------------------------------


def _pillow():
    from PIL import Image

    for plugin, opener in (("pillow_heif", "register_heif_opener"),
                           ("pillow_avif", "register_avif_opener")):
        try:
            getattr(__import__(plugin), opener)()
        except (ImportError, AttributeError):
            pass
    try:
        __import__("pillow_jxl")           # registers itself on import
    except ImportError:
        pass
    return Image


def _dng_preview(data: bytes) -> bytes | None:
    """The largest JPEG embedded in a DNG: its full-size preview.

    Every DNG written by a camera or a phone carries one. Without rawpy
    it is the picture as the camera rendered it, which for a chat is
    what was wanted anyway. Each embedded JPEG is measured by Pillow
    rather than cut at the first end-of-image marker, since a preview
    can itself contain a thumbnail with one.
    """
    Image = _pillow()
    best, best_pixels = None, 0
    start = data.find(b"\xff\xd8\xff", 8)
    for _ in range(32):
        if start == -1:
            break
        try:
            with Image.open(io.BytesIO(data[start:])) as probe:
                pixels = probe.size[0] * probe.size[1]
                if probe.format == "JPEG" and pixels > best_pixels:
                    best, best_pixels = data[start:], pixels
        except Exception:  # noqa: BLE001 - not every marker starts a whole JPEG
            pass
        start = data.find(b"\xff\xd8\xff", start + 3)
    return best


def _open(data: bytes, kind: str, notes: list[str]):
    Image = _pillow()
    if kind == "image/x-adobe-dng":
        try:
            import rawpy

            with rawpy.imread(io.BytesIO(data)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False)
            return Image.fromarray(rgb)
        except ImportError:
            preview = _dng_preview(data)
            if preview is not None:
                notes.append("DNG converted from its embedded preview (install rawpy to "
                             "develop the raw data)")
                return Image.open(io.BytesIO(preview))
        except Exception as exc:  # noqa: BLE001 - fall through to Pillow's TIFF reader
            notes.append(f"rawpy could not develop it: {exc}")
    return Image.open(io.BytesIO(data))


def _fit(image, notes: list[str]):
    from PIL import ImageOps

    image = ImageOps.exif_transpose(image)
    width, height = image.size
    if max(width, height) > MAX_EDGE:
        image.thumbnail((MAX_EDGE, MAX_EDGE))
        notes.append(f"resized from {width}x{height}")
    return image


def _as_webp(image, *, lossless: bool, animated: bool) -> bytes:
    out = io.BytesIO()
    mode = "RGBA" if "A" in image.getbands() or image.mode in ("P", "LA") else "RGB"
    options = {"format": "WEBP", "method": 6}
    if lossless:
        options.update(lossless=True, quality=100)
    else:
        options.update(quality=WEBP_QUALITY)
    if animated:
        from PIL import ImageSequence

        frames = [f.convert(mode) for f in ImageSequence.Iterator(image)]
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=image.info.get("duration", 100), loop=image.info.get("loop", 0),
                       **options)
    else:
        image.convert(mode).save(out, **options)
    return out.getvalue()


def _webp_name(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0] if "." in filename else (filename or "image")
    return f"{stem or 'image'}.webp"


def compress(data: bytes, filename: str = "", *, declared: str = "") -> CompressResult | None:
    """Convert an image upload, or ``None`` if *data* is not an image.

    Never raises for a bad image: an upload this cannot decode is
    returned unchanged with the reason in ``notes``.
    """
    kind = sniff_image(data, filename)
    if not kind:
        return None
    if kind == SVG_TYPE:
        cleaned = sanitize_svg(data)
        name = filename if filename.lower().endswith(".svg") else _webp_name(filename)[:-5] + ".svg"
        return CompressResult(cleaned, SVG_TYPE, name, converted=cleaned != data,
                              original_type=kind, original_bytes=len(data),
                              notes=["SVG kept as SVG, scripts and external references removed"])

    notes: list[str] = []
    original = CompressResult(data, kind, filename or "image", converted=False,
                              original_type=kind, original_bytes=len(data), notes=notes)
    import warnings

    try:
        with warnings.catch_warnings():
            # A bomb warning is a refusal here, not a warning: the
            # alternative is decoding gigabytes because a phone said so.
            from PIL import Image as _Image

            warnings.simplefilter("error", _Image.DecompressionBombWarning)
            image = _open(data, kind, notes)
            width, height = image.size
            if width * height > MAX_PIXELS:
                raise ValueError(f"{width}x{height} is over {MAX_PIXELS} pixels")
            animated = bool(getattr(image, "is_animated", False)) and getattr(image, "n_frames", 1) > 1
            image.load()
        if not animated:
            image = _fit(image, notes)
        lossless = kind in _LOSSLESS_SOURCES
        encoded = _as_webp(image, lossless=lossless, animated=animated)
        if lossless and len(encoded) > LOSSLESS_CEILING:
            encoded = _as_webp(image, lossless=False, animated=animated)
            notes.append(f"lossy WebP: lossless was over {LOSSLESS_CEILING} bytes")
        width, height = image.size
    except ImportError:
        notes.append("Pillow is not installed (pip install 'hypernix[images]'); stored as sent")
        return original
    except Exception as exc:  # noqa: BLE001 - an undecodable upload is stored, not refused
        hint = {
            "image/heic": " (install pillow-heif)", "image/heif": " (install pillow-heif)",
            "image/jxl": " (install pillow-jxl-plugin)",
            "image/x-adobe-dng": " (install rawpy)",
        }.get(kind, "")
        notes.append(f"could not convert {kind}{hint}: {type(exc).__name__}: {exc}; stored as sent")
        return original
    resized = any(n.startswith("resized") for n in notes)
    if kind in _ALREADY_SMALL and not resized and len(encoded) >= len(data):
        # AVIF beats WebP, and a WebP re-encoded is a WebP made worse:
        # converting either only to grow it helps nobody.
        notes.append(f"kept as {kind}: WebP was not smaller")
        return CompressResult(data, kind, filename or "image", converted=False,
                              original_type=kind, original_bytes=len(data),
                              width=width, height=height, notes=notes)
    return CompressResult(encoded, "image/webp", _webp_name(filename), converted=True,
                          original_type=kind, original_bytes=len(data),
                          width=width, height=height, notes=notes)
