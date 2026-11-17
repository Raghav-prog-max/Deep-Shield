import hashlib
import re
import uuid
import warnings
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .config import Settings
from .database import Database, utc_now


IMAGE_EXTENSIONS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
CHUNK_SIZE = 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


class MediaService:
    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database
        self.root = settings.data_dir / "media"

    def accept(self, upload: UploadFile) -> dict:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="A filename is required")
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS and suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported file extension")

        self.root.mkdir(parents=True, exist_ok=True)
        media_id = str(uuid.uuid4())
        storage_name = f"{media_id}{suffix}"
        temp_path = self.root / f".{media_id}.upload"
        final_path = self.root / storage_name
        digest = hashlib.sha256()
        size = 0
        try:
            with temp_path.open("xb") as stream:
                while chunk := upload.file.read(CHUNK_SIZE):
                    size += len(chunk)
                    if size > self.settings.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Upload exceeds the configured size limit")
                    digest.update(chunk)
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            try:
                media_type = self._validate(temp_path, suffix)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Media could not be validated: {exc}") from exc
            safe_name = self._safe_original_name(upload.filename)
            record = {
                "media_id": media_id,
                "original_name": safe_name,
                "media_type": media_type,
                "storage_name": storage_name,
                "sha256": digest.hexdigest(),
                "size_bytes": size,
                "created_at": utc_now(),
            }
            temp_path.replace(final_path)
            try:
                self.database.add_media(record)
            except Exception:
                final_path.unlink(missing_ok=True)
                raise
            return {key: value for key, value in record.items() if key != "storage_name"}
        except HTTPException:
            temp_path.unlink(missing_ok=True)
            raise
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=f"Media could not be validated: {exc}") from exc
        finally:
            upload.file.close()

    def _validate(self, path: Path, suffix: str) -> str:
        if suffix in IMAGE_EXTENSIONS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(path) as image:
                        if image.format != IMAGE_EXTENSIONS[suffix]:
                            raise ValueError("Image content does not match a supported format")
                        if image.width * image.height > MAX_IMAGE_PIXELS:
                            raise ValueError("Image exceeds the pixel limit")
                        image.verify()
            except Image.DecompressionBombError as exc:
                raise ValueError("Image exceeds the pixel limit") from exc
            except Image.DecompressionBombWarning as exc:
                raise ValueError("Image exceeds the pixel limit") from exc
            return "image"

        try:
            import cv2
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="Video validation is unavailable: OpenCV is not installed") from exc
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError("Video decoder could not open this file")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError("Video contains no readable frame")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or frame_count <= 0:
                raise ValueError("Video duration could not be determined")
            duration = frame_count / fps
            if duration > self.settings.max_video_seconds:
                raise HTTPException(
                    status_code=413,
                    detail=f"Video exceeds the configured {self.settings.max_video_seconds:g}-second limit",
                )
            return "video"
        finally:
            capture.release()

    @staticmethod
    def _safe_original_name(filename: str) -> str:
        name = Path(filename.replace("\\", "/")).name
        name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
        return name[:255] or "uploaded-media"

    def get_path(self, media_record: dict) -> Path:
        return self.root / media_record["storage_name"]
