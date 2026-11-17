from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    max_upload_bytes: int
    max_video_seconds: float
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        max_upload_bytes = int(os.getenv("DEEP_SHIELD_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
        max_video_seconds = float(os.getenv("DEEP_SHIELD_MAX_VIDEO_SECONDS", "60"))
        if max_upload_bytes <= 0 or max_video_seconds <= 0:
            raise ValueError("Upload and video limits must be positive")
        return cls(
            data_dir=Path(os.getenv("DEEP_SHIELD_DATA_DIR", "./data")).expanduser().resolve(),
            max_upload_bytes=max_upload_bytes,
            max_video_seconds=max_video_seconds,
            cors_origins=_csv(os.getenv(
                "DEEP_SHIELD_CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000",
            )),
        )


settings = Settings.from_env()
