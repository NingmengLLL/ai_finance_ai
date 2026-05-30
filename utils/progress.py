from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from utils.logger_handler import logger


T = TypeVar("T")


class PlainProgress:
    def __init__(self, iterable: Iterable[T], desc: str = "", unit: str = "", total: int | None = None):
        self.iterable = iterable
        self.desc = desc
        self.unit = unit
        self.total = total

    def __iter__(self):
        for index, item in enumerate(self.iterable, start=1):
            total_text = f"/{self.total}" if self.total is not None else ""
            unit_text = f" {self.unit}" if self.unit else ""
            logger.info(f"{self.desc}: {index}{total_text}{unit_text}")
            yield item

    def set_postfix_str(self, text: str) -> None:
        logger.info(text)


def progress_bar(
    iterable: Iterable[T],
    desc: str = "",
    unit: str = "",
    total: int | None = None,
    enabled: bool = True,
):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, unit=unit, total=total, dynamic_ncols=True, ascii=True)
    except Exception:
        return PlainProgress(iterable, desc=desc, unit=unit, total=total)


def set_progress_detail(progress, text: str) -> None:
    if hasattr(progress, "set_postfix_str"):
        progress.set_postfix_str(text)
    else:
        logger.info(text)
