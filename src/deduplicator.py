#!/usr/bin/env python3
# coding: utf-8

"""
    Find duplicate files by content
"""
from __future__ import division, print_function, unicode_literals

import errno
import fnmatch
import hashlib
import itertools
import operator
import os
import sys
import time
from typing import Iterable, Sequence
from argparse import ArgumentParser

assert sys.version_info[:2] >= (3, 6), "Only Python 3.6+ is supported"

__version__ = "Deduplicator v1.1"

# Search options
EXCLUDED_DIR_BEGINS = frozenset(["."])
EXCLUDED_DIR_ENDS = frozenset([".git", ".app", ".pkg", ".kext"])
EXCLUDED_SHORT_FILE_NAMES = frozenset([".DS_Store"])
MIN_FILE_SIZE: int = 0  # 1M

# Consts
FILE_READ_BLOCK_SIZE = 1 << 20  # 1M
# Flags
PRINT_ACCESS_ERRORS = False
PRINT_SKIPPED = False


class FileIsNotAccessibleException(OSError):
    pass


def wrap_long_filename(filename: str) -> str:
    """
    Only for Windows
    """
    prefix = "\\\\?\\"
    if sys.platform != "win32" or filename.startswith(prefix):
        return filename
    return prefix + filename


def get_file_size(filename: str) -> int:
    try:
        return os.path.getsize(wrap_long_filename(filename))
    except OSError as e:
        if e.errno == errno.EINVAL:
            raise FileIsNotAccessibleException(
                e.errno, "Can't get size: " + e.strerror, filename
            ) from e
        raise


STATS = {"HASHES_CALCULATED": 0}


def calculate_sha1(filename: str) -> str:
    sha1 = hashlib.sha1()
    try:
        with open(
            wrap_long_filename(filename), mode="rb", buffering=FILE_READ_BLOCK_SIZE
        ) as f:
            while True:
                data = f.read(FILE_READ_BLOCK_SIZE)
                if not data:
                    STATS["HASHES_CALCULATED"] += 1
                    return sha1.hexdigest()
                sha1.update(data)
    except (FileNotFoundError, PermissionError) as e:
        raise FileIsNotAccessibleException(
            e.errno, "Can't get hash: " + e.strerror, filename
        ) from e


class Progress:
    UPDATE_INTERVAL_SEC = 2.0
    LastProgressTime = None

    def __init__(self):
        raise NotImplementedError("Static class")

    @classmethod
    def reset_progress(cls) -> None:
        cls.LastProgressTime = 0.0

    @classmethod
    def update_progress(cls) -> None:
        if (time.time() - cls.LastProgressTime) > cls.UPDATE_INTERVAL_SEC:
            cls.LastProgressTime = time.time()
            print("#", end="", file=sys.stderr)
            sys.stderr.flush()

    @classmethod
    def already_updated(cls) -> None:
        cls.LastProgressTime = time.time()


class FileItem(object):
    def __init__(self, fullname: str) -> None:
        self._fullname = fullname
        self._size = get_file_size(self._fullname)
        self._hash = None

    @property
    def fullname(self):
        return self._fullname

    @property
    def path(self):
        return os.path.split(self.fullname)[-1]

    @property
    def size(self):
        return self._size

    def __repr__(self):
        return f"{self.__class__.__name__}({self._fullname!r})"

    def __str__(self):
        return f"{self.size / (1 << 20):.2f} MB '{self._fullname}'"

    def __bool__(self):
        return bool(os.path.isfile(self.fullname) and self.size and self.hash())

    def hash(self):
        if self._hash is None:
            try:
                self._hash = calculate_sha1(self.fullname)
            except FileIsNotAccessibleException as e:
                if PRINT_ACCESS_ERRORS:
                    print(e, file=sys.stderr)
                return None
            Progress.update_progress()
        return self._hash


# end of class FileItem ################################################################################################


def search_file_iter(search_path: str, mask: str) -> Iterable[FileItem]:
    print(
        f"Searching in '{search_path}' for files greater than {MIN_FILE_SIZE / (1 << 20):.3f} MB",
        file=sys.stderr,
    )
    for path, dir_names, short_names in os.walk(search_path):
        if dir_names:
            dirs_to_delete = [
                d
                for d in dir_names
                if (
                    d.startswith(tuple(EXCLUDED_DIR_BEGINS))
                    or d.endswith(tuple(EXCLUDED_DIR_ENDS))
                )
            ]
            for dir_to_d in dirs_to_delete:
                if PRINT_SKIPPED:
                    print(f"Skipping '{os.path.join(path, dir_to_d)}'", file=sys.stderr)
                dir_names.remove(dir_to_d)

        if not short_names:
            continue

        for short_name in short_names:
            if short_name in EXCLUDED_SHORT_FILE_NAMES:
                continue
            fullname = os.path.join(path, short_name)

            if not fnmatch.fnmatch(short_name, mask):
                continue

            if not os.path.isfile(fullname):
                if PRINT_SKIPPED:
                    print(f"Not a file '{fullname}'", file=sys.stderr)
                continue  # skip if link or something
            try:
                file_item = FileItem(fullname)
            except FileIsNotAccessibleException as e:
                if PRINT_ACCESS_ERRORS:
                    print(e, file=sys.stderr)
                continue
            if file_item.size < MIN_FILE_SIZE:
                continue
            yield file_item
        Progress.update_progress()


def duplicates_iter(file_list: Sequence[FileItem]) -> Iterable[Sequence[FileItem]]:
    size_getter = operator.attrgetter("size")
    hash_getter = operator.methodcaller("hash")
    print("Sorting by size...", file=sys.stderr)
    sorted_by_size = sorted(file_list, key=size_getter)
    print("Grouping by size and SHA-1 hash...", file=sys.stderr)
    for _n, same_sized in itertools.groupby(sorted_by_size, key=size_getter):
        same_sized = tuple(same_sized)
        if len(same_sized) == 1:
            # print(f'One size: {same_sized[0]}')
            continue
        one_sized_files_with_hash = tuple(filter(None, same_sized))
        sorted_by_hash = sorted(one_sized_files_with_hash, key=hash_getter)
        for hash_as_key, duplicates in itertools.groupby(
            sorted_by_hash, key=hash_getter
        ):
            duplicates = tuple(duplicates)
            if len(duplicates) > 1 and hash_as_key:
                yield duplicates


def main(*, search_path: str, mask: str, power, verbose_level: int) -> None:
    search_path = search_path[0]
    assert power >= 0
    global PRINT_ACCESS_ERRORS
    global PRINT_SKIPPED
    global MIN_FILE_SIZE
    # Set global flags
    PRINT_ACCESS_ERRORS = verbose_level > 0
    PRINT_SKIPPED = verbose_level > 1
    MIN_FILE_SIZE = 1 << 10 + power

    if sys.platform == "win32":
        search_path = os.sep.join(search_path.split("/"))
        assert (
            os.path.splitdrive(search_path)[0] != search_path
        ), f"Add slash to search path '{search_path + os.sep}'"
    assert os.path.isdir(search_path), f"Not a dir '{search_path}'"

    Progress.reset_progress()
    file_items = []
    for item in search_file_iter(search_path, mask):
        file_items.append(item)
        Progress.update_progress()
    Progress.reset_progress()
    print(f"Searching duplicates in {len(file_items)} filtered files", file=sys.stderr)
    print("Duplicates:", file=sys.stderr)
    print("", file=sys.stderr)
    for group in duplicates_iter(file_items):
        for item in sorted(group, key=lambda i: i.path):
            print(item)
            Progress.already_updated()
        print("")  # End of Group
    if PRINT_ACCESS_ERRORS:
        print(f"HASHES: {STATS['HASHES_CALCULATED']}")


def parse_args() -> dict:
    default_power = 10
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("search_path", nargs=1, help="Path for search")
    parser.add_argument("mask", nargs="?", default="*", help="File mask, default *")
    parser.add_argument(
        "--power",
        "-2",
        type=int,
        default=default_power,
        help=f"Power of 2 for minimum file size in KB, default={default_power}",
    )
    parser.add_argument("--verbose-level", "-v", action="count", default=0)
    parser.add_argument("--version", "-V", action="version", version=__version__)
    kwargs = parser.parse_args()
    return vars(kwargs)


if __name__ == "__main__":
    main(**parse_args())
