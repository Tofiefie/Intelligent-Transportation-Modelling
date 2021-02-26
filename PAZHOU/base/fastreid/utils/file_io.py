#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""file_io.py
"""
import errno
import logging
import os
import shutil
from collections import OrderedDict
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    List,
    MutableMapping,
    Optional,
    Union,
)

__all__ = ["PathManager", "get_cache_dir"]


def get_cache_dir(cache_dir=None) -> str:
    """
    Returns a default directory to cache static files
    (usually downloaded from Internet), if None is provided.
    Args:
        cache_dir (None or str): if not None, will be returned as is.
            If None, returns the default cache directory as:
        1) $FVCORE_CACHE, if set
        2) otherwise ~/.torch/fvcore_cache
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(
            os.getenv("FVCORE_CACHE", "~/.torch/fvcore_cache")
        )
    return cache_dir


class PathHandler:
    """
    PathHandler is a base class that defines common I/O functionality for a URI
    protocol. It routes I/O for a generic URI which may look like "protocol://*"
    or a canonical filepath "/foo/bar/baz".
    """

    _strict_kwargs_check = True

    def _check_kwargs(self, kwargs: Dict[str, Any]):
        """
        Checks if the given arguments are empty. Throws a ValueError if strict
        kwargs checking is enabled and args are non-empty. If strict kwargs
        checking is disabled, only a warning is logged.
        Args:
            kwargs (Dict[str, Any])
        """
        if self._strict_kwargs_check:
            if len(kwargs) > 0:
                raise ValueError("Unused arguments: {}".format(kwargs))
        else:
            logger = logging.getLogger(__name__)
            for k, v in kwargs.items():
                logger.warning(
                    "[PathManager] {}={} argument ignored".format(k, v)
                )

    def _get_supported_prefixes(self):
        """
        Returns:
            List[str]: the list of URI prefixes this PathHandler can support
        """
        raise NotImplementedError()

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.
        If URI points to a remote resource, this function may download and cache
        the resource to local disk. In this case, this function is meant to be
        used with read-only resources.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            local_path (str): a file path which exists on the local file system
        """
        raise NotImplementedError()

    def _open(
            self, path: str, mode: str = "r", buffering: int = -1, **kwargs: Any
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI, similar to the built-in `open`.
        Args:
            