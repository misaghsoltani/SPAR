"""Utilities for typed lazy imports with optional caching."""

from __future__ import annotations

from contextlib import contextmanager, suppress
import importlib
import logging
import sys
import threading
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence
    from logging import Logger
    from types import ModuleType
    from typing import TypeAlias

    ImportTarget: TypeAlias = tuple[str, str]
    LazyAttribute: TypeAlias = ModuleType | type | Callable[..., "LazyAttribute"]
    CacheDict: TypeAlias = dict[str, LazyAttribute]
    ModuleCacheDict: TypeAlias = dict[str, ModuleType]
    AttributeGetter: TypeAlias = Callable[[str], LazyAttribute]
    DirectoryLister: TypeAlias = Callable[[], list[str]]
    ImportMapping: TypeAlias = Mapping[str, ImportTarget]
    GenericItem: TypeAlias = type | tuple[type, ...] | str
    LazyImporterKwarg: TypeAlias = str | bool | Sequence[str] | None


LazyAttrT = TypeVar("LazyAttrT")

__all__: list[str] = ["LazyImportError", "LazyImporter", "lazy_import_module"]

logger: Logger = logging.getLogger(__name__)


class CacheInfo(TypedDict):
    cached_attributes: int
    cached_modules: int
    total_attributes: int
    cache_hit_ratio: float
    cache_enabled: bool
    thread_safe: bool
    debug_mode: bool
    module_name: str
    uncached_attributes: list[str]


class CompatibilityInfo(TypedDict):
    supports_type_checking: bool
    supports_star_import: bool
    supports_dir: bool
    supports_getattr: bool
    supports_all: bool
    supports_relative_imports: bool
    thread_safe: bool
    cache_enabled: bool
    debug_mode: bool
    total_imports: int
    cached_imports: int
    module_name: str


class LazyImportError(ImportError):
    """Raised when a lazy import fails."""

    def __init__(self, name: str | None, module_name: str, original_error: Exception) -> None:
        """Initialize lazy import error.

        Args:
            name: The attribute name that failed to import
            module_name: The module where the import failed
            original_error: The original exception that caused the failure
        """
        super().__init__(f"Failed to lazily import '{name}' from '{module_name}': {original_error}")
        self.name: str | None = name
        self.module_name: str = module_name
        self.original_error: Exception = original_error


class LazyImporter(Generic[LazyAttrT]):
    """Stateful helper that exposes ``__getattr__``/``__dir__`` friendly lazy imports."""

    __slots__: tuple[str, ...] = (
        "_all_attrs",
        "_cache",
        "_lock",
        "_module_cache",
        "cache_enabled",
        "debug_mode",
        "imports",
        "module_name",
        "thread_safe",
        "type_checking_imports",
    )

    def __init__(
        self,
        imports: ImportMapping,
        module_name: str,
        *,
        type_checking_imports: Sequence[str] | None = None,
        cache_enabled: bool = True,
        thread_safe: bool = True,
        debug_mode: bool = False,
        validate_imports: bool = True,
    ) -> None:
        """Initialize the stateful helper that exposes ``__getattr__``/``__dir__`` friendly lazy imports.

        Args:
            imports: Dictionary mapping attribute names to (module_name, attr_name) tuples
            module_name: The name of the module using this lazy importer
            type_checking_imports: List of import statements for TYPE_CHECKING blocks
            cache_enabled: Whether to enable import caching
            thread_safe: Whether to use thread-safe operations
            debug_mode: Whether to enable debugging information
            validate_imports: Whether to validate import mappings at initialization
        """
        if not module_name:
            raise ValueError("module_name must be a non-empty string")

        normalized_imports: dict[str, ImportTarget] = {
            name: (module, attribute) for name, (module, attribute) in imports.items()
        }

        self.imports: Mapping[str, ImportTarget] = MappingProxyType(normalized_imports)
        self.module_name: str = module_name
        self.cache_enabled: bool = cache_enabled
        self.thread_safe: bool = thread_safe
        self.debug_mode: bool = debug_mode
        self.type_checking_imports: tuple[str, ...] = tuple(type_checking_imports or ())

        self._cache: dict[str, LazyAttrT] = {}
        self._module_cache: ModuleCacheDict = {}
        self._lock: threading.RLock | None = threading.RLock() if thread_safe else None
        self._all_attrs: tuple[str, ...] = tuple(self.imports.keys())

        if validate_imports:
            self._validate_imports()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _debug_log(self, message: str) -> None:
        if self.debug_mode:
            logger.debug(f"[LazyImporter:{self.module_name}] {message}")

    def _validate_imports(self) -> None:
        """Validate import mappings for consistency.

        Raises:
            ValueError: If any import mapping is invalid
        """
        for public_name, (module_name, attribute_name) in self.imports.items():
            if not public_name.strip():
                raise ValueError(f"Invalid attribute name: {public_name!r}")
            if not module_name.strip():
                raise ValueError(f"Invalid module name for '{public_name}': {module_name!r}")
            if not attribute_name.strip():
                raise ValueError(f"Invalid attribute target for '{public_name}': {attribute_name!r}")
            if not public_name.isidentifier():
                raise ValueError(f"Attribute name '{public_name}' must be a valid identifier")
            if not attribute_name.isidentifier():
                raise ValueError(f"Target attribute '{attribute_name}' for '{public_name}' must be a valid identifier")

    def _get_module(self, module_name: str) -> ModuleType:
        """Get or import a module with internal caching.

        Args:
            module_name: The fully qualified module name to import

        Returns:
            The imported module
        """
        cached: ModuleType | None = self._module_cache.get(module_name)
        if cached is not None:
            return cached

        if module_name.startswith("."):
            anchor: str = self._get_package_anchor()
            self._debug_log(f"import_module('{module_name}', package='{anchor}') [relative]")
            module: ModuleType = importlib.import_module(module_name, package=anchor)
        else:
            self._debug_log(f"import_module('{module_name}')")
            module = importlib.import_module(module_name)
        self._module_cache[module_name] = module
        return module

    def _get_package_anchor(self) -> str:
        """Resolve the package anchor for relative imports with minimal overhead."""
        module: ModuleType | None = sys.modules.get(self.module_name)
        pkg: str | None = getattr(module, "__package__", None) if module is not None else None
        if isinstance(pkg, str) and pkg:
            return pkg
        parent: str = self.module_name.rpartition(".")[0]
        return parent or self.module_name

    def _import_attribute(self, name: str) -> LazyAttrT:
        """Import an attribute using the lazy import mapping.

        Args:
            name: The attribute name to import

        Returns:
            The imported attribute

        Raises:
            AttributeError: If the attribute is not found in the import mapping
            LazyImportError: If the import fails
        """
        module_name: str
        attribute_name: str
        available: str
        try:
            module_name, attribute_name = self.imports[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.imports))
            raise AttributeError(
                f"'{self.module_name}' has no attribute '{name}'. Available attributes: {available}"
            ) from exc

        module: ModuleType
        try:
            module = self._get_module(module_name)
            return getattr(module, attribute_name)
        except (ImportError, AttributeError) as error:
            raise LazyImportError(name, module_name, error) from error

    def _load_and_cache(self, name: str) -> LazyAttrT:
        value: LazyAttrT = self._import_attribute(name)
        module: ModuleType | None
        if self.cache_enabled:
            self._cache[name] = value
            module = sys.modules.get(self.module_name)
            if module is not None:
                with suppress(AttributeError, TypeError):
                    setattr(module, name, value)
        return value

    # ------------------------------------------------------------------
    # Public API used by modules
    # ------------------------------------------------------------------
    def get_attr(self, name: str) -> LazyAttrT:
        """Load an attribute through the ``__getattr__`` protocol.

        Args:
            name: Attribute name to import.

        Returns:
            The imported attribute, optionally retrieved from the cache.
        """
        if self.cache_enabled and name in self._cache:
            self._debug_log(f"cache hit: {name}")
            return self._cache[name]

        if self.thread_safe and self._lock is not None:
            with self._lock:
                if self.cache_enabled and name in self._cache:
                    return self._cache[name]
                return self._load_and_cache(name)

        return self._load_and_cache(name)

    def get_all(self) -> list[str]:
        """Get the list of all available attribute names.

        Returns:
            List of all attribute names available for lazy import
        """
        return list(self._all_attrs)

    def get_dir(self) -> list[str]:
        """Get the list of all available attribute names.

        Returns:
            List of all attribute names available for lazy import
        """
        return list(self._all_attrs)

    def get_available_attributes(self) -> list[str]:
        """Get a sorted list of all available attribute names.

        Returns:
            List of all attribute names available for lazy import
        """
        return sorted(self.imports)

    def has_attribute(self, name: str) -> bool:
        """Check if an attribute is available for lazy import.

        Args:
            name: The attribute name to check

        Returns:
            True if the attribute can be imported, False otherwise
        """
        return name in self.imports

    def get_import_info(self, name: str) -> tuple[str, str] | None:
        """Get import information for an attribute.

        Args:
            name: The attribute name to get information for

        Returns:
            Tuple of (module_name, attribute_name) or None if not found
        """
        return self.imports.get(name)

    def is_cached(self, name: str) -> bool:
        """Check if an attribute is currently cached.

        Args:
            name: The attribute name to check

        Returns:
            True if the attribute is cached, False otherwise
        """
        return name in self._cache

    def preload_attributes(self, *names: str) -> None:
        """Load selected attributes before their first access.

        This is useful when a caller wants import failures during setup.

        Args:
            *names: Names of attributes to preload

        Raises:
            AttributeError: If any of the specified attributes don't exist
        """
        available: str
        for name in names:
            if name not in self.imports:
                available = ", ".join(sorted(self.imports))
                raise AttributeError(f"Cannot preload unknown attribute '{name}'. Available attributes: {available}")
            self.get_attr(name)

    def _clear(self) -> None:
        """Internal method to clear caches."""
        module: ModuleType | None = sys.modules.get(self.module_name)
        if module is not None:
            for attr_name in list(self._cache):
                if hasattr(module, attr_name):
                    delattr(module, attr_name)
        self._cache.clear()
        self._module_cache.clear()

    def clear_cache(self) -> None:
        """Clear cached values from the importer and module namespace."""
        if self.thread_safe and self._lock is not None:
            with self._lock:
                self._clear()
        else:
            self._clear()

    def get_cache_info(self) -> CacheInfo:
        """Get information about the current cache state.

        Returns:
            Dictionary containing cache statistics and information
        """
        total_attrs: int = len(self.imports)
        cached_attrs: int = len(self._cache)
        return {
            "cached_attributes": cached_attrs,
            "cached_modules": len(self._module_cache),
            "total_attributes": total_attrs,
            "cache_hit_ratio": cached_attrs / total_attrs if total_attrs else 0.0,
            "cache_enabled": self.cache_enabled,
            "thread_safe": self.thread_safe,
            "debug_mode": self.debug_mode,
            "module_name": self.module_name,
            "uncached_attributes": [name for name in self.imports if name not in self._cache],
        }

    def generate_type_checking_block(self) -> str:
        """Generate a TYPE_CHECKING block for the imports.

        Returns:
            Source text for the corresponding ``TYPE_CHECKING`` block.
        """
        if not self.type_checking_imports:
            return ""

        lines: list[str] = ["if TYPE_CHECKING:"]
        lines.extend(f"    {statement}" for statement in self.type_checking_imports)
        return "\n".join(lines)

    def test_import(self, name: str) -> bool:
        """Test an attribute import without changing the cache.

        Args:
            name: Attribute name to test.

        Returns:
            ``True`` when the attribute can be imported.
        """
        target: tuple[str, str] | None = self.imports.get(name)
        if target is None:
            return False

        module_name: str
        attribute_name: str
        module_name, attribute_name = target
        module: ModuleType
        try:
            if module_name.startswith("."):
                # Resolve relative import against the calling module's package
                module = importlib.import_module(module_name, package=self._get_package_anchor())
            else:
                module = importlib.import_module(module_name)
        except ImportError:
            return False
        return hasattr(module, attribute_name)

    @contextmanager
    def no_cache(self) -> Generator[None, None, None]:
        """Context manager to temporarily disable caching.

        Tests use this context when they need uncached imports.
        Note: This creates a temporary copy of the importer with caching disabled.

        Example:
            ```python
            with lazy_importer.no_cache():
                # This import will not be cached
                attr = lazy_importer.get_attr("my_attribute")
            ```

        Yields:
            None
        """
        # Create a temporary state where we bypass caching
        original_cache: dict[str, LazyAttrT] = self._cache
        original_module_cache: dict[str, ModuleType] = self._module_cache
        self._cache = {}
        self._module_cache = {}
        try:
            yield
        finally:
            self._cache = original_cache
            self._module_cache = original_module_cache

    def validate_all_imports(self) -> dict[str, bool]:
        """Validate all import mappings without caching.

        Returns:
            Dictionary mapping attribute names to their import success status
        """
        return {name: self.test_import(name) for name in self.imports}

    def get_module_attributes(self) -> dict[str, ImportTarget]:
        """Get a mapping of all available attributes and their import information.

        The mapping is useful for IDE introspection and debugging.

        Returns:
            Dictionary mapping attribute names to (module_name, attribute_name) tuples
        """
        return dict(self.imports)

    def get_type_stubs(self) -> dict[str, str]:
        """Generate import statements for type stubs.

        Returns:
            Mapping from public attribute names to import statements.
        """
        return {
            public_name: f"from {module_name} import {attribute_name} as {public_name}"
            for public_name, (module_name, attribute_name) in self.imports.items()
        }

    @staticmethod
    def supports_star_import() -> bool:
        """Report support for star imports.

        Returns:
            ``True`` because the importer defines ``__all__``.
        """
        # Star imports are supported since we implement __all__
        return True

    def get_import_compatibility_info(self) -> CompatibilityInfo:
        """Describe the import hooks exposed to Python tooling.

        Returns:
            Importer capabilities and active cache settings.
        """
        return {
            "supports_type_checking": True,
            "supports_star_import": True,
            "supports_dir": True,
            "supports_getattr": True,
            "supports_all": True,
            "supports_relative_imports": True,
            "thread_safe": self.thread_safe,
            "cache_enabled": self.cache_enabled,
            "debug_mode": self.debug_mode,
            "total_imports": len(self.imports),
            "cached_imports": len(self._cache),
            "module_name": self.module_name,
        }


def lazy_import_module(
    imports: ImportMapping,
    module_name: str,
    *,
    type_checking_imports: Sequence[str] | None = None,
    cache_enabled: bool = True,
    thread_safe: bool = True,
    debug_mode: bool = False,
    validate_imports: bool = True,
    **kwargs: LazyImporterKwarg,
) -> tuple[AttributeGetter, DirectoryLister, list[str]]:
    """Create the module hooks for a lazy import interface.

    This wrapper creates a :class:`LazyImporter` and returns the three module
    hooks used by ``__getattr__``, ``__dir__``, and ``__all__``.

    Args:
        imports: Mapping from attribute names to module and attribute pairs.
        module_name: Name of the module that exposes the lazy imports.
        type_checking_imports: Import statements for ``TYPE_CHECKING`` blocks.
        cache_enabled: Whether to cache imported attributes.
        thread_safe: Whether to protect imports with a lock.
        debug_mode: Whether to include debugging information.
        validate_imports: Whether to validate mappings during initialization.
        **kwargs: Additional arguments passed to :class:`LazyImporter`.

    Returns:
        The ``__getattr__`` hook, ``__dir__`` hook, and ``__all__`` names.

    Example:
        ```python
        from __future__ import annotations

        from typing import TYPE_CHECKING

        from spar.utils.import_utils.lazy_importer import lazy_import_module

        if TYPE_CHECKING:
            from my_package.my_module import my_func
            form spar.utils.import_utils.lazy_importer import ImportMapping, AttributeGetter, DirectoryLister


        IMPORTS: ImportMapping = {"my_func": ("my_package.my_module", "my_func")}
        __getattr__: AttributeGetter
        __dir__: DirectoryLister
        __all__: list[str]
        __getattr__, __dir__, __all__ = lazy_import_module(
            IMPORTS, __name__, type_checking_imports=["from my_package.my_module import my_func"]
        )
        ```
    """
    importer: LazyImporter[LazyAttribute] = LazyImporter(
        imports=imports,
        module_name=module_name,
        type_checking_imports=type_checking_imports,
        cache_enabled=cache_enabled,
        thread_safe=thread_safe,
        debug_mode=debug_mode,
        validate_imports=validate_imports,
        **kwargs,
    )
    return importer.get_attr, importer.get_dir, importer.get_all()
