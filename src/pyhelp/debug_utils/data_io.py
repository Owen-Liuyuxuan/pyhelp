from __future__ import annotations

import ast
import inspect
import json
import keyword
import linecache
import os
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


SupportedObject = Any
TargetType = Literal["original", "numpy", "torch"]


class DataIOError(RuntimeError):
    """Custom exception for unified numpy / torch IO errors."""


def _is_numpy_array(obj: Any) -> bool:
    return isinstance(obj, np.ndarray)


def _is_torch_tensor(obj: Any) -> bool:
    return _TORCH_AVAILABLE and isinstance(obj, torch.Tensor)


def _normalize_suffix(path: str | Path, default_suffix: str = ".npz") -> Path:
    p = Path(path)
    if p.suffix == "":
        p = p.with_suffix(default_suffix)
    return p


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sanitize_filename_stem(name: str, fallback: str = "data") -> str:
    """
    Convert an inferred expression name into a safe file stem.

    Rules:
    - Replace path separators with "_"
    - Keep only [0-9A-Za-z._-]
    - Collapse repeated underscores
    - Strip leading/trailing separators
    - Avoid empty names
    - Avoid common Windows reserved names
    - Avoid Python keywords for readability
    """
    if not name:
        return fallback

    name = name.replace(os.sep, "_")
    if os.altsep:
        name = name.replace(os.altsep, "_")

    name = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("._-")

    if not name:
        name = fallback

    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
    if name.lower() in reserved:
        name = f"{fallback}_{name}"

    if keyword.iskeyword(name):
        name = f"{name}_data"

    return name


def _subscript_to_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None

    if isinstance(node, ast.Constant):
        return str(node.value)

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Tuple):
        parts: list[str] = []
        for elt in node.elts:
            part = _subscript_to_name(elt)
            if part is None:
                return None
            parts.append(part)
        return "_".join(parts)

    if isinstance(node, ast.Slice):
        parts = [
            _subscript_to_name(node.lower) if node.lower else "",
            _subscript_to_name(node.upper) if node.upper else "",
            _subscript_to_name(node.step) if node.step else "",
        ]
        suffix = "_".join(p for p in parts if p)
        return "slice" if not suffix else f"slice_{suffix}"

    return None


def _expr_to_name(expr: ast.AST) -> str | None:
    """
    Convert a restricted AST expression into a readable variable-like name.

    Allowed:
    - x
    - batch.image
    - data["points"]
    - data[0]
    - sample["cam"].image

    Rejected:
    - function calls
    - binary operators
    - comprehensions
    - arbitrary expressions
    """
    if isinstance(expr, ast.Name):
        return expr.id

    if isinstance(expr, ast.Attribute):
        base = _expr_to_name(expr.value)
        if base is None:
            return None
        return f"{base}_{expr.attr}"

    if isinstance(expr, ast.Subscript):
        base = _expr_to_name(expr.value)
        if base is None:
            return None

        slice_name = _subscript_to_name(expr.slice)
        if slice_name is None:
            return base

        return f"{base}_{slice_name}"

    return None


def _find_call_argument_expr_source(
    frame: Any,
    func_name: str,
    arg_index: int = 0,
    max_lines: int = 30,
) -> str | None:
    """
    Best-effort extraction of the source expression for a positional argument
    at the call site of `func_name`.

    This uses runtime frame introspection to locate the call-site source,
    then AST parsing to recover the argument expression.
    """
    try:
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno

        source_lines = linecache.getlines(filename)
        if not source_lines:
            return None

        start_idx = max(0, lineno - 1)
        end_idx = min(len(source_lines), start_idx + max_lines)
        window = "".join(source_lines[start_idx:end_idx])

        tree = ast.parse(window)
        candidate_calls: list[ast.Call] = []

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> Any:
                called_name = None
                if isinstance(node.func, ast.Name):
                    called_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    called_name = node.func.attr

                if called_name == func_name:
                    candidate_calls.append(node)

                self.generic_visit(node)

        Visitor().visit(tree)

        if not candidate_calls:
            return None

        call = candidate_calls[0]
        if len(call.args) <= arg_index:
            return None

        expr = call.args[arg_index]
        try:
            return ast.unparse(expr)
        except Exception:
            return None

    except Exception:
        return None


def _guess_variable_name_from_ast(
    func_name: str,
    arg_index: int = 0,
    frame_depth: int = 3,
) -> str | None:
    """
    Try to infer the calling argument expression via AST, then map it to a safe stem.

    Parameters
    ----------
    func_name:
        The public API name expected at the call site, e.g. "save_data".
    arg_index:
        Positional argument index to extract.
    frame_depth:
        How many `.f_back` hops to move up from the current frame to reach the
        original user call site. This is best-effort and depends on wrapper depth.
    """
    try:
        frame = inspect.currentframe()
        if frame is None:
            return None

        caller_frame = frame
        for _ in range(frame_depth):
            if caller_frame.f_back is None:
                return None
            caller_frame = caller_frame.f_back

        expr_source = _find_call_argument_expr_source(
            frame=caller_frame,
            func_name=func_name,
            arg_index=arg_index,
        )
        if not expr_source:
            return None

        expr_ast = ast.parse(expr_source, mode="eval").body
        name = _expr_to_name(expr_ast)
        if not name:
            return None

        return _sanitize_filename_stem(name)

    except Exception:
        return None


def _guess_variable_name_from_identity(obj: Any, frame_depth: int = 3) -> str | None:
    """
    Fallback variable name guessing using caller locals identity matching.

    This is less accurate than AST extraction but works when source recovery fails.
    """
    try:
        frame = inspect.currentframe()
        if frame is None:
            return None

        caller_frame = frame
        for _ in range(frame_depth):
            if caller_frame.f_back is None:
                return None
            caller_frame = caller_frame.f_back

        candidates: list[str] = []
        for var_name, var_value in caller_frame.f_locals.items():
            if var_value is obj:
                candidates.append(var_name)

        if not candidates:
            return None

        candidates.sort(key=lambda x: (len(x), x))
        return _sanitize_filename_stem(candidates[0])

    except Exception:
        return None


def _guess_default_name(
    obj: Any,
    *,
    caller_func_name: str,
    arg_index: int = 0,
) -> str | None:
    """
    Guess a default save name using:
    1. AST call-site argument extraction
    2. locals() identity matching fallback
    """
    guessed = _guess_variable_name_from_ast(
        func_name=caller_func_name,
        arg_index=arg_index,
        frame_depth=4,
    )
    if guessed:
        return guessed

    guessed = _guess_variable_name_from_identity(obj, frame_depth=4)
    if guessed:
        return guessed

    return None


def _resolve_save_path(
    obj: Any,
    path: str | Path | None = None,
    name: str | None = None,
    default_suffix: str = ".npz",
    caller_func_name: str = "save_data",
) -> Path:
    if path is not None:
        return _normalize_suffix(path, default_suffix=default_suffix)

    if name:
        return _normalize_suffix(_sanitize_filename_stem(name), default_suffix=default_suffix)

    guessed = _guess_default_name(obj, caller_func_name=caller_func_name, arg_index=0)
    if guessed:
        return _normalize_suffix(guessed, default_suffix=default_suffix)

    raise DataIOError(
        "Unable to infer a default file name from the call-site argument. "
        "Please provide `path=` or `name=` explicitly."
    )


def _torch_dtype_to_string(dtype: Any) -> str | None:
    if not _TORCH_AVAILABLE or dtype is None:
        return None
    return str(dtype).replace("torch.", "")


def _string_to_torch_dtype(dtype_str: str | None) -> Any:
    if not _TORCH_AVAILABLE or dtype_str is None:
        return None
    if not hasattr(torch, dtype_str):
        raise DataIOError(f"Unsupported torch dtype string: {dtype_str}")
    return getattr(torch, dtype_str)


def _numpy_dtype_to_string(dtype: np.dtype) -> str:
    return np.dtype(dtype).str


def _string_to_numpy_dtype(dtype_str: str) -> np.dtype:
    return np.dtype(dtype_str)


def _extract_metadata_and_numpy_payload(obj: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Convert the input object into:
    - a numpy payload suitable for np.savez / np.savez_compressed
    - a JSON-serializable metadata dictionary
    """
    if _is_numpy_array(obj):
        payload = obj
        meta = {
            "format_version": 1,
            "original_type": "numpy",
            "numpy_dtype": _numpy_dtype_to_string(obj.dtype),
            "shape": list(obj.shape),
        }
        return payload, meta

    if _is_torch_tensor(obj):
        tensor = obj.detach()
        torch_dtype = _torch_dtype_to_string(obj.dtype)
        if torch_dtype == "bfloat16":
            tensor = tensor.to(torch.float32)
        payload = tensor.cpu().numpy()
        meta = {
            "format_version": 1,
            "original_type": "torch",
            "torch_dtype": torch_dtype,
            "numpy_dtype": _numpy_dtype_to_string(payload.dtype),
            "shape": list(tensor.shape),
            "device": str(obj.device),
            "requires_grad": bool(obj.requires_grad),
        }
        return payload, meta

    raise DataIOError(
        f"Unsupported object type: {type(obj)!r}. "
        "Only numpy.ndarray and torch.Tensor are supported."
    )


def _resolve_original_torch_device(
    saved_device: str,
    device: str | Any | None = None,
    fallback_to_cpu_if_unavailable: bool = False,
) -> str | Any:
    """
    Resolve target torch device when reconstructing an originally saved torch tensor.
    """
    if device is not None:
        return device

    if not _TORCH_AVAILABLE:
        raise DataIOError("PyTorch is not available.")

    if saved_device.startswith("cuda"):
        if not torch.cuda.is_available():
            if fallback_to_cpu_if_unavailable:
                return "cpu"
            raise DataIOError(
                f"Saved tensor device is {saved_device!r}, but CUDA is not available. "
                "Pass `device='cpu'` or set `fallback_to_cpu_if_unavailable=True`."
            )

    return saved_device


def _reconstruct_as_original(
    payload: np.ndarray,
    meta: dict[str, Any],
    *,
    device: str | Any | None = None,
    fallback_to_cpu_if_unavailable: bool = False,
) -> Any:
    original_type = meta.get("original_type")

    if original_type == "numpy":
        return payload.astype(_string_to_numpy_dtype(meta["numpy_dtype"]), copy=False)

    if original_type == "torch":
        if not _TORCH_AVAILABLE:
            raise DataIOError("PyTorch is not available, cannot reconstruct torch.Tensor.")

        torch_dtype = _string_to_torch_dtype(meta.get("torch_dtype"))
        saved_device = meta.get("device", "cpu")
        target_device = _resolve_original_torch_device(
            saved_device=saved_device,
            device=device,
            fallback_to_cpu_if_unavailable=fallback_to_cpu_if_unavailable,
        )
        requires_grad = bool(meta.get("requires_grad", False))

        tensor = torch.from_numpy(payload)
        if torch_dtype is not None and tensor.dtype != torch_dtype:
            tensor = tensor.to(dtype=torch_dtype)
        tensor = tensor.to(device=target_device)
        tensor.requires_grad_(requires_grad)
        return tensor

    raise DataIOError(f"Unknown original_type in metadata: {original_type}")


def _convert_loaded_object(
    payload: np.ndarray,
    meta: dict[str, Any],
    as_type: TargetType = "original",
    dtype: Any | None = None,
    device: str | Any | None = None,
    fallback_to_cpu_if_unavailable: bool = False,
) -> Any:
    if as_type == "original":
        obj = _reconstruct_as_original(
            payload=payload,
            meta=meta,
            device=device,
            fallback_to_cpu_if_unavailable=fallback_to_cpu_if_unavailable,
        )

        if _is_numpy_array(obj):
            if dtype is not None:
                obj = obj.astype(dtype, copy=False)
            return obj

        if _is_torch_tensor(obj):
            if dtype is not None or device is not None:
                obj = obj.to(
                    dtype=dtype if dtype is not None else obj.dtype,
                    device=device if device is not None else obj.device,
                )
            return obj

        return obj

    if as_type == "numpy":
        arr = payload
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    if as_type == "torch":
        if not _TORCH_AVAILABLE:
            raise DataIOError("PyTorch is not available, cannot load as torch.Tensor.")

        target_device = device if device is not None else "cpu"
        if isinstance(target_device, str) and target_device.startswith("cuda"):
            if not torch.cuda.is_available():
                if fallback_to_cpu_if_unavailable:
                    target_device = "cpu"
                else:
                    raise DataIOError(
                        f"Requested target device is {target_device!r}, but CUDA is not available. "
                        "Pass `device='cpu'` or set `fallback_to_cpu_if_unavailable=True`."
                    )

        tensor = torch.from_numpy(payload)
        if dtype is not None or device is not None:
            tensor = tensor.to(
                dtype=dtype if dtype is not None else tensor.dtype,
                device=target_device,
            )
        return tensor

    raise DataIOError(f"Unsupported as_type: {as_type}")


def _save_impl(
    obj: Any,
    *,
    path: str | Path | None = None,
    name: str | None = None,
    compressed: bool = True,
    overwrite: bool = True,
    caller_func_name: str = "save_data",
) -> Path:
    save_path = _resolve_save_path(
        obj=obj,
        path=path,
        name=name,
        default_suffix=".npz",
        caller_func_name=caller_func_name,
    )
    save_path = save_path.expanduser().resolve()
    _ensure_parent_dir(save_path)

    if save_path.exists() and not overwrite:
        raise DataIOError(f"File already exists: {save_path}")

    payload, meta = _extract_metadata_and_numpy_payload(obj)
    meta_json = json.dumps(meta, ensure_ascii=False)

    save_func = np.savez_compressed if compressed else np.savez
    save_func(save_path, data=payload, meta=np.array(meta_json, dtype=np.unicode_))

    return save_path


def save_data(
    obj: Any,
    path: str | Path | None = None,
    *,
    name: str | None = None,
    compressed: bool = True,
    overwrite: bool = True,
) -> Path:
    """
    Save one numpy.ndarray or torch.Tensor into one .npz file with metadata.

    Parameters
    ----------
    obj:
        The object to save. Supported:
        - numpy.ndarray
        - torch.Tensor
    path:
        Output file path. If None, try to infer from `name` or call-site variable.
    name:
        Optional fallback file stem when `path` is None.
    compressed:
        Whether to use np.savez_compressed.
    overwrite:
        Whether to overwrite an existing file.

    Returns
    -------
    Path
        The final saved path.
    """
    return _save_impl(
        obj=obj,
        path=path,
        name=name,
        compressed=compressed,
        overwrite=overwrite,
        caller_func_name="save_data",
    )


def load_data(
    path: str | Path,
    *,
    as_type: TargetType = "original",
    dtype: Any | None = None,
    device: str | Any | None = None,
    fallback_to_cpu_if_unavailable: bool = False,
) -> Any:
    """
    Load one object from a .npz file.

    Parameters
    ----------
    path:
        Path to the saved file.
    as_type:
        - "original": restore as original saved type
        - "numpy": always return numpy.ndarray
        - "torch": always return torch.Tensor
    dtype:
        Optional target dtype during loading.
    device:
        Optional target device for torch output or original torch reconstruction.
    fallback_to_cpu_if_unavailable:
        If True, fall back to CPU when the saved/requested CUDA device is unavailable.

    Returns
    -------
    Any
        The loaded object.
    """
    load_path = Path(path).expanduser().resolve()
    if not load_path.exists():
        raise DataIOError(f"File not found: {load_path}")

    with np.load(load_path, allow_pickle=False) as npz_file:
        if "data" not in npz_file or "meta" not in npz_file:
            raise DataIOError(
                f"Invalid file format: {load_path}. Expected keys: 'data' and 'meta'."
            )

        payload = npz_file["data"]
        meta_raw = npz_file["meta"]

        if isinstance(meta_raw, np.ndarray):
            if meta_raw.ndim == 0:
                meta_json = str(meta_raw.item())
            else:
                raise DataIOError("Metadata format is invalid: expected scalar string array.")
        else:
            meta_json = str(meta_raw)

        meta = json.loads(meta_json)

    return _convert_loaded_object(
        payload=payload,
        meta=meta,
        as_type=as_type,
        dtype=dtype,
        device=device,
        fallback_to_cpu_if_unavailable=fallback_to_cpu_if_unavailable,
    )


def save_array(
    array: np.ndarray,
    path: str | Path | None = None,
    *,
    name: str | None = None,
    compressed: bool = True,
    overwrite: bool = True,
) -> Path:
    """
    Save a numpy.ndarray into one .npz file with metadata.
    """
    if not _is_numpy_array(array):
        raise DataIOError(f"Expected numpy.ndarray, got {type(array)!r}")

    return _save_impl(
        obj=array,
        path=path,
        name=name,
        compressed=compressed,
        overwrite=overwrite,
        caller_func_name="save_array",
    )


def load_array(
    path: str | Path,
    *,
    dtype: Any | None = None,
) -> np.ndarray:
    """
    Load a file and always return a numpy.ndarray.
    """
    obj = load_data(path, as_type="numpy", dtype=dtype)
    if not isinstance(obj, np.ndarray):
        raise DataIOError(f"Internal error: expected numpy.ndarray, got {type(obj)!r}")
    return obj


def save_tensor(
    tensor: Any,
    path: str | Path | None = None,
    *,
    name: str | None = None,
    compressed: bool = True,
    overwrite: bool = True,
) -> Path:
    """
    Save a torch.Tensor into one .npz file with metadata.
    """
    if not _is_torch_tensor(tensor):
        raise DataIOError(f"Expected torch.Tensor, got {type(tensor)!r}")

    return _save_impl(
        obj=tensor,
        path=path,
        name=name,
        compressed=compressed,
        overwrite=overwrite,
        caller_func_name="save_tensor",
    )


def load_tensor(
    path: str | Path,
    *,
    dtype: Any | None = None,
    device: str | Any | None = None,
    fallback_to_cpu_if_unavailable: bool = False,
) -> Any:
    """
    Load a file and always return a torch.Tensor.
    """
    if not _TORCH_AVAILABLE:
        raise DataIOError("PyTorch is not available, cannot load tensor.")

    obj = load_data(
        path,
        as_type="torch",
        dtype=dtype,
        device=device,
        fallback_to_cpu_if_unavailable=fallback_to_cpu_if_unavailable,
    )
    if not _is_torch_tensor(obj):
        raise DataIOError(f"Internal error: expected torch.Tensor, got {type(obj)!r}")
    return obj


def peek_data_metadata(path: str | Path) -> dict[str, Any]:
    """
    Read metadata only from the saved file without reconstructing the object.
    """
    load_path = Path(path).expanduser().resolve()
    if not load_path.exists():
        raise DataIOError(f"File not found: {load_path}")

    with np.load(load_path, allow_pickle=False) as npz_file:
        if "meta" not in npz_file:
            raise DataIOError(f"Invalid file format: {load_path}. Missing key 'meta'.")

        meta_raw = npz_file["meta"]
        if isinstance(meta_raw, np.ndarray):
            if meta_raw.ndim == 0:
                meta_json = str(meta_raw.item())
            else:
                raise DataIOError("Metadata format is invalid: expected scalar string array.")
        else:
            meta_json = str(meta_raw)

    return json.loads(meta_json)


__all__ = [
    "DataIOError",
    "save_data",
    "load_data",
    "save_array",
    "load_array",
    "save_tensor",
    "load_tensor",
    "peek_data_metadata",
]
