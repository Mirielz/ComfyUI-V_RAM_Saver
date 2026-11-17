import ctypes
import gc
import logging
import os
import re
import torch
import comfy.model_management as mm
import comfy.model_patcher
from collections.abc import Iterable
from server import PromptServer

class AnyType(str):
    """Wildcard socket type for ComfyUI: compares equal to every type so any output can connect to it."""
    def __eq__(self, other: object) -> bool:
        """
        Always reports equality so ComfyUI accepts any connection.

        Args:
            other: object being compared (ignored)

        Returns:
            always True
        """
        return True

    def __ne__(self, other: object) -> bool:
        """
        Always reports inequality as False (mirror of __eq__).

        Args:
            other: object being compared (ignored)

        Returns:
            always False
        """
        return False

    def __hash__(self) -> int:
        """
        Hashes the underlying string so the type stays usable as a dict key.

        Returns:
            hash of the wildcard string
        """
        return hash(str(self))

any_type = AnyType('*')

if os.name == 'nt':
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)  # enable ANSI/VT processing

_COLORS = {'red': '31', 'green': '32', 'yellow': '33', 'magenta': '35', 'cyan': '36', 'gray': '90'}

def _c(text: str, color: str, bold: bool = False) -> str:
    """
    Wraps text in ANSI color codes for console output; returns plain text if NO_COLOR is set or the color is unknown.

    Args:
        text: text to colorize (converted with str())
        color: key of _COLORS ('red', 'green', 'yellow', 'magenta', 'cyan', 'gray')
        bold: apply bold style

    Returns:
        colorized string, or plain string when NO_COLOR is set or color is not in _COLORS
    """
    code = _COLORS.get(color)
    if os.environ.get('NO_COLOR') or code is None:
        return str(text)
    
    return f"\033[{'1;' if bold else ''}{code}m{text}\033[0m"

def _empty_module(model: torch.nn.Module) -> tuple[int, int]:
    """
    Replaces every parameter/buffer storage of the model with an empty tensor (no .to(), no copy), freeing its memory.
    Tensors that cannot be emptied are set to None.

    Args:
        model: module whose parameters and buffers are emptied

    Returns:
        vram_bytes: total bytes emptied on non-CPU, non-meta devices (actually freed only if nothing else references the storage)
        ram_bytes: total bytes emptied on CPU (same caveat)
    """
    vram = ram = 0
    seen = set()
    for module in model.modules():
        for store in (module._parameters, module._buffers):
            for name, t in list(store.items()):
                if t is None or id(t) in seen:
                    continue
                seen.add(id(t))
                try:
                    dev = t.device.type
                    nbytes = 0 if dev == 'meta' else t.numel() * t.element_size()
                except Exception:
                    dev, nbytes = 'cpu', 0
                try:
                    t.data = torch.empty(0, dtype = t.dtype)
                except Exception as e:
                    logging.warning(f'V_RAM_Saver: cannot empty {type(module).__name__}.{name} ({type(t).__name__}): {e}. Set to None.')
                    store[name] = None
                if dev == 'cpu':
                    ram += nbytes
                elif dev != 'meta':
                    vram += nbytes
    
    return vram, ram

def _kind(obj: object) -> str:
    """
    Short label of a routed object for status messages.

    Args:
        obj: object connected to wipe_model

    Returns:
        'MODEL' for a ModelPatcher, otherwise the class name (CLIP, VAE, ...)
    """
    if isinstance(obj, comfy.model_patcher.ModelPatcher):
        return 'MODEL'
    
    return type(obj).__name__  # CLIP, VAE, ...

def _note_color(note: str) -> str:
    """
    Colors a status note by its content (red: failed/aborted, yellow: skipped, green: offloaded, gray: other).

    Args:
        note: plain status note

    Returns:
        colorized note for console output
    """
    if 'FAILED' in note or 'aborted' in note:
        return _c(note, 'red')
    if 'skipped' in note:
        return _c(note, 'yellow')
    if note.startswith('offloaded'):
        return _c(note, 'green')
    return _c(note, 'gray')

def _resolve_roots(obj: object) -> list[comfy.model_patcher.ModelPatcher]:
    """
    Collects the unique ModelPatchers behind a routed object (the object itself, its .patcher, and any returned by get_models()).

    Args:
        obj: routed object (MODEL, CLIP, VAE, ...)

    Returns:
        list of unique ModelPatchers; empty if the type is unsupported
    """
    roots, seen = [], set()

    def add(p: object) -> None:
        """
        Appends p to the enclosing `roots` list if it is a ModelPatcher not already added (deduplicated by id via `seen`).

        Args:
            p: candidate object; ignored if it is not a ModelPatcher
        """
        if isinstance(p, comfy.model_patcher.ModelPatcher) and id(p) not in seen:
            seen.add(id(p))
            roots.append(p)

    add(obj)
    add(getattr(obj, 'patcher', None))
    get_models = getattr(obj, 'get_models', None)
    
    if callable(get_models):
        try:
            models = get_models()
            if isinstance(models, Iterable):
                for m in models:
                    add(m)
        except Exception as e:
            logging.warning(f'V_RAM_Saver: get_models() failed: {e}.')
    
    return roots

def _source_names(prompt: dict | None, unique_id: str | int, input_name: str = 'wipe_model') -> list[str]:
    """
    Best-effort lookup of loader file names upstream of the routed socket by walking the prompt graph (max 8 hops, following model/clip/vae links).
    Never raises.

    Args:
        prompt: ComfyUI prompt graph (hidden PROMPT input)
        unique_id: id of this node in the prompt
        input_name: name of the input socket to trace from

    Returns:
        list of 'kind: filename' strings (e.g. 'ckpt: model.safetensors'); empty if nothing is found
    """
    names = []
    if prompt:
        try:
            link = prompt[str(unique_id)]['inputs'].get(input_name)
            for _ in range(8):
                if not (isinstance(link, list) and len(link) == 2):
                    break
                inputs = prompt[str(link[0])].get('inputs', {})
                for k, v in inputs.items():
                    m = re.match(r'(.+)_name\d*$', k)
                    if m and isinstance(v, str):
                        names.append(f'{m.group(1)}: {v}')
                link = next((inputs[k] for k in ('model', 'clip', 'vae') if isinstance(inputs.get(k), list)), None)
        except Exception:
            pass
    
    return names

def _wipe(patchers: list[comfy.model_patcher.ModelPatcher]) -> list[tuple[str, int, int]]:
    """
    Hard-wipes models without a CPU copy: removes them from ComfyUI's loaded-model list, unpatches without moving weights, clears backups, empties weights, then runs gc and clears the cache. Relies on internal ComfyUI attributes; a missing one is logged as a warning and its step is skipped instead of aborting.

    Args:
        patchers: ModelPatchers to wipe (nested ones only if the caller included them)

    Returns:
        list of (architecture_name, vram_bytes, ram_bytes) per wiped model; dynamic-VRAM models are not listed
    """
    uuids = {u for u in (getattr(p, 'clone_base_uuid', None) for p in patchers) if u is not None}
    cur = getattr(mm, 'current_loaded_models', None)
    
    if cur is None or not uuids:
        logging.warning('V_RAM_Saver: current_loaded_models/clone_base_uuid unavailable, skipping ComfyUI bookkeeping cleanup.')
        cur = []

    # 1. drop ComfyUI's bookkeeping for these models
    for i in range(len(cur) - 1, -1, -1):
        lm = cur[i]
        m = getattr(lm, 'model', None)
        if m is not None and getattr(m, 'clone_base_uuid', None) in uuids:
            fin = getattr(lm, 'model_finalizer', None)
            detach = getattr(fin, 'detach', None)
            if callable(detach):
                detach()
            lm.model_finalizer = None
            lm.real_model = None
            del cur[i]

    details = []
    done = set()

    # 2. unpatch WITHOUT moving weights (device_to=None skips model.to())
    for p in patchers:    
        try:
            p.unpatch_model(device_to = None, unpatch_weights = True)
        except Exception as e:
            logging.warning(f'V_RAM_Saver: unpatch_model failed ({e}), continuing with wipe.')
        for attr in ('backup', 'backup_buffers', 'object_patches_backup'):
            d = getattr(p, attr, None)
            if d and hasattr(d, 'clear'):
                d.clear()

        # 3. empty the weights (dynamic-VRAM patchers already freed in step 2)
        is_dyn = getattr(p, 'is_dynamic', None)
        if not callable(is_dyn):
            logging.warning('V_RAM_Saver: is_dynamic() unavailable, treating model as non-dynamic.')
        if p.model is None or (callable(is_dyn) and is_dyn()) or id(p.model) in done:
            continue
        done.add(id(p.model))
        v, r = _empty_module(p.model)
        details.append((p.model.__class__.__name__, v, r))

    gc.collect()
    mm.soft_empty_cache()
    
    return details

def _with_nested(roots: list[comfy.model_patcher.ModelPatcher], include_nested: bool = True) -> list[comfy.model_patcher.ModelPatcher]:
    """
    Description summary: optionally expands patchers with their nested additional models, removing duplicates and keeping order.

    Args:
        roots: ModelPatchers to expand
        include_nested: add each root's nested additional models; if False, only the roots are returned

    Returns:
        Roots (followed by their nested additional models if include_nested), without duplicates
    """
    out, seen = [], set()
    for r in roots:
        for p in [r] + (list(r.get_nested_additional_models()) if include_nested else []):
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    
    return out

class V_RAM_Saver:
    """
    Inline passthrough node: wipes or offloads ONE routed model and/or schedules a node-cache reset, then returns its input unchanged.
    Wire `anything` from the LAST consumer of the routed model.
    """
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        """
        Declares the node's inputs for ComfyUI.

        Returns:
            dict with 'required' (anything), 'optional' (wipe_model, wipe, offload_model, clear_cache, wipe_nested) and 'hidden' (prompt, unique_id) inputs
        """
        return {
            'required': {'anything': (any_type, {}),},
            'optional': {
                'wipe_model': (any_type, {}),
                'wipe': ('BOOLEAN', {'default': False}),
                'offload_model': ('BOOLEAN', {'default': False}),
                'clear_cache': ('BOOLEAN', {'default': False}),
                'wipe_nested': ('BOOLEAN', {'default': True}),
            },
            'hidden': {'prompt': 'PROMPT', 'unique_id': 'UNIQUE_ID'},
        }

    RETURN_TYPES = (any_type, 'STRING')
    RETURN_NAMES = ('anything', 'status')
    FUNCTION = 'run'
    CATEGORY = 'X-tools'
    DESCRIPTION = (
        'Passes its input through unchanged. wipe: removes the routed model from VRAM and RAM '
        'without copying it to CPU (forces clear_cache). wipe_nested: also wipes its nested '
        'additional models (ControlNet, etc.); turn off to wipe only the routed model. '
        'offload_model: normal ComfyUI unload of the routed model only (copies to CPU). '
        'clear_cache: resets the node output cache after the prompt ends.'
    )

    def run(self, anything: object, wipe_model: object = None, wipe: bool = False, offload_model: bool = False, clear_cache: bool = False, wipe_nested: bool = True, prompt: dict | None = None, unique_id: str | int | None = None) -> tuple[object, str]:
        """
        Passes `anything` through and optionally wipes the routed model (no CPU copy, forces clear_cache), offloads it normally to CPU, and/or schedules a node-cache reset after the prompt ends.

        Args:
            anything: value passed through unchanged; also enforces execution order
            wipe_model: model/CLIP/VAE to free
            wipe: wipe wipe_model from VRAM and RAM (takes priority over offload_model)
            offload_model: normal ComfyUI unload of wipe_model (copies to CPU)
            clear_cache: reset the node output cache after the prompt ends
            wipe_nested: when wiping, also wipe the nested additional models (ControlNet, etc.); if False, only the routed model is wiped
            prompt: hidden ComfyUI prompt graph
            unique_id: hidden id of this node

        Returns:
            anything: the unchanged input
            status: '; '-joined notes on actions taken, or 'nothing to do'
        """
        notes = []
        roots = _resolve_roots(wipe_model) if wipe_model is not None else []

        wiping = bool(wipe and roots)
        if wipe and wipe_model is None:
            notes.append('wipe skipped: no model connected')
        elif wipe and not roots:
            notes.append('wipe skipped: unsupported input type')

        offloading = bool(offload_model and roots and not wiping)
        if offload_model and wiping:
            notes.append('offload skipped (wipe active)')
        elif offload_model and not roots:
            notes.append('offload skipped: no usable model connected')

        do_clear = bool(clear_cache or wiping)
        if wiping and not clear_cache:
            notes.append('clear_cache forced by wipe')

        # Cache reset first: unload_models must be explicitly False, otherwise
        # prompt_worker falls back to free_memory's value and unloads every model.
        if do_clear:
            try:
                q = PromptServer.instance.prompt_queue
                q.set_flag('unload_models', False)
                q.set_flag('free_memory', True)
                notes.append('cache reset scheduled')
            except Exception as e:
                notes.append(f'cache reset FAILED: {e}')
                if wiping:
                    wiping = False
                    notes.append('wipe aborted (stale cache would break next run)')

        rich = {}  # plain note -> colored console version
        
        if wiping and unique_id:
            try:
                details = _wipe(_with_nested(roots, wipe_nested))
                kind = _kind(wipe_model)
                src = _source_names(prompt, unique_id)
                src_txt = f" [{', '.join(src)}]" if src else ''
                parts = [(a, f'VRAM {v / 2**20:.0f} MB, RAM {r / 2**20:.0f} MB') for a, v, r in details]
                if not parts:
                    parts = [('dynamic-VRAM model', 'size not measured')]
                plain = f'wiped {kind}{src_txt}: ' + ' + '.join(f'{a} ({s})' for a, s in parts)
                rich[plain] = (
                    _c('wiped ', 'green', True) + _c(kind, 'magenta', True) + _c(src_txt, 'cyan') + ': '
                    + ' + '.join(f"{a} ({_c(s, 'green')})" for a, s in parts)
                )
                notes.append(plain)
            except Exception as e:
                notes.append(f'wipe FAILED: {e}')
        elif offloading:
            try:
                for r in roots:
                    mm.unload_model_and_clones(r, unload_additional_models = True, all_devices = True)
                notes.append('offloaded to CPU')
            except Exception as e:
                notes.append(f'offload FAILED: {e}')

        status = '; '.join(notes) if notes else 'nothing to do'
        console = ' | '.join(rich.get(n) or _note_color(n) for n in notes) if notes else _c(status, 'gray')
        print(f"{_c('V_RAM_Saver', 'cyan', True)}: {console}")
        
        return (anything, status)

NODE_CLASS_MAPPINGS = {'V_RAM_Saver': V_RAM_Saver,}
NODE_DISPLAY_NAME_MAPPINGS = {'V_RAM_Saver': 'V_RAM Saver (Wipe Model)',}