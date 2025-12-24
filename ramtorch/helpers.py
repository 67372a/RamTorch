from collections import OrderedDict
from typing import Callable, Dict
import torch
import torch.nn as nn
from .modules.linear import Linear, _get_device_state


def add_custom_hooks(tensor: torch.Tensor, hook_name: str = "_custom_hooks"):
    """
    Add a custom hook dictionary to a tensor, similar to _post_accumulate_grad_hooks

    Args:
        tensor: The tensor to add hooks to
        hook_name: Name of the hook attribute (default: "_custom_hooks")

    Returns:
        The tensor with the hook attribute added
    """
    if not hasattr(tensor, hook_name):
        setattr(tensor, hook_name, OrderedDict())
        setattr(tensor, f"{hook_name}_counter", 0)
    return tensor


def register_ramtorch_hook(tensor: torch.Tensor, hook: Callable, hook_name: str) -> int:
    """
    Register a hook to the tensor

    Args:
        tensor: The tensor to register the hook on
        hook: Callable to register
        hook_name: Name of the hook attribute

    Returns:
        hook_id: Integer ID to remove the hook later
    """
    # Ensure hook dict exists
    if not hasattr(tensor, hook_name):
        add_custom_hooks(tensor, hook_name)

    hooks = getattr(tensor, hook_name)
    counter_name = f"{hook_name}_counter"
    counter = getattr(tensor, counter_name)

    # Add hook with unique ID
    hook_id = counter
    hooks[hook_id] = hook
    setattr(tensor, counter_name, counter + 1)

    return hook_id


def register_ramtorch_grad_hook(module, hook_fn, param_names=None):
    """
    Register backward hooks on module parameters.

    Args:
        module: PyTorch module to register hooks on
        hook_fn: Hook function that takes gradient tensor and optionally returns modified gradient
        param_names: Optional list of parameter names to register hooks on. If None, registers on all parameters.

    Returns:
        List of hook handles that can be used to remove hooks later

    Example:
    ```python
        def my_hook(grad):
            print(f"Gradient norm: {grad.norm()}")
            return grad * 0.5  # Scale gradient

        handles = register_ramtorch_grad_hook(model, my_hook)
        # Later: [h.remove() for h in handles]
    ```
    """
    handles = []

    for name, param in module.named_parameters():
        if param.requires_grad:
            # Filter by parameter names if specified
            if param_names is None or name in param_names:
                if hasattr(param, "is_ramtorch") and param.is_ramtorch:
                    handle = register_ramtorch_hook(
                        param, hook_fn, "_ramtorch_backward_hooks"
                    )
                else:
                    handle = param.register_hook(hook_fn)
                # TODO this works but if it not ramtorch then i need to add handles
                handles.append(handle)

    return handles


def register_ramtorch_post_accumulate_grad_hook(module, hook_fn, param_names=None):
    """
    Register post-accumulate gradient hooks on module parameters.

    IMPORTANT: Post-accumulate hooks work differently for ramtorch tensors:

    For ramtorch tensors (CPU-bouncing parameters):
        - Hook receives the tensor itself as argument: hook_fn(tensor)
        - Access gradients via tensor.ramtorch_grad (NOT tensor.grad)
        - Gradients are on GPU when hook executes
        - Modify in-place: tensor.ramtorch_grad.add_(value)
        - Hook should NOT return anything

    For regular tensors:
        - Hook receives the tensor itself as argument: hook_fn(tensor)
        - Access gradients via tensor.grad
        - Gradients are on their native device
        - Modify in-place: tensor.grad.add_(value)
        - Hook should NOT return anything

    Example usage:
    ```python
        def post_accum_fn(tensor):
            if hasattr(tensor, "is_ramtorch") and tensor.is_ramtorch:
                tensor.ramtorch_grad.add_(60)  # Modify GPU gradient
            else:
                tensor.grad.add_(60)  # Modify regular gradient

        register_ramtorch_post_accumulate_grad_hook(model, post_accum_fn)
    ```
    Args:
        module: PyTorch module to register hooks on
        hook_fn: Callable that takes (tensor) and modifies gradients in-place
        param_names: Optional list of parameter names to filter (None = all params)

    Returns:
        List of hook handles
    """
    handles = []

    for name, param in module.named_parameters():
        if param.requires_grad:
            # Filter by parameter names if specified
            if param_names is None or name in param_names:
                if hasattr(param, "is_ramtorch") and param.is_ramtorch:
                    handle = register_ramtorch_hook(
                        param, hook_fn, "_ramtorch_post_accumulate_grad_hooks"
                    )
                else:
                    handle = param.register_post_accumulate_grad_hook(hook_fn)
                # TODO this works but if it not ramtorch then i need to add handles
                handles.append(handle)

    return handles


def move_model_to_device(
    model: nn.Module, device: torch.device|str = None
):
    """
    Moves model parameters and buffers to the specified device,
    but skips any parameter or buffer that has `is_ramtorch = True`.
    """

    if device is None:
        device = torch.cuda.current_device()

    if isinstance(device, str):
        device = torch.device(device)

    for name, param in model.named_parameters(recurse=True):
        if getattr(param, "is_ramtorch", False):
            # Skip moving this param
            continue
        # Move only if not already on the target device
        if param.device != device:
            with torch.no_grad():
                new_param = param.to(device)
            param.data = new_param
            if param._grad is not None:
                param._grad = param._grad.to(device)

    for full_name, buf in model.named_buffers(recurse=True):
        if getattr(buf, "is_ramtorch", False):
            continue
        if buf.device == device:
            continue

        with torch.no_grad():
            new_buf = buf.to(device)

        # Traverse to the owning module
        module = model
        *parents, attr = full_name.split(".")
        for p in parents:
            module = getattr(module, p)

        module._buffers[attr] = new_buf

    return model

def _copy_hooks_to_ramtorch(source_param, target_param, hook_type):
    """
    Copy hooks from a source parameter to a target ramtorch parameter.
    
    Args:
        source_param:  Original parameter with hooks
        target_param: New ramtorch parameter to copy hooks to
        hook_type: Type of hook to copy ('backward', 'post_accumulate')
    """
    if hook_type == "backward":
        # Copy standard backward hooks
        if hasattr(source_param, "_backward_hooks") and source_param._backward_hooks:
            if not hasattr(target_param, "_ramtorch_backward_hooks"):
                target_param._ramtorch_backward_hooks = OrderedDict()
                target_param._ramtorch_backward_hooks_counter = 0
            
            for hook_id, hook_fn in source_param._backward_hooks.items():
                new_id = target_param._ramtorch_backward_hooks_counter
                target_param._ramtorch_backward_hooks[new_id] = hook_fn
                target_param._ramtorch_backward_hooks_counter += 1
    
    elif hook_type == "post_accumulate":
        # Copy post-accumulate grad hooks (PyTorch 2.0+)
        if hasattr(source_param, "_post_accumulate_grad_hooks") and source_param._post_accumulate_grad_hooks:
            if not hasattr(target_param, "_ramtorch_post_accumulate_grad_hooks"):
                target_param._ramtorch_post_accumulate_grad_hooks = OrderedDict()
                target_param._ramtorch_post_accumulate_grad_hooks_counter = 0
            
            for hook_id, hook_fn in source_param._post_accumulate_grad_hooks.items():
                new_id = target_param._ramtorch_post_accumulate_grad_hooks_counter
                target_param._ramtorch_post_accumulate_grad_hooks[new_id] = hook_fn
                target_param._ramtorch_post_accumulate_grad_hooks_counter += 1

def replace_linear_with_ramtorch(module: nn.Module, device: str = "cuda", target_dtype: torch.dtype = None):
    """
    Recursively replaces all nn.Linear layers in a model with CPUBouncingLinear.

    Args:
        module (nn.Module): The input model or submodule.
        device (str): Target device for computation (used by CPUBouncingLinear).

    Returns:
        nn.Module: The modified model with replacements applied in-place.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            # Determine dtype: use target if provided, else keep original
            dtype = target_dtype if target_dtype is not None else child.weight.dtype

            # Create a replacement
            new_layer = Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=device,
                dtype=dtype,
                skip_init=True,
            )

            # Reference weights and bias
            with torch.no_grad():
                if child.weight.data.device.type != "meta":
                    new_layer.weight.data.copy_(child.weight.data.to(dtype=dtype))
                new_layer.weight.requires_grad = child.weight.requires_grad 
                new_layer.weight.is_ramtorch = True

                if child.bias is not None:
                    if child.bias.data.device.type != "meta":
                        new_layer.bias.data.copy_(child.bias.data.to(dtype=dtype))
                    new_layer.bias.requires_grad = child.bias.requires_grad
                    new_layer.bias.is_ramtorch = True

            # Copy parameter hooks (backward and post-accumulate)
            _copy_hooks_to_ramtorch(child.weight, new_layer.weight, "backward")
            _copy_hooks_to_ramtorch(child.weight, new_layer.weight, "post_accumulate")
            
            if child.bias is not None and new_layer.bias is not None:
                _copy_hooks_to_ramtorch(child.bias, new_layer.bias, "backward")
                _copy_hooks_to_ramtorch(child.bias, new_layer.bias, "post_accumulate")
            
            # Copy PyTorch's native module-level forward hooks directly
            # This makes forward hooks just work without any custom implementation
            if hasattr(child, "_forward_hooks") and child._forward_hooks:
                new_layer._forward_hooks = OrderedDict(child._forward_hooks)
            
            if hasattr(child, "_forward_pre_hooks") and child._forward_pre_hooks:
                new_layer._forward_pre_hooks = OrderedDict(child._forward_pre_hooks)
            
            # Also copy state dict hooks if present
            if hasattr(child, "_state_dict_hooks") and child._state_dict_hooks:
                new_layer._state_dict_hooks = OrderedDict(child._state_dict_hooks)
            
            if hasattr(child, "_load_state_dict_pre_hooks") and child._load_state_dict_pre_hooks: 
                new_layer._load_state_dict_pre_hooks = OrderedDict(child._load_state_dict_pre_hooks)

            # Replace the module in-place
            setattr(module, name, new_layer)

        else:
            # Recurse into children
            replace_linear_with_ramtorch(child, device=device, target_dtype=target_dtype)

    return module


def reattach_is_ramtorch_flags(module: nn.Module):
    """
    Recursively traverse the module hierarchy and reattach `is_ramtorch = True`
    flags to all parameters and buffers inside any module that declares
    `is_ramtorch = True`.

    This is useful after model deserialization, replacement, or rebuilds where
    the attribute may have been lost.

    Args:
        module (nn.Module): Root module to process.
    """
    # If the current module itself is marked as a RAMTorch module,
    # mark all its parameter and buffer tensors.
    if getattr(module, "is_ramtorch", False):
        for name, param in module.named_parameters(recurse=False):
            if isinstance(param, torch.Tensor):
                param.is_ramtorch = True
        for name, buffer in module.named_buffers(recurse=False):
            if isinstance(buffer, torch.Tensor):
                buffer.is_ramtorch = True

    # Recurse into children
    for child in module.children():
        reattach_is_ramtorch_flags(child)
