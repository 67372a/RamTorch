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
    model: nn.Module, device: torch.device = torch.cuda.current_device()
):
    """
    Moves model parameters and buffers to the specified device,
    but skips any parameter or buffer that has `is_ramtorch = True`.
    """
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


def replace_linear_with_ramtorch(module: nn.Module, device: str = "cuda"):
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
            # Create a replacement
            new_layer = Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=device,
                dtype=child.weight.dtype,
                skip_init=True,
            )

            # Reference weights and bias
            with torch.no_grad():
                new_layer.weight.data = child.weight.data
                new_layer.weight.is_ramtorch = True
                if child.bias is not None:
                    new_layer.bias.data = child.bias.data
                    new_layer.bias.is_ramtorch = True

            # Replace the module in-place
            setattr(module, name, new_layer)

        else:
            # Recurse into children
            replace_linear_with_ramtorch(child, device=device)

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

def transfer_ramtensor_to_device(tensor_cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Transfer a RamTorch tensor to GPU using asynchronous streams and ping-pong buffers.
    For non-RamTorch tensors, falls back to standard .to(device).
    
    Args:
        tensor_cpu: Tensor with is_ramtorch=True or regular CPU tensor
        device: Target GPU device
    
    Returns:
        Tensor on GPU (transfer synchronized with compute stream)
    """
    if not getattr(tensor_cpu, 'is_ramtorch', False):
        return tensor_cpu.to(device, non_blocking=True)
    
    state = _get_device_state(device)
    transfer_stream = state["transfer_stream"]
    buffers = state["w_buffers"] if tensor_cpu.dim() > 0 else state["b_buffers"]
    transfer_event = state["transfer_forward_finished_event"]
    compute_event = state["compute_forward_start_event"]
    
    # Ping-pong buffer selection
    buffer_idx = state["forward_clk"] % 2
    state["forward_clk"] += 1
    
    # Async transfer on dedicated stream
    with torch.cuda.stream(transfer_stream):
        transfer_stream.wait_event(compute_event)
        buffers[buffer_idx] = tensor_cpu.to(device, non_blocking=True)
        transfer_event.record()
    
    # Compute stream waits for transfer
    torch.cuda.current_stream().wait_event(transfer_event)
    compute_event.record()
    
    return buffers[buffer_idx]
