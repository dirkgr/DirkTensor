"""
LLDB pretty printers for xtensor types
"""

import lldb


def get_shape_from_xtensor(valobj):
    """Extract shape dimensions from an xtensor object"""
    # m_shape is in the base class xstrided_container
    # It can be either:
    # - std::array whose actual data is in __elems_ (libc++) or _M_elems (libstdc++)
    # - svector (small vector) with m_data member

    # IMPORTANT: Get the non-synthetic value to access the raw C++ object
    raw_val = valobj.GetNonSyntheticValue()

    # First, try direct access
    shape = raw_val.GetChildMemberWithName('m_shape')

    if not shape.IsValid():
        # Try to get it from base class
        # The structure is: xtensor_container -> xstrided_container -> m_shape
        base_class_count = raw_val.GetNumChildren()
        for i in range(base_class_count):
            base = raw_val.GetChildAtIndex(i)
            if 'xstrided' in str(base.GetType()):
                shape = base.GetChildMemberWithName('m_shape')
                if shape.IsValid():
                    break

    shape_dims = []
    if shape.IsValid():
        # First try svector (has m_data, m_begin, m_end)
        m_data = shape.GetChildMemberWithName('m_data')
        if m_data.IsValid():
            # svector case - read from m_data
            m_begin = shape.GetChildMemberWithName('m_begin')
            m_end = shape.GetChildMemberWithName('m_end')
            if m_begin.IsValid() and m_end.IsValid():
                begin_addr = m_begin.GetValueAsUnsigned()
                end_addr = m_end.GetValueAsUnsigned()
                # Calculate number of elements
                elem_type = m_begin.GetType().GetPointeeType()
                if elem_type.IsValid():
                    elem_size = elem_type.GetByteSize()
                    if elem_size > 0:
                        num_elems = (end_addr - begin_addr) // elem_size
                        # Read from m_data array
                        for i in range(min(num_elems, m_data.GetNumChildren())):
                            elem = m_data.GetChildAtIndex(i)
                            if elem.IsValid():
                                val = elem.GetValueAsUnsigned(0)
                                # Filter out garbage values (unreasonably large)
                                if val < 1000000000:  # sanity check
                                    shape_dims.append(val)
        else:
            # std::array case - Shape is std::array<size_t, N>
            # The actual data is in __elems_ (libc++) or _M_elems (libstdc++)
            elems = shape.GetChildMemberWithName('__elems_')
            if not elems.IsValid():
                elems = shape.GetChildMemberWithName('_M_elems')

            if elems.IsValid():
                # elems is a C array, iterate over it
                num_dims = elems.GetNumChildren()
                for i in range(num_dims):
                    elem = elems.GetChildAtIndex(i)
                    if elem.IsValid():
                        val = elem.GetValueAsUnsigned(0)
                        shape_dims.append(val)

    return shape_dims


def get_storage_from_container(valobj):
    """Find m_storage in a container, checking base classes if needed"""
    storage = valobj.GetChildMemberWithName('m_storage')
    if storage.IsValid():
        return storage

    # Try base classes
    for i in range(valobj.GetNumChildren()):
        base = valobj.GetChildAtIndex(i)
        if 'xstrided_container' in str(base.GetType()) or 'xarray_container' in str(base.GetType()):
            storage = base.GetChildMemberWithName('m_storage')
            if storage.IsValid():
                return storage

    return None


def get_data_pointer(valobj):
    """Get the data pointer and element type from a tensor or view"""
    raw_val = valobj.GetNonSyntheticValue()
    storage = raw_val.GetChildMemberWithName('m_storage')

    if not storage.IsValid():
        return None, None

    # Try direct p_begin first (for containers)
    p_begin = storage.GetChildMemberWithName('p_begin')

    if not p_begin.IsValid() or p_begin.GetValueAsUnsigned() == 0:
        # For views: m_storage.m_e points to the underlying expression
        m_e = storage.GetChildMemberWithName('m_e')
        if m_e.IsValid():
            # Dereference the pointer to get the underlying expression
            underlying = m_e.Dereference()
            if underlying.IsValid():
                # IMPORTANT: Get non-synthetic value to access raw C++ members
                underlying_raw = underlying.GetNonSyntheticValue()
                # Find m_storage in the underlying container (may be in base class)
                underlying_storage = get_storage_from_container(underlying_raw)
                if underlying_storage and underlying_storage.IsValid():
                    p_begin = underlying_storage.GetChildMemberWithName('p_begin')

    if p_begin.IsValid() and p_begin.GetValueAsUnsigned() != 0:
        addr = p_begin.GetValueAsUnsigned()
        elem_type = p_begin.GetType().GetPointeeType()
        return addr, elem_type

    return None, None


def get_data_values(valobj, max_values=4):
    """Get the first few data values from the tensor"""
    addr, elem_type = get_data_pointer(valobj)

    values = []
    if addr and elem_type and elem_type.IsValid():
        for i in range(max_values):
            # Create a value at the offset address
            elem_addr = addr + (i * elem_type.GetByteSize())
            elem = valobj.CreateValueFromAddress(f"elem_{i}", elem_addr, elem_type)
            if elem.IsValid():
                value = elem.GetValue()
                if value:
                    values.append(value)
            else:
                break

    return values


def get_strides(valobj, shape_dims):
    """Get strides for the tensor or view"""
    raw_val = valobj.GetNonSyntheticValue()

    # Try to get m_strides directly (for views and containers)
    strides_obj = raw_val.GetChildMemberWithName('m_strides')

    if not strides_obj.IsValid():
        # Try in base class
        for i in range(raw_val.GetNumChildren()):
            base = raw_val.GetChildAtIndex(i)
            if 'xstrided' in str(base.GetType()):
                strides_obj = base.GetChildMemberWithName('m_strides')
                if strides_obj.IsValid():
                    break

    strides = []
    if strides_obj.IsValid():
        # Try std::array (__elems_)
        elems = strides_obj.GetChildMemberWithName('__elems_')
        if not elems.IsValid():
            elems = strides_obj.GetChildMemberWithName('_M_elems')

        if elems.IsValid():
            num_dims = min(len(shape_dims), elems.GetNumChildren())
            for i in range(num_dims):
                elem = elems.GetChildAtIndex(i)
                if elem.IsValid():
                    strides.append(elem.GetValueAsUnsigned(0))

    # If we couldn't get strides, calculate default row-major strides
    if not strides and shape_dims:
        strides = [1]
        for i in range(len(shape_dims) - 1, 0, -1):
            strides.insert(0, strides[0] * shape_dims[i])

    return strides


def format_tensor_values(valobj, shape_dims, max_per_dim=3):
    """Format tensor values in xtensor style with nested braces"""
    if not shape_dims:
        return ""

    addr, elem_type = get_data_pointer(valobj)
    if not addr or not elem_type or not elem_type.IsValid():
        return ""

    elem_size = elem_type.GetByteSize()

    # Get strides and offset for proper indexing
    strides = get_strides(valobj, shape_dims)

    # Get offset (for views)
    raw_val = valobj.GetNonSyntheticValue()
    m_offset = raw_val.GetChildMemberWithName('m_offset')
    offset_value = m_offset.GetValueAsUnsigned(0) if m_offset.IsValid() else 0

    def read_value(index):
        """Read a single value at the given strided index"""
        elem_addr = addr + ((offset_value + index) * elem_size)
        elem = valobj.CreateValueFromAddress(f"elem", elem_addr, elem_type)
        if elem.IsValid():
            val = elem.GetValue()
            if val:
                # Round to 6 significant digits for floats
                try:
                    num = float(val)
                    # Use 'g' format for significant digits, remove trailing zeros
                    formatted = f"{num:.6g}"
                    return formatted
                except (ValueError, TypeError):
                    # Not a float, return as-is
                    return val
        return "?"

    def compute_index(indices):
        """Compute flat index from multi-dimensional indices using strides"""
        idx = 0
        for i, stride in zip(indices, strides):
            idx += i * stride
        return idx

    def format_recursive(dims, dim_indices):
        """Recursively format tensor dimensions"""
        if len(dims) == 0:
            return ""

        if len(dims) == 1:
            # Last dimension - print values
            n = dims[0]
            values = []

            if n <= max_per_dim * 2:
                # Show all values
                for i in range(n):
                    indices = dim_indices + [i]
                    flat_idx = compute_index(indices)
                    values.append(read_value(flat_idx))
            else:
                # Show first few, ..., last few
                for i in range(max_per_dim):
                    indices = dim_indices + [i]
                    flat_idx = compute_index(indices)
                    values.append(read_value(flat_idx))
                values.append("...")
                for i in range(n - max_per_dim, n):
                    indices = dim_indices + [i]
                    flat_idx = compute_index(indices)
                    values.append(read_value(flat_idx))

            return "{" + ", ".join(f"{v:>9}" if v != "..." else v for v in values) + "}"

        # Multiple dimensions - recurse
        n = dims[0]
        parts = []

        if n <= max_per_dim * 2:
            # Show all sub-tensors
            for i in range(n):
                sub = format_recursive(dims[1:], dim_indices + [i])
                parts.append(sub)
        else:
            # Show first few, ..., last few
            for i in range(max_per_dim):
                sub = format_recursive(dims[1:], dim_indices + [i])
                parts.append(sub)
            parts.append("...")
            for i in range(n - max_per_dim, n):
                sub = format_recursive(dims[1:], dim_indices + [i])
                parts.append(sub)

        return "{" + ", ".join(parts) + "}"

    return format_recursive(shape_dims, [])


class XTensorSyntheticProvider:
    """Synthetic provider for xt::xtensor and xt::xarray types"""

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.update()

    def update(self):
        try:
            self.shape_dims = get_shape_from_xtensor(self.valobj)
            self.data_values = get_data_values(self.valobj)
        except Exception as e:
            self.shape_dims = []
            self.data_values = []

    def num_children(self):
        # Show only values (shape is already in summary)
        return 1

    def get_child_index(self, name):
        if name == "values":
            return 0
        return -1

    def get_child_at_index(self, index):
        if index == 0:
            # Return formatted tensor values
            try:
                formatted = format_tensor_values(self.valobj, self.shape_dims)
                if formatted:
                    # Escape the string properly for LLDB
                    escaped = formatted.replace('\\', '\\\\').replace('"', '\\"')
                    return self.valobj.CreateValueFromExpression("values", f'"{escaped}"')
            except Exception as e:
                return self.valobj.CreateValueFromExpression("values", f'"error: {str(e)}"')
        return None


def xtensor_summary(valobj, internal_dict):
    """Summary provider for xtensor types"""
    try:
        shape_dims = get_shape_from_xtensor(valobj)

        if shape_dims and any(d > 0 for d in shape_dims):
            shape_str = "shape=(" + ", ".join(str(d) for d in shape_dims) + ")"

            # Get formatted values
            formatted = format_tensor_values(valobj, shape_dims)
            if formatted:
                # Show a prefix of the formatted values (max 60 chars)
                max_len = 60
                if len(formatted) > max_len:
                    prefix = formatted[:max_len].rstrip() + " ..."
                else:
                    prefix = formatted
                return f"{shape_str} {prefix}"
            else:
                return shape_str

        return "empty"
    except Exception as e:
        return f"error: {str(e)}"


def __lldb_init_module(debugger, internal_dict):
    """Initialize the module for lldb"""

    # Register summary and synthetic providers for xtensor types
    debugger.HandleCommand(
        'type summary add -F xtensor_lldb.xtensor_summary '
        '-x "^xt::xtensor_container<.*>$" '
        '-w xtensor'
    )

    debugger.HandleCommand(
        'type synthetic add '
        '-l xtensor_lldb.XTensorSyntheticProvider '
        '-x "^xt::xtensor_container<.*>$" '
        '-w xtensor'
    )

    # Also handle xt::xarray (same underlying type)
    debugger.HandleCommand(
        'type summary add -F xtensor_lldb.xtensor_summary '
        '-x "^xt::xarray<.*>$" '
        '-w xtensor'
    )

    # Handle xarray_container (result type from tensordot, etc.)
    debugger.HandleCommand(
        'type summary add -F xtensor_lldb.xtensor_summary '
        '-x "^xt::xarray_container<.*>$" '
        '-w xtensor'
    )

    debugger.HandleCommand(
        'type synthetic add '
        '-l xtensor_lldb.XTensorSyntheticProvider '
        '-x "^xt::xarray_container<.*>$" '
        '-w xtensor'
    )

    # Handle xtensor expression types (xfunction, xview, etc.)
    debugger.HandleCommand(
        'type summary add -F xtensor_lldb.xtensor_summary '
        '-x "^xt::xfunction<.*>$" '
        '-w xtensor'
    )

    debugger.HandleCommand(
        'type synthetic add '
        '-l xtensor_lldb.XTensorSyntheticProvider '
        '-x "^xt::xfunction<.*>$" '
        '-w xtensor'
    )

    debugger.HandleCommand(
        'type summary add -F xtensor_lldb.xtensor_summary '
        '-x "^xt::xview<.*>$" '
        '-w xtensor'
    )

    debugger.HandleCommand(
        'type synthetic add '
        '-l xtensor_lldb.XTensorSyntheticProvider '
        '-x "^xt::xview<.*>$" '
        '-w xtensor'
    )

    debugger.HandleCommand(
        'type summary add -F xtensor_lldb.xtensor_summary '
        '-x "^xt::xstrided_view<.*>$" '
        '-w xtensor'
    )

    debugger.HandleCommand(
        'type synthetic add '
        '-l xtensor_lldb.XTensorSyntheticProvider '
        '-x "^xt::xstrided_view<.*>$" '
        '-w xtensor'
    )

    # Enable the category
    debugger.HandleCommand('type category enable xtensor')

    print("xtensor pretty printers loaded")
