"""
LLDB pretty printers for xtensor types
"""

import lldb


def get_shape_from_xtensor(valobj):
    """Extract shape dimensions from an xtensor object"""
    # m_shape is in the base class xstrided_container
    # It's a std::array whose actual data is in __elems_ (libc++) or _M_elems (libstdc++)

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
        # Shape is std::array<size_t, N>
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


def get_data_values(valobj, max_values=4):
    """Get the first few data values from the tensor"""
    raw_val = valobj.GetNonSyntheticValue()
    storage = raw_val.GetChildMemberWithName('m_storage')

    values = []
    if storage.IsValid():
        # Try to get p_begin (for xt::uvector)
        p_begin = storage.GetChildMemberWithName('p_begin')

        if p_begin.IsValid() and p_begin.GetValueAsUnsigned() != 0:
            # Get the actual pointer value
            addr = p_begin.GetValueAsUnsigned()
            # Get the type of elements (e.g., float)
            elem_type = p_begin.GetType().GetPointeeType()

            if elem_type.IsValid():
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


def format_tensor_values(valobj, shape_dims, max_per_dim=3):
    """Format tensor values in xtensor style with nested braces"""
    if not shape_dims:
        return ""

    raw_val = valobj.GetNonSyntheticValue()
    storage = raw_val.GetChildMemberWithName('m_storage')

    if not storage.IsValid():
        return ""

    p_begin = storage.GetChildMemberWithName('p_begin')
    if not p_begin.IsValid() or p_begin.GetValueAsUnsigned() == 0:
        return ""

    addr = p_begin.GetValueAsUnsigned()
    elem_type = p_begin.GetType().GetPointeeType()

    if not elem_type.IsValid():
        return ""

    elem_size = elem_type.GetByteSize()

    def read_value(index):
        """Read a single value at the given flat index"""
        elem_addr = addr + (index * elem_size)
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

    def format_recursive(dims, offset):
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
                    values.append(read_value(offset + i))
            else:
                # Show first few, ..., last few
                for i in range(max_per_dim):
                    values.append(read_value(offset + i))
                values.append("...")
                for i in range(n - max_per_dim, n):
                    values.append(read_value(offset + i))

            return "{" + ", ".join(f"{v:>9}" if v != "..." else v for v in values) + "}"

        # Multiple dimensions - recurse
        n = dims[0]
        stride = 1
        for d in dims[1:]:
            stride *= d

        parts = []

        if n <= max_per_dim * 2:
            # Show all sub-tensors
            for i in range(n):
                sub = format_recursive(dims[1:], offset + i * stride)
                parts.append(sub)
        else:
            # Show first few, ..., last few
            for i in range(max_per_dim):
                sub = format_recursive(dims[1:], offset + i * stride)
                parts.append(sub)
            parts.append("...")
            for i in range(n - max_per_dim, n):
                sub = format_recursive(dims[1:], offset + i * stride)
                parts.append(sub)

        return "{" + ", ".join(parts) + "}"

    return format_recursive(shape_dims, 0)


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

    # Enable the category
    debugger.HandleCommand('type category enable xtensor')

    print("xtensor pretty printers loaded")
