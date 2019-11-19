import petname


def extract_named_args(arg_list):
    result = {}
    for i in range(0, len(arg_list)):
        a = arg_list[i]
        if a.startswith("--"):
            if i + 1 < len(arg_list) and not arg_list[i + 1].startswith("--"):
                result[a] = arg_list[i + 1]
            else:
                result[a] = None
    return result


def try_to_number_or_bool(value):
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            if isinstance(value, str) and (value.lower() == "false" or value.lower() == "true"):
                value = (value.lower() == "true")
            pass
    return value


def generate_short_id():
    return petname.Generate(2, '-', 6)


def generate_long_id():
    return petname.Generate(3, '-', 6)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def get_value(self, name, default=None):
        if name in self:
            return self[name]
        else:
            return default

    def has_value(self, name):
        return name in self

    @staticmethod
    def resolve(parent_ori, name_ori):
        parent = parent_ori
        prev_parent = parent_ori
        groups = name_ori.split('.')
        for group in groups[:-1]:
            parent = parent.get_value(group)
            if parent is None:
                parent = DotDict()
                setattr(prev_parent, group, parent)
            else:
                prev_parent = parent
        return parent, groups[-1]
