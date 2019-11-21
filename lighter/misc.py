import json
import petname
import sys

from box import Box


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


def _to_json(obj, filename=None,
             encoding="utf-8", errors="strict", **json_kwargs):
    json_dump = json.dumps(obj,
                           ensure_ascii=False, **json_kwargs)
    if filename:
        with open(filename, 'w', encoding=encoding, errors=errors) as f:
            f.write(json_dump if sys.version_info >= (3, 0) else
                    json_dump.decode("utf-8"))
    else:
        return json_dump


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

    def to_dict(self):
        """
        Turn the Box and sub Boxes back into a native
        python dictionary.
        :return: python dictionary of this Box
        """
        out_dict = dict(self)
        for k, v in out_dict.items():
            if v is self:
                out_dict[k] = out_dict
            elif hasattr(v, 'to_dict'):
                out_dict[k] = v.to_dict()
            elif hasattr(v, 'to_list'):
                out_dict[k] = v.to_list()
        return out_dict

    def to_json(self, filename=None,
                encoding="utf-8", errors="strict", **json_kwargs):
        """
        Transform the Box object into a JSON string.
        :param filename: If provided will save to file
        :param encoding: File encoding
        :param errors: How to handle encoding errors
        :param json_kwargs: additional arguments to pass to json.dump(s)
        :return: string of JSON or return of `json.dump`
        """
        return _to_json(self.to_dict(), filename=filename,
                        encoding=encoding, errors=errors, **json_kwargs)

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


# class DotDict(Box):
#     """dot.notation access to dictionary attributes"""
#     @staticmethod
#     def resolve(parent_ori, name_ori):
#         parent = parent_ori
#         prev_parent = parent_ori
#         groups = name_ori.split('.')
#         for group in groups[:-1]:
#             parent = parent.get_value(group)
#             if parent is None:
#                 parent = DotDict()
#                 setattr(prev_parent, group, parent)
#             else:
#                 prev_parent = parent
#         return parent, groups[-1]
