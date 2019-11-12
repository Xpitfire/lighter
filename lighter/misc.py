import os
import lighter
from string import Template
from shutil import copyfile


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
    def resolve(parent, name):
        prev_parent = parent
        groups = name.split('.')
        for group in groups[:-1]:
            parent = parent.get_value(group)
            if parent is None:
                parent = DotDict()
                setattr(prev_parent, group, parent)
        return parent, groups[-1]


def get_lighter_root():
    return '/'.join(os.path.abspath(lighter.__file__).split('/')[:-1])


def create_template_file(project_name, module_name, source_name, target_name, template):
    root = get_lighter_root()
    template_file_name = os.path.join(root, 'templates/{}/{}'.format(module_name, source_name))
    if not os.path.exists(template_file_name):
        return
    with open(template_file_name, 'r') as file:
        src = Template(file.read())
        result = src.substitute(template)
    target_dir_name = os.path.join(project_name, module_name)
    if not os.path.exists(target_dir_name):
        os.makedirs(target_dir_name)
    target_file_name = os.path.join(target_dir_name, target_name)
    with open(target_file_name, 'w') as file:
        file.write(result)


def copy_file(project_name, module_name, source_name, target_name):
    root = get_lighter_root()
    template_file_name = os.path.join(root, 'templates/{}/{}'.format(module_name, source_name))
    if not os.path.exists(template_file_name):
        return
    target_dir_name = os.path.join(project_name, module_name)
    if not os.path.exists(target_dir_name):
        os.makedirs(target_dir_name)
    copyfile(template_file_name, os.path.join(target_dir_name, target_name))


def create_init(project_name, module_name: str = None):
    if module_name is None:
        target_dir_name = project_name
    else:
        target_dir_name = os.path.join(project_name, module_name)
    if not os.path.exists(target_dir_name):
        os.makedirs(target_dir_name)
    target_file_name = os.path.join(target_dir_name, '__init__.py')
    with open(target_file_name, 'a'):
        pass
