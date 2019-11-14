import os
import lighter
from string import Template
from shutil import copyfile


def get_lighter_root():
    return '/'.join(os.path.abspath(lighter.__file__).split('/')[:-1])


def get_lighter_path(module_name, source_name):
    root = get_lighter_root()
    path = 'templates'
    if module_name is not None:
        path = os.path.join(path, module_name)
    path = os.path.join(path, source_name)
    template_file_name = os.path.join(root, path)
    return os.path.exists(template_file_name), template_file_name


def create_template_file(project_name, source_name, target_name, template, module_name=None, override=False):
    exists, template_file_name = get_lighter_path(module_name, source_name)
    if not exists:
        return
    with open(template_file_name, 'r') as file:
        src = Template(file.read())
        result = src.substitute(template)
    target_dir_name = project_name
    if module_name is not None:
        target_dir_name = os.path.join(project_name, module_name)
    if not os.path.exists(target_dir_name):
        os.makedirs(target_dir_name)
    target_file_name = os.path.join(target_dir_name, target_name)
    if override or not os.path.exists(target_file_name):
        with open(target_file_name, 'w') as file:
            file.write(result)


def copy_file(project_name, source_name, target_name, module_name=None, override=False):
    exists, template_file_name = get_lighter_path(module_name, source_name)
    if not exists:
        return
    target_dir_name = project_name
    if module_name is not None:
        target_dir_name = os.path.join(project_name, module_name)
    if not os.path.exists(target_dir_name):
        os.makedirs(target_dir_name)
    target_path = os.path.join(target_dir_name, target_name)
    if override or not os.path.exists(target_path):
        copyfile(template_file_name, target_path)


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
