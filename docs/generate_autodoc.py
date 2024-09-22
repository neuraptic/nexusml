import os


def generate_autodoc_rst(src_dir: str, output_file: str, delete_path_start: str = '') -> None:
    """
    This module will find all submodules inside the given src_dir. Then create a rst file for sphinx to autodoc all
    these modules.

    args:
    src_dir: Directory path where modules needs to be found.
    output_file: name and extension of the file where the sphinx modules config will be saved.
    delete_path_start: string to be deleted from the path saved in the sphinx config.
    """
    with open(output_file, 'w') as f:
        # Add a title to the file
        f.write('Module Documentation\n')
        f.write('====================\n\n')

        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    module = os.path.splitext(os.path.relpath(os.path.join(root, file), start=delete_path_start))[0]
                    print(os.path.join(root, file))
                    module = module.replace(os.sep, '.')
                    f.write(f'.. automodule:: {module}\n')
                    f.write('    :members:\n')
                    f.write('    :undoc-members:\n')
                    f.write('    :show-inheritance:\n\n')


if __name__ == '__main__':
    nexusml_dir = '../nexusml'  # Adjust this path as necessary
    output_rst = 'modules.rst'  # This file will be generated inside the docs directory
    start_to_delete: str = '../'
    generate_autodoc_rst(src_dir=nexusml_dir,
                         output_file=output_rst,
                         delete_path_start=start_to_delete)
