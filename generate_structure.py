import os
from pathlib import Path


def generate_file_structure(startpath: str, output_file: str = 'project_structure.txt'):
    """Generate a file structure tree and save it to a file."""

    # Convert to Path object for better Windows path handling
    start_path = Path(startpath)

    # Files/directories to exclude
    exclude = {
        '__pycache__',
        '.git',
        '.idea',
        'venv',
        'env',
        '.pytest_cache',
        '.vscode',
        'node_modules'
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'Project Structure for: {start_path.absolute()}\n')
        f.write('=' * 50 + '\n\n')

        for root, dirs, files in os.walk(start_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude]

            # Get relative path
            rel_path = os.path.relpath(root, start_path)
            if rel_path == '.':
                level = 0
            else:
                level = len(Path(rel_path).parts)

            # Print directory
            indent = '    ' * level
            directory = os.path.basename(root)
            if level == 0:
                f.write(f'{directory}/\n')
            else:
                f.write(f'{indent}└── {directory}/\n')

            # Print files
            sub_indent = '    ' * (level + 1)
            for file in sorted(files):
                if not any(file.endswith(ext) for ext in ['.pyc', '.pyo']):
                    f.write(f'{sub_indent}└── {file}\n')


if __name__ == '__main__':
    # Get current directory if no path provided
    project_path = os.getcwd()
    output_file = 'project_structure.txt'

    print(f'Generating file structure for: {project_path}')
    generate_file_structure(project_path, output_file)
    print(f'File structure has been saved to: {os.path.abspath(output_file)}')