import os

def print_tree(start_path, prefix=''):
    items = os.listdir(start_path)
    for index, name in enumerate(sorted(items)):
        path = os.path.join(start_path, name)
        connector = '└── ' if index == len(items) - 1 else '├── '
        print(prefix + connector + name)
        if os.path.isdir(path):
            extension = '    ' if index == len(items) - 1 else '│   '
            print_tree(path, prefix + extension)

# 修改成你要查看的路径
print_tree('./')
