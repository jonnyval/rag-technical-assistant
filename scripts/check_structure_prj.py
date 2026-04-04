import os

# Папки, которые мы не хотим видеть в дереве
IGNORE_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.idea', '.vscode', 'data'}

def print_tree(directory, prefix=""):
    items = sorted(os.listdir(directory))
    # Фильтруем скрытые папки и игнорируемые директории
    items = [i for i in items if i not in IGNORE_DIRS and not (os.path.isdir(os.path.join(directory, i)) and i.startswith('.'))]
    
    for i, item in enumerate(items):
        path = os.path.join(directory, item)
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        print(prefix + connector + item)
        
        if os.path.isdir(path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(path, new_prefix)

if __name__ == "__main__":
    print(f"📦 {os.path.basename(os.path.abspath('.'))}")
    print_tree('.')