def check_layout(func):
    def function_wrapper(*args, **kwargs):
        layout = func(*args, **kwargs)
        if layout is not None and not layout.is_valid():
            raise Exception('Layout ist not valid!')
        return layout
    return function_wrapper


class Tree:
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)
        obj.parent = self

    def get_values_dfs(self):
        values = [self.value]
        for child in self.children:
            values.extend(child.get_values_dfs())
        return values
