import os


class File:
    # based on Django.core.files.base.File
    seek = property(lambda self: self.file.seek)
    write = property(lambda self: self.file.write)

    def __init__(self, file, name=None):
        self.file = file
        if name is None:
            name = getattr(file, "name", None)
        self.name = name
        if hasattr(file, "mode"):
            self.mode = file.mode

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self, mode=None, *args, **kwargs):
        if not self.closed:
            self.seek(0)
        elif self.name and os.path.exists(self.name):
            self.file = open(self.name, mode or self.mode, *args, **kwargs)
        else:
            raise ValueError("The file cannot be reopened.")
        return self

    def close(self):
        self.file.close()

    @property
    def closed(self):
        return not self.file or self.file.closed

    def __iter__(self):
        return iter(self.file)
