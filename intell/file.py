import os


class File:
    # based on Django.core.files.base.File
    encoding = property(lambda self: self.file.encoding)
    fileno = property(lambda self: self.file.fileno)
    flush = property(lambda self: self.file.flush)
    isatty = property(lambda self: self.file.isatty)
    newlines = property(lambda self: self.file.newlines)
    read = property(lambda self: self.file.read)
    readinto = property(lambda self: self.file.readinto)
    readline = property(lambda self: self.file.readline)
    readlines = property(lambda self: self.file.readlines)
    seek = property(lambda self: self.file.seek)
    tell = property(lambda self: self.file.tell)
    truncate = property(lambda self: self.file.truncate)
    write = property(lambda self: self.file.write)
    writelines = property(lambda self: self.file.writelines)


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
