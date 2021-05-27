class TypeError(Exception):
    def __init__(self, value):
        self.value = value

    # 返回异常类对象的说明信息
    def __str__(self):
        return ("{} is not supported Type".format(repr(self.value)))
