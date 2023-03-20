class Proxy:
    def __init__(self) -> None:
        self.x = 2

    def __call__(self, x) -> int:
        return self.x * x
    

proxy = Proxy()

print(proxy(5))