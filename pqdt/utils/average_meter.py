class AverageMeter:
    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values):
        if isinstance(values, list):
            for index, value in enumerate(values):
                self._val[index] = value
                self._sum[index] += value
                self._count[index] += 1
            return

        self._val[0] = values
        self._sum[0] += values
        self._count[0] += 1

    def val(self, index=None):
        if index is not None:
            return self._val[index]
        if self.items is None:
            return self._val[0]
        return [self._val[i] for i in range(self.n_items)]

    def count(self, index=None):
        if index is not None:
            return self._count[index]
        if self.items is None:
            return self._count[0]
        return [self._count[i] for i in range(self.n_items)]

    def avg(self, index=None):
        if index is not None:
            return self._sum[index] / self._count[index]
        if self.items is None:
            return self._sum[0] / self._count[0]
        return [self._sum[i] / self._count[i] for i in range(self.n_items)]
