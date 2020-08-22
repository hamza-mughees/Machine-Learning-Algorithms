class Quick_Sort:
    def __init__(self):
        self.lst = []
        self.inds = []
    
    def sort(self, lst):
        self.lst = lst
        self.inds = [i for i in range(len(self.lst))]
        self._sort(self.lst, 0, len(self.lst)-1)

    def _sort(self, lst, start, end):
        if start >= end: return
        index = self._partition(lst, start, end)
        self._sort(lst, start, index-1)
        self._sort(lst, index+1, end)

    def _partition(self, lst, start, end):
        piv_ind = start
        piv_val = lst[end]
        for i in range(start, end):
            if lst[i] < piv_val:
                self._swap(i, piv_ind)
                piv_ind += 1
        self._swap(piv_ind, end)
        return piv_ind

    def _swap(self, i, j):
        self.lst[i], self.lst[j] = self.lst[j], self.lst[i]
        self.inds[i], self.inds[j] = self.inds[j], self.inds[i]
