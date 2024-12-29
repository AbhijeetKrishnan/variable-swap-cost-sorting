from typing import List, Callable, Tuple, SupportsIndex
from abc import ABC, abstractmethod
import random


class SortingAlgorithm[T](ABC):
    def __init__(
        self,
        name: str,
        swap_cost: Callable[[T, T], float],
        extract_cost: Callable[[T], float],
        insert_cost: Callable[[T], float],
    ):
        self.name = name
        self.swap_cost = swap_cost
        self.extract_cost = extract_cost
        self.insert_cost = insert_cost
        self.total_cost = 0.0

    @abstractmethod
    def sort(self, arr: List[T]) -> List[T]:
        pass

    def _swap(self, arr: List[T], i: T, j: T) -> None:
        if i != j:
            self.total_cost += self.swap_cost(i, j)
            arr[i], arr[j] = arr[j], arr[i]

    def _extract(self, arr: List[T], i: SupportsIndex) -> T:
        self.total_cost += self.extract_cost(i)
        return arr.pop(i)

    def _insert(self, arr: List[T], i: SupportsIndex, x: T) -> None:
        self.total_cost += self.insert_cost(i)
        arr.insert(i, x)


class BubbleSort(SortingAlgorithm):
    def sort(self, arr: List[int]) -> List[int]:
        n = len(arr)
        arr = arr.copy()
        for i in range(n):
            for j in range(n - i - 1):
                if arr[j] > arr[j + 1]:
                    self._swap(arr, j, j + 1)
        return arr


class SelectionSort(SortingAlgorithm):
    def sort(self, arr: List[int]) -> List[int]:
        n = len(arr)
        arr = arr.copy()
        for i in range(n):
            min_index = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_index]:
                    min_index = j
            x = self._extract(arr, min_index)
            self._insert(arr, i, x)
        return arr


class InsertionSort(SortingAlgorithm):
    def sort(self, arr: List[int]) -> List[int]:
        n = len(arr)
        arr = arr.copy()
        for i in range(1, n):
            key = arr[i]
            
            j = i - 1
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
            if j != i - 1:
                self.total_cost += self.extract_cost(i)
                self.total_cost += self.insert_cost(j + 1)
        return arr


class QuickSort(SortingAlgorithm):
    def sort(self, arr: List[int]) -> List[int]:
        arr = arr.copy()
        self._quick_sort(arr, 0, len(arr) - 1)
        return arr

    def _quick_sort(self, arr: List[int], low: int, high: int) -> None:
        if low < high:
            pi = self._partition(arr, low, high)
            self._quick_sort(arr, low, pi - 1)
            self._quick_sort(arr, pi + 1, high)

    def _partition(self, arr: List[int], low: int, high: int) -> int:
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] < pivot:
                i += 1
                self._swap(arr, i, j)
        self._swap(arr, i + 1, high)
        return i + 1
    

class ShadowSort(SortingAlgorithm):
    def sort(self, arr: List[int]) -> List[int]:
        arr = arr.copy()
        sorted_arr = sorted(arr)
        for i in range(len(arr)):
            if arr[i] != sorted_arr[i]:
                for j in range(i + 1, len(arr)):
                    if arr[j] == sorted_arr[i]:
                        ele = self._extract(arr, j)
                        self._insert(arr, i, ele)
                        break
        return arr


class DescSort(SortingAlgorithm):
    def sort(self, arr: List[int]) -> List[int]:
        arr = arr.copy()
        for i in range(len(arr)):
            max_lt = -1
            max_lt_idx = -1
            for j in range(i + 1, len(arr)):
                if arr[j] < arr[i] and arr[j] > max_lt:
                    max_lt = arr[j]
                    max_lt_idx = j
            if max_lt_idx != -1:
                self._extract(arr, max_lt_idx)
                self._insert(arr, i, max_lt)
        return arr
    

class MaxSort(SortingAlgorithm):
    def sort(self, arr: List[int]) -> List[int]:
        arr = arr.copy()
        i = 0
        while i < len(arr):
            found_lt = True
            insert_idx = i
            while found_lt:
                found_lt = False
                max_lt = -1 # -Inf
                max_idx = i
                for j in range(i + 1, len(arr)):
                    if arr[j] < arr[i] and arr[j] > max_lt:
                        found_lt = True
                        max_lt = max(max_lt, arr[j])
                        max_idx = j
                if found_lt:
                    self._extract(arr, max_idx)
                    self._insert(arr, insert_idx, max_lt)
                    i += 1
            i += 1
        return arr

def get_lis(arr: List[int]) -> Tuple[List[bool], int]:
    """
    Find longest non-decreasing subsequence in a given array. If there are multiple sequences, returning the one with
    the least start index. Return a binary array of the same length where 1 indicates the element is part of the longest
    non-decreasing subsequence.
    """

    n = len(arr)
    lis = [(1, -1)] * n
    for i in range(1, n):
        for j in range(i):
            if arr[i] >= arr[j]:
                if lis[i][0] < lis[j][0] + 1:
                    lis[i] = (lis[j][0] + 1, j)
    max_len = max(lis)[0]
    bool_lis = [False] * n
    idx = -1
    for i in range(n):
        if lis[i][0] == max_len:
            idx = i
            break
    while idx != -1:
        bool_lis[idx] = True
        idx = lis[idx][1]
    return bool_lis, max_len

class LisSort(SortingAlgorithm):

    def find_in_arr(self, arr: List[Tuple[int, int]], x: Tuple[int, int]) -> int:
        for i, ele in enumerate(arr):
            if ele == x:
                return i
        return -1
    
    def sort(self, arr: List[int]) -> List[int]:

        # find longest subsequence of non-decreasing elements
        is_lis, max_lis = get_lis(arr)

        arr_enum = [(x, i) for i, x in enumerate(arr)]
        lis = []
        non_lis = []
        for i, ele in enumerate(arr_enum):
            if is_lis[i]:
                lis.append(ele)
            else:
                non_lis.append(ele)
        non_lis.sort(reverse=True)
        done = [False] * len(non_lis)

        for lis_ele in reversed(lis):
            lis_idx = self.find_in_arr(arr_enum, lis_ele)

            # find all elements in non_lis that are > lis_ele (in descending order) and insert at lis_idx + 1
            for i, x in enumerate(non_lis):
                if x > lis_ele and not done[i]:
                    idx = self.find_in_arr(arr_enum, x)

                    if idx != lis_idx + 1:
                        self._extract(arr_enum, idx)
                        if idx < lis_idx:
                            lis_idx -= 1
                        self._insert(arr_enum, lis_idx + 1, x)
                        done[i] = True

        # find all elements <= lis[0] and insert at 0
        for i, x in enumerate(non_lis):
            if x <= lis[0] and not done[i]:
                idx = self.find_in_arr(arr_enum, x)
                self._extract(arr_enum, idx)
                self._insert(arr_enum, 0, x)
                done[i] = True
        return [x[0] for x in arr_enum]


def insert_cost(i: int) -> float:
    return i


def extract_cost(i: int) -> float:
    return i


def swap_cost(i: int, j: int) -> float:
    return extract_cost(i) + insert_cost(j) + extract_cost(j + 1) + insert_cost(i)


if __name__ == "__main__":
    random.seed(1)
    trials = 20 # per algorithm, per array length
    length = 10 ** 6 # max array length
    lengths = [10, 50, 100, 500, 1000, 5000, 10000]
    algorithms = [
        BubbleSort("bubble", swap_cost, extract_cost, insert_cost),
        SelectionSort("selection", swap_cost, extract_cost, insert_cost),
        InsertionSort("insertion", swap_cost, extract_cost, insert_cost),
        QuickSort("quick", swap_cost, extract_cost, insert_cost),
        # ShadowSort("shadow", swap_cost, extract_cost, insert_cost),
        # DescSort("desc", swap_cost, extract_cost, insert_cost),
        # MaxSort("max", swap_cost, extract_cost, insert_cost),
        LisSort("LIS", swap_cost, extract_cost, insert_cost),
    ]
    with open('scores.csv', 'w') as f:
        f.write('Algorithm,Length,Trial,Cost\n')
        for algorithm in algorithms:
            print(f"Running {algorithm.name}")
            for arr_length in lengths:
                print(f"Array length: {arr_length}")
                for trial in range(1, trials + 1):
                    print(f"Trial {trial}")
                    arr = [random.randint(1, arr_length) for _ in range(arr_length)]
                    # print(arr)

                    algorithm.total_cost = 0.0
                    algo_sorted = algorithm.sort(arr)
                    assert algo_sorted == sorted(arr), f"{algorithm.name} failed on {arr} with sorted order {algo_sorted}, expected {sorted(arr)}"

                    f.write(f"{algorithm.name},{arr_length},{trial},{algorithm.total_cost}\n")