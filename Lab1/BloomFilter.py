import math
from bitarray import bitarray


class BloomFilter(object):

    def __init__(self, currency, number_expected_elements):
        self.currency = currency
        self.number_expected_elements = number_expected_elements

        self.size = round((self.number_expected_elements * math.log(self.currency)) / ((math.log(2)) ** 2))

        self.bloom_filter = bitarray(self.size)
        self.bloom_filter.setall(0)

        self.number_hash_functions = round((self.size / self.number_expected_elements) * math.log(2))

    def _hash(self, item, K):
        hash = 0
        for x in item:
            hash += ord(x) + K
        return hash % self.size

    def add_to_filter(self, item):
        for i in range(self.number_hash_functions):
            self.bloom_filter[self._hash(item, i)] = 1

    def check_is_not_in_filter(self, item):
        for i in range(self.number_hash_functions):
            if self.bloom_filter[self._hash(item, i)] == 0:
                return True
        return False
