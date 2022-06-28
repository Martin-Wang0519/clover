class Solution(object):
    def findTheWinner(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: int
        """
        if n == 1:
            return 1
        next = n if (k + 1) % n == 0 else (k + 1) % n
        ans = (next + self.findTheWinner(n - 1, k) - 1) % n if (next + self.findTheWinner(n - 1, k) - 1) % n != 0 \
            else n
        return ans


if __name__ == '__main__':
    a = Solution()
    print(a.findTheWinner(5,2))
