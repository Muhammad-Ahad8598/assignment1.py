# class MinStack:
#     def __init__(self):
#         self.stack = []  # Main stack to store elements
#         self.min_stack = []  # Auxiliary stack to track minimums
#     def push(self, val:int) -> None:
#         self.stack.append(val)
#         if not self.min_stack or val <= self.min_stack[-1]:
#             self.min_stack.append(val)
#     def pop(self) -> None:
#         if self.stack:
#             val = self.stack.pop()
#             if val == self.min_stack[-1]:
#                 self.min_stack.pop()
#     def top(self):
#         if self.stack:
#             return self.stack[-1]
#         raise  IndexError("Stack is empty")
#     def get_min(self):
#         if self.min_stack:
#             return self.min_stack[-1]
#         raise  IndexError("Min_Stack is empty")
    

# result = MinStack()
# result.push(-2)
# result.push(0)
# result.push(-3)
# print(result.get_min())
# result.pop()
# print(result.top())
# print(result.get_min())



# from collections import Counter

# def containsDuplicate(nums):
#     count = Counter(nums)

#     return any(v>=2 for v in count.values())

# nums = [1,2,3,2,2,4]
# print(containsDuplicate(nums))


# def containsDuplicate(nums):
#     value  = set()
#     for num in nums:
#         if num in value:
#             return True
#         value.add(num)
#     return False

# nums = [1,2,3,4]
# print(containsDuplicate(nums))






# def kthSmallest(root, k):
#     count = 0
#     result = None

#     def inorder(node):
#         nonlocal count, result
#         if not node and result is not None:
#             return
#         inorder(node.left)
#         count += 1
#         if count == k:
#             result = node.val
#             return 
#         inorder(node.right)
#     inorder(root)
#     return result


# intervals = [[1,3],[2,6],[8,10],[15,18]]
# def merge(intervals):
#     if not intervals:
#         return []
#     intervals.sort(key = lambda x: x[0])
#     merged= [intervals[0]]
#     for current in intervals[1:]:
#         last = merged[-1] 
#     if current[0] <= last[1]:
#         last[1] = max(current[1], last[1])
#     else:
#         merged.append(current)
#     return merged



# def twoSum(nums,k):
#     hashmap = {}
#     for i,num in enumerate(nums):
#         diff = k - num
#         if diff in hashmap:
#             return [hashmap[diff] , i]
#         hashmap[num]= i 


# nums = [1,2,3,4]
# k = 7
# print(twoSum(nums, k))

# def isValid(s):
#     stack = []
#     mapping = {')':'(', ']': '[', '}':'{'}
#     for char in s:
#         if char in mapping:
#             top = stack.pop() if stack else '#'
#             if mapping[char] != top:
#                 return False
#         else:
#             stack.append(char)
#     return not stack


# s = "{}{}[]()"
# print(isValid(s))


# def mergeList(l1, l2):
#     dummy = listnode()
#     current = dummy
#     while l1 and l2:
#         if l1.val < l2.val:
#             current.next = l1
#             l1 = l1.next
#         else:
#             current.next =  l2
#             l2 = l2.next
#         current = current.next
#     current = l1 or l2
#     return dummy.next


def maxProfit(prices):
    minPrice = float('inf')
    maxProfit = 0
    for price in prices:
        if price < minPrice:
            minPrice = price
        maxProfit = max(maxProfit , price - minPrice)
    return maxProfit