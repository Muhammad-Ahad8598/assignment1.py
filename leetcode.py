# # Problem no 34 
# # Input:
# # nums = [2, 4, 4, 4, 8, 10]
# # target = 4

# # Output:
# # [1, 3]
# def searchRange(nums, target):
#     def findFirst():
#         left, right = 0, len(nums)-1
#         res = -1
#         while left <= right:
#             mid = (left + right) // 2
#             if nums[mid] == target:
#                 res = mid
#                 right = mid - 1  # keep going left
#             elif nums[mid] < target:
#                 left = mid + 1
#             else:
#                 right = mid - 1
#         return res

#     def findLast():
#         left, right = 0, len(nums)-1
#         res = -1
#         while left <= right:
#             mid = (left + right) // 2
#             if nums[mid] == target:
#                 res = mid
#                 left = mid + 1  # keep going right
#             elif nums[mid] < target:
#                 left = mid + 1
#             else:
#                 right = mid - 1
#         return res

#     return [findFirst(), findLast()]


# nums = [2, 2, 2, 4, 8, 10]
# target = 2
# print(searchRange(nums , target))

# # Find the first and last Position of an element in a sorted Array

# # def firstlast_postion(nums , target):
# #     left , right = 0,len(nums)  - 1 

# #     while left <= right:
# #         mid = left + (right - left) // 2 # divide the array into mid 

# #         if mid == target:
# #             return mid 
# #         elif mid < target:
# #            right = mid + 1 
# #            if right == target:
# #                return right
# #         else:
# #             left = mid - 1 
# #             if left == target:
# #                 return left



# def valid_parentheses(s):
#     stack = []
#     map = {')': '(', '}': '{', ']': '['}
#     for char in s:
#         if char in map:
#             result = stack.pop() if stack else '#'
#             if map[char] != result:
#               return False
#         else:
#             stack.append(char)
        
#     return not stack

# print(valid_parentheses("()[]"))  # True
# print(valid_parentheses("([)]"))    # False


# class Node:
#     def __init__(self, val):
#         self.val = val
#         self.next = None

#     def merge_list(self, l1, l2):
#         dummy = Node(0)
#         current = dummy

#         while l1 and l2:
#             if l1.val < l2.val:
#                 current.next = l1
#                 l1 = l1.next
#             else:
#                 current.next = l2
#                 l2 = l2.next
#             current = current.next

#         current.next = l1 if l1 else l2
#         return dummy.next

#     def lst_to_linked(self, lst):
#         dummy = Node(0)
#         current = dummy
#         for val in lst:
#             current.next = Node(val)
#             current = current.next
#         return dummy.next

#     def print_node(self, node):
#         while node:
#             print(node.val, end="->")
#             node = node.next
#         print("None")


# # Create an instance to call non-static methods
# solution = Node(0)

# l1 = solution.lst_to_linked([1, 2, 4])
# l2 = solution.lst_to_linked([1, 3, 4])
# result = solution.merge_list(l1, l2)
# solution.print_node(result)


# class Solution:
#     def mergeTwoLists(self, l1, l2):  # Renamed method to match expected name
#         dummy = Node(0)  # Create a dummy node to simplify the merging process
#         current = dummy  # Pointer to build the merged list

#         while l1 and l2:  # Traverse both lists
#             if l1.val < l2.val:
#                 current.next = l1
#                 l1 = l1.next
#             else:
#                 current.next = l2
#                 l2 = l2.next
#             current = current.next

#         # Attach the remaining elements of l1 or l2
#         current.next = l1 if l1 else l2
#         return dummy.next  # Return the merged list (excluding the dummy node)

# # Helper class and functions for testing
# class Node:
#     def __init__(self, val):
#         self.val = val
#         self.next = None

# def lst_to_linked(lst):
#     dummy = Node(0)
#     current = dummy
#     for val in lst:
#         current.next = Node(val)
#         current = current.next
#     return dummy.next

# def print_node(node):
#     while node:
#         print(node.val, end= "->")  
#         node = node.next
#     print("None")

# # Test the Solution class
# solution = Solution()
# l1 = lst_to_linked([])
# l2 = lst_to_linked([0])
# result = solution.mergeTwoLists(l1, l2)
# print_node(result)


# def palindrom(s):
#     # s = ''.join(c.lower() for c in s if c.isalnum())  # Remove non-alphanumeric & lowercase
#     s = ''.join(c.lower() for c in s if c.isalnum())
#     left = 0
#     right = len(s) - 1
#     while left<right:
#         if s[left] != s[right]:
#             return False
#         left += 1
#         right -= 1 
#     return True

# s = "mam"
# print(palindrom(s)) 


# def maxProfit(self, prices):
#     min_price = float('inf')
#     max_profit = 0
#     for price in prices:
    
#         if price<min_price:
#            # update the price if current price is minimum 
#             min_price = price
#         else:
#             # calculate the max profit to sale the stock 
#             profit = price - min_price
#             max_profit = max(max_profit, profit)

#     return max_profit


# prices = [7,6,4,3,1]

# print(maxProfit(prices))

# from collections import Counter

# class Solution(object):
#     def vaild_anagrams(self , s1,s2):
#         return Counter(s1) == Counter(s2)

# result = Solution()
# s1 = 'xyz'
# s2 = 'yzx'
# print(result.vaild_anagrams(s1 , s2))


# Step 1: Define the Node class
# class Node:
    # def __init__(self, data):
        # self.data = data  # Data stored in the node
        # self.next = None  # Reference to the next node (initially None)
# find the cycle by using hare a tortise 
    # def has_cycle(self,head):
        # slow = head
        # fast = head
        # while fast and fast.next:
            # slow = slow.next
            # fast = fast.next.next
            # if slow == fast:
            #   return True
        # return False
    
# result = Node()
# head = [3,2,0,-4], pos = 1
# print(result.has_cycle(head))

# class TreeNode:
#     def __init__(self, val, left=None , right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# class Solution:    
#     def depth(self, root: TreeNode) -> int:
#         if not root:
#             return 0
#         left = self.depth(root.left)
#         right = self.depth(root.right)
#         return 1 + max(left , right)

# # Helper to build tree from list (Level Order)
# from collections import deque
# def build_tree(values):
#     if not values:
#         return None
#     root = TreeNode(values[0])
#     queue = deque([root])
#     i = 1
#     while queue and i < len(values):
#         curr = queue.popleft()
#         if values[i] is not None:
#             curr.left = TreeNode(values[i])
#             queue.append(curr.left)
#         i += 1
#         if i < len(values) and values[i] is not None:
#             curr.right = TreeNode(values[i])
#             queue.append(curr.right)
#         i += 1
#     return root

# # Test
# result = Solution()
# tree = build_tree([1,None,2])
# print(result.depth(tree))  # Output: 3


# def binary_search(nums,target):
#     left = 0
#     right = len(nums) - 1
#     while left <= right :
#         mid = (left + right) // 2
#         if nums[mid] == target:
#             return mid
#         elif target < nums[mid] :
#             right = mid - 1
#         else:
#             left = mid + 1

#     return -1 
    

# nums = [-1,0,3,5,9,12]
# target = 2
# print(binary_search(nums , target))




# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# class Solution:
#     def invert_tree(self, root: TreeNode):
#         if not root:
#             return None
        
#         root.left, root.right = self.invert_tree(root.right), self.invert_tree(root.left)
#         return root

# # Helper function to convert list to binary tree
# def list_to_tree(arr, i=0):
#     if i >= len(arr) or arr[i] is None:
#         return None
    
#     node = TreeNode(arr[i])
#     node.left = list_to_tree(arr, 2 * i + 1)
#     node.right = list_to_tree(arr, 2 * i + 2)
    
#     return node

# # Helper function to convert tree to list (level order)
# def tree_to_list(root):
#     if not root:
#         return []
    
#     queue = [root]
#     result = []
    
#     while queue:
#         node = queue.pop(0)
#         if node:
#             result.append(node.val)
#             queue.append(node.left)
#             queue.append(node.right)
    
#     return result

# # Input
# root = list_to_tree([])

# # Invert tree
# solution = Solution()
# inverted_root = solution.invert_tree(root)

# # Output
# print(tree_to_list(inverted_root))  # Output: [2, 3, 1]


# class Solution(object):
#     def  singleNumber(self,nums):
#         result = 0 
#         for num in nums:
#             result ^= num
#         return result
    
# solution = Solution()
# nums = [1]
# print(solution.singleNumber(nums))


# class Node:
#     def __init__(self , val):
#         self.val = val
#         self.next = None
# class Solution:
#     def reverseList(self, head:Node):
#         current = head
#         prev = None
#         while current :
#             next_node = current.next  # store the next node
#             current.next = prev  # reverse the node
#             prev = current      # make prev node to currentNode
#             current = next_node # update the currentNode

#         return prev


# solution = Solution()
# head = [1,2,3,4,5]
# print(solution.reverseList(head))



# class Node:
#     def __init__(self, val):
#         self.val = val
#         self.next = None

# class Solution:
#     def reverseList(self, head: Node):
#         current = head
#         prev = None
#         while current:
#             next_node = current.next  # store the next node
#             current.next = prev  # reverse the node
#             prev = current  # make prev node to currentNode
#             current = next_node  # update the currentNode

#         return prev  # prev will be the new head

# # Helper function to create a linked list from a list
# def create_linked_list(values):
#     if not values:
#         return None
#     head = Node(values[0])
#     current = head
#     for val in values[1:]:
#         current.next = Node(val)
#         current = current.next
#     return head

# # Helper function to print a linked list
# def print_linked_list(head):
#     current = head
#     while current:
#         print(current.val, end=" -> ")
#         current = current.next
#     print("None")

# solution = Solution()
# head = create_linked_list([])  # Create the linked list
# reversed_head = solution.reverseList(head)  # Reverse the linked list
# print_linked_list(reversed_head)  # Print the reversed list


# class Slotuion:
# def majority(nums):
#         candidate = None
#         count = 0

#         for num in nums:
#             if count == 0:
#                 candidate = num
#             count += (1 if num == candidate else -1)
#         return candidate

# nums = [2,2,1,1,1,2,2]
# print(majority(nums))



# def missingNumber(nums):
#     max_num = max(nums)
#     min_num = min(nums)

#     for i in range(min_num , max_num +1):
#         # print(i , end="")
#      if i not in nums:
#         print(i, end="")
      
# class Solution:

#     def missingNumber(nums):
#         missing = len(nums)  # we will get the lenght 
#         for i, num in enumerate(nums):
#             missing ^= i ^ num  # XOR all indices and values
#         return missing
    
# result = Solution
    
# nums = [9,6,4,2,3,5,7,0,1]
# print(result.missingNumber(nums))





# def twoSum(nums , target):
#     hashmap = {}
#     for i , num in enumerate(nums):
#         diff = target - num
#         if diff in hashmap:
#             return [hashmap[diff],i]
#         hashmap[num] = i

#     return None

# nums = [5,7,9,2]
# target = 9
# print(twoSum(nums , target))



# def isValid(s):
#     stack = []
#     map = {')':'(' , ']':'[' , '}':'{'}
#     for char in s:
#         if char in map: 
#             result = stack.pop() if stack else None
#             if map[char] != result:
#                 return False
#         else:
#             stack.append(char)
#     return not stack


# print(isValid("([])"))  # True


# def meregeList(l1,l2):
#     dummy = Node(0)
#     current = dummy
#     while l1 and l2:
#         if l1.val < l2.val:
#             current.next = l1
#             l1 = l1.next
#         else:
#             current.next = l2
#             l2 = l2.next
#         current = current.next

#     current.next = l1 if l1 else l2
#     return dummy.next


# def palindrom(s):
#     s = ''.join(c.lower() for c in s if c.isalnum())
#     left = 0
#     right = len(s) - 1
#     while left < right:
#         if s[left] != s[right]:
#             return False
#         left += 1 
#         right -= 1
#     return True


# s = 'HEHE'
# print(palindrom(s))

# def maxProfit(prices):
#     min_price = float('inf')
#     max_profit = 0
#     for price in prices:
#         if price<min_price:
#             min_price = price
#         else:
#             profit = price - min_price
#             max_profit = max(profit , max_profit)
#     return max_profit

# prices = [1,2,3,4,5,6,7]
# print(maxProfit(prices))

# from collections import Counter

# def isAnagram(s,t):
#   return Counter(s) == Counter(t)

# s = 'HELLO'
# t = 'ELLOH'
# print(isAnagram(s,t))


# def hasCycle(head):
#     slow = fast = head
#     while fast and fast.next:
#         slow = slow.next
#         fast = fast.next.next
#         if slow == fast:
#             return True
#     return False



# def reverseString(s):
#     left = 0
#     right = len(s) - 1
#     while left < right:
#         s[left] , s[right] = s[right] , s[left]
#         left += 1
#         right -= 1

# s = ["H","a","n","n","a","h"]
# reverseString(s)
# print(s)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# class Solution:
#     def diameterOfBinaryTree(self, root):
#         self.diameter = 0

#         def dfs(node):
#             if not node:
#                 return 0
#             left = dfs(node.left)
#             right = dfs(node.right)
#             self.diameter = max(self.diameter, left + right)
#             return 1 + max(left, right)

#         dfs(root)
#         return self.diameter

# # Sample tree creation
# root = TreeNode(1)
# root.left = TreeNode(2)
# root.right = TreeNode(3)
# root.left.left = TreeNode(4)
# root.left.right = TreeNode(5)

# # Run the function
# sol = Solution()
# print(sol.diameterOfBinaryTree(root))  # Output: 3


# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# class Solution:
#     def sortedArrayToBST(self, nums):
#         if not nums:
#             return None
#         mid = len(nums) // 2
#         root = TreeNode(nums[mid])
#         root.left = self.sortedArrayToBST(nums[:mid])
#         root.right = self.sortedArrayToBST(nums[mid+1:])
#         return root

# def preOrderTraversal(root):
#     if root:
#         print(root.val, end=" ")
#         preOrderTraversal(root.left)
#         preOrderTraversal(root.right)

# # Example usage
# nums = [-10, -3, 0, 5, 9]
# sol = Solution()
# root = sol.sortedArrayToBST(nums)
# preOrderTraversal(root)


# def max_sum_sliding(arr, k):
#     max_sum = sum(arr[:k])  # Step 1: Compute sum of the first k elements
#     current_sum = max_sum    # Step 2: Initialize current sum with max_sum

#     for i in range(k, len(arr)):  # Step 3: Slide the window across the array
#         current_sum += arr[i] - arr[i - k]  # Step 4: Update the sum efficiently
#         max_sum = max(max_sum, current_sum)  # Step 5: Update max sum if needed

#     return max_sum  # Step 6: Return the maximum sum found

# print(max_sum_sliding([2, 1, 5, 1, 3, 2], 3))  # Output: 9




# class Solution(object):
#     def maxSubArray(self, nums , k):
#         current_sum = sum(nums[:k])
#         max_sum = current_sum
#         for i in range(k , len(nums)):
#            current_sum += nums[i] - nums[i-k]
#            max_sum = max(current_sum , max_sum)
#         return max_sum

# result = Solution()
# nums = [-2,1,-3,4,-1,2,1,-5,4]
# k= 6
# print(result.maxSubArray(nums,k))

# class Solution:
#     def maxSubArray(self, nums):
#         max_sum = float('-inf')  # Maximum sum shuru mein -infinity
#         current_sum = 0  # Current sum track karne ke liye
        
#         for num in nums:
#             current_sum = max(num, current_sum + num)  # Ya to naya element lein ya purana sum badhate jaayein
#             max_sum = max(max_sum, current_sum)  # Maximum sum update karein
        
#         return max_sum

# nums = [-2,1,-3,4,-1,2,1,-5,4]
# print(Solution().maxSubArray(nums))  # Output: 6



# class Solution:
#     def climbStait(self , n):
#         if n < 1:
#             return 1
#         first = 1 
#         second = 1

#         for _ in range(2 , n + 1):
#             first , second = second , first + second
        
#         return second
    
# result = Solution()
# n = 6
# print(result.climbStait(n))


# def isSymmetric(root):
#     if not root:
#         return None
#     def checkSymmetric(left, right):
#         if not left and not right:
#             return True
#         if not left or not right:
#             return False
        
#         return (left.val == right.val) and checkSymmetric(left.left , right.right) and checkSymmetric(left.right , right.left)
    
#     return checkSymmetric( root.left , root.right)






# class Solution:
#     def productExceptSelf(self, nums):
#         n = len(nums)  # Step 1: Get the length of the input array
#         answer = [1] * n  # Step 2: Initialize the answer array with 1's. This will store the result.

#         # Step 3: Calculate prefix products and store them in the answer array
#         prefix = 1  # Initialize a variable to store the cumulative product from the left.
#         for i in range(n):
#             answer[i] = prefix  # Store the current prefix product in answer[i]
#             prefix *= nums[i]  # Update the prefix by multiplying with the current number in nums

#         # Step 4: Calculate suffix products and multiply them with prefix products in the answer array
#         suffix = 1  # Initialize a variable to store the cumulative product from the right.
#         for i in range(n - 1, -1, -1):  # Iterate backwards (right to left)
#             answer[i] *= suffix  # Multiply the current answer[i] with the current suffix product
#             suffix *= nums[i]  # Update the suffix by multiplying with the current number in nums

#         return answer  # Step 5: Return the answer array which now holds the product of all elements except self.




# def productExceptSelf(nums):
#     output = [1]* len(nums)

#     prefix = 1
#     for i in range(len(nums)):
#         output[i] = prefix
#         prefix *= nums[i]

#     suffix = 1
#     for i in range(len(nums)-1, -1, -1):
#         output[i] *= suffix
#         suffix *= nums[i]

#     return output

# [24,12,8,6]
# nums = [-1,1,0,-3,3]

# print(productExceptSelf(nums))



# my solution 



# def maxProfit(prices):
#     min_price = float('-inf')
#     max_profit = 0

#     for price in prices:
#         if price < min_price:
#             min_price = price
#         if prices[price+1] > min_price:
#             profit = min_price - prices[price+1]
#         else:
#             prices[price+1] < min_price
#             min_price = prices[price+1]

#             max_profit += profit
#     return max_profit



# optimal solution 

# def maxProfit(prices):
#     min_price = float('inf')  # Set to infinity initially
#     max_profit = 0  # Initialize profit to 0

#     for price in prices:
#         if price < min_price:
#             min_price = price  # Found a new minimum price
#         else:
#             profit = price - min_price  # Calculate profit by selling at current price
#             if profit > max_profit:  # Update max_profit if current profit is greater
#                 max_profit = profit

#     return max_profit

# # Test the function with an example
# prices = [1, 7, 4, 9, 2, 5]
# print(maxProfit(prices))  # Output: 8

# second optimal solution 

# def maxProfit(prices):
#     profit = 0
#     for i in range(1, len(prices)):
#         if prices[i] > prices[i-1]:
#             profit += prices[i] - prices[i-1]
#     return profit

# def maxProfit(prices):
#     profit = 0
#     for i in range(1,len(prices)):
#         if prices[i] > prices[i-1]:
#             profit += prices[i] - prices[i-1]
#     return profit

# prices = [7,1,5,3,6,4]
# print(maxProfit(prices))


# def rob(nums):
#     prev1 = 0
#     prev2 = 0

#     for num in nums:
#         temp = prev1
#         prev1 = max(prev1, prev2 + num)
#         prev2 = temp
#     return prev1

# nums = [1,2,3,1]
# print(rob(nums))


# def isBSt(root, low = float('-inf'), high = float('inf')):
#     # base case
#     if not root:
#         return True
#       # validation for bst
#     if not (low < root.val < high):
#         return False
#     # recursively check the bst
#     return isBSt(root.left, low, root.val ) and isBSt(root.right, root.val, high)














