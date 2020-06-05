'''
#################################### Dynamic Programming Part II #########################
### This part covers DP problems from Backpack problems

416. Partition Equal Subset Sum
494. Target Sum
474. Ones and Zeroes
322. Coin Change
518. Coin Change 2
139. Word Break
377. Combination Sum IV
309. Best Time to Buy and Sell Stock with Cooldown
714. Best Time to Buy and Sell Stock with Transaction Fee
123. Best Time to Buy and Sell Stock III
188. Best Time to Buy and Sell Stock IV
583. Delete Operation for Two Strings
72. Edit Distance
650. 2 Keys Keyboard (can be a math problem)

'''


############################################## 0-1 Backpack Problems #####
# Personally, I think this one is hard 
'''
0-1 背包
有一个容量为 N 的背包，要用这个背包装下物品的价值最大，这些物品有两个属性：体积 w 和价值 v。

定义一个二维数组 dp 存储最大价值，其中 dp[i][j] 表示前 i 件物品体积不超过 j 的情况下能达到的最大价值。设第 i 件物品体积为 w，价值为 v，根据第 i 件物品是否添加到背包中，可以分两种情况讨论：

第 i 件物品没添加到背包，总体积不超过 j 的前 i 件物品的最大价值就是总体积不超过 j 的前 i-1 件物品的最大价值，dp[i][j] = dp[i-1][j]。
第 i 件物品添加到背包中，dp[i][j] = dp[i-1][j-w] + v。
第 i 件物品可添加也可以不添加，取决于哪种情况下最大价值更大。因此，0-1 背包的状态转移方程为：



// W 为背包总体积
// N 为物品数量
// weights 数组存储 N 个物品的重量
// values 数组存储 N 个物品的价值
public int knapsack(int W, int N, int[] weights, int[] values) {
    int[][] dp = new int[N + 1][W + 1];
    for (int i = 1; i <= N; i++) {
        int w = weights[i - 1], v = values[i - 1];
        for (int j = 1; j <= W; j++) {
            if (j >= w) {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - w] + v);
            } else {
                dp[i][j] = dp[i - 1][j];
            }
        }
    }
    return dp[N][W];
}
空间优化

在程序实现时可以对 0-1 背包做优化。观察状态转移方程可以知道，前 i 件物品的状态仅与前 i-1 件物品的状态有关，因此可以将 dp 定义为一维数组，其中 dp[j] 既可以表示 dp[i-1][j] 也可以表示 dp[i][j]。此时，



因为 dp[j-w] 表示 dp[i-1][j-w]，因此不能先求 dp[i][j-w]，防止将 dp[i-1][j-w] 覆盖。也就是说要先计算 dp[i][j] 再计算 dp[i][j-w]，在程序实现时需要按倒序来循环求解。

public int knapsack(int W, int N, int[] weights, int[] values) {
    int[] dp = new int[W + 1];
    for (int i = 1; i <= N; i++) {
        int w = weights[i - 1], v = values[i - 1];
        for (int j = W; j >= 1; j--) {
            if (j >= w) {
                dp[j] = Math.max(dp[j], dp[j - w] + v);
            }
        }
    }
    return dp[W];
}
无法使用贪心算法的解释

0-1 背包问题无法使用贪心算法来求解，也就是说不能按照先添加性价比最高的物品来达到最优，这是因为这种方式可能造成背包空间的浪费，从而无法达到最优。考虑下面的物品和一个容量为 5 的背包，如果先添加物品 0 再添加物品 1，那么只能存放的价值为 16，浪费了大小为 2 的空间。最优的方式是存放物品 1 和物品 2，价值为 22.

id	w	v	v/w
0	1	6	6
1	2	10	5
2	3	12	4
变种

完全背包：物品数量为无限个

多重背包：物品数量有限制

多维费用背包：物品不仅有重量，还有体积，同时考虑这两种限制

其它：物品之间相互约束或者依赖
'''


'''
416. Partition Equal Subset Sum
Medium

Given a non-empty array containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

Note:

Each of the array element will not exceed 100.
The array size will not exceed 200.
 

Example 1:

Input: [1, 5, 11, 5]

Output: true

Explanation: The array can be partitioned as [1, 5, 5] and [11].
 

Example 2:

Input: [1, 2, 3, 5]

Output: false

Explanation: The array cannot be partitioned into equal sum subsets.
'''

# This is a backpack problem 
# Greedy solution won't work, as for ex: [5,4,3,3,3]. Using greedy, 5 and 4 will be separated into two sets,
#	so it won't work

# Time: O(M*N), where M is len of nums, N is the target value
# Space: O(M*N).
# We can actually optimize the space usage to O(N) by using a dynamic list instead of a 2-D one for dp
def canPartition(nums: [int]):
	if len(nums) == 0:
		return True
	if len(nums) == 1 or sum(nums) % 2 != 0:
		return False
	target = sum(nums) // 2
	# dp[i][s] means with first i numbers, if we can achieve sum of s	
	dp = [[False] * (target + 1) for _ in range(len(nums) + 1)]
	for i in range(len(nums) + 1):
		dp[i][0] = True 
	
	# dp[i][s] can be true if:
	#	1. dp[i-1][s] is true, which means the target is achieved before the ith number is counted
	#	2. dp[i-1][s - nums[i-1]] is true, which means the target can be achieved with the 
	#        addition of the i th number (the reason we have i-1 is because we are using 
	#        1-base for i th number. Therefore the i th number will be index of i - 1)
	for i in range(1, len(nums) + 1):
		for s in range(1, target + 1):
			dp[i][s] = dp[i - 1][s] or dp[i - 1][s - nums[i - 1]]
	return dp[len(nums)][target]

print(canPartition([1, 5, 11, 5]))
print(canPartition([1, 3, 11, 5]))
print(canPartition([1, 2, 3, 5]))


'''
494. Target Sum
Medium

You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols + and -. For each integer, you should choose one from + and - as its new symbol.

Find out how many ways to assign symbols to make sum of integers equal to target S.

Example 1:
Input: nums is [1, 1, 1, 1, 1], S is 3. 
Output: 5
Explanation: 

-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3

There are 5 ways to assign symbols to make the sum of nums be target 3.
Note:
The length of the given array is positive and will not exceed 20.
The sum of elements in the given array will not exceed 1000.
Your output answer is guaranteed to be fitted in a 32-bit integer.
'''
# Backpack DP problem	0-1 Knapsack
# Lets have dp[i][j] to represent using numbers until index i, we have dp[i][j] ways to achieve sum of j - 1000
# Time: O(MN)  Space: O(MN) We can actually optimize the space complexity to O(N) as we only need 1 row
#	above the current row to calculate all the dp cells. 
def findTargetSumWays(nums: [int], S: int):
	numsTotal = sum(nums)
	if len(nums) == 0 or (numsTotal - S) % 2 != 0:
		return 0
	
	dp = [[0] * (numsTotal * 2 + 1) for _ in range(len(nums))]
	dp[0][nums[0] + numsTotal] += 1
	dp[0][-nums[0] + numsTotal] += 1
	for i in range(1, len(nums)):
		for j in range(-numsTotal, numsTotal + 1):
			if dp[i - 1][j + numsTotal] > 0:
				# we can performe a + operation
				dp[i][j + numsTotal + nums[i]] += dp[i - 1][j + numsTotal]
				# or a - operation
				dp[i][j + numsTotal - nums[i]] += dp[i - 1][j + numsTotal]
	return dp[-1][S + numsTotal]
	
print(findTargetSumWays([1, 1, 1, 1, 1], 3))


'''
474. Ones and Zeroes
Medium

In the computer world, use restricted resource you have to generate maximum benefit is what we always want to pursue.
For now, suppose you are a dominator of m 0s and n 1s respectively. On the other hand, there is an array with strings consisting of only 0s and 1s.

Now your task is to find the maximum number of strings that you can form with given m 0s and n 1s. Each 0 and 1 can be used at most once.

Note:
The given numbers of 0s and 1s will both not exceed 100
The size of given string array won't exceed 600.
 

Example 1:
Input: Array = {"10", "0001", "111001", "1", "0"}, m = 5, n = 3
Output: 4

Explanation: This are totally 4 strings can be formed by the using of 5 0s and 3 1s, which are “10,”0001”,”1”,”0”
 
Example 2:
Input: Array = {"10", "0", "1"}, m = 1, n = 1
Output: 2

Explanation: You could form "10", but then you'd have nothing left. Better form "0" and "1".
'''

# DP   0-1 Knapsack
# Time: O(I*M*N) where I is max length of each string, M is m, N is n 
# Space: O(M*N)
def findMaxForm(strs: [str], m: int, n: int):
	
	# helper function to calculate the number of 0's and 1's in a string
	def count01(s):
		one = 0
		zero = 0
		for i in s:
			if i == '1':
				one += 1
			else: 
				zero += 1
		return (zero, one)

	# dp[i][j] means with m 0's and n 1's, how many strings we can form
	dp = [[0] * (n + 1) for _ in range(m + 1)]
	
	for s in strs:
		zero, one = count01(s)
		for i in range(m, -1, -1):
			for j in range(n, -1, -1):
				# dp of current equals dp of m n exclude the usage of the current string, + 1
				remainingM = i - zero
				remainingN = j - one
				if remainingM >= 0 and remainingN >= 0:
					dp[i][j] = max(dp[i][j], dp[remainingM][remainingN] + 1)
	
	return dp[m][n]

print(findMaxForm(["10", "0001", "111001", "1", "0"], 5, 3))


'''
322. Coin Change
Medium

You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

Example 1:

Input: coins = [1, 2, 5], amount = 11
Output: 3 
Explanation: 11 = 5 + 5 + 1
Example 2:

Input: coins = [2], amount = 3
Output: -1
Note:
You may assume that you have an infinite number of each kind of coin.
'''
# DP Knapsack problem 
# Time: O(M*N)  Space: O(N)
def coinChange(coins: [int], amount: int):
	# dp[i] means: in order to get to the amount of i, we need dp[i] coins
	dp = [0] + [float('inf')] * amount

	for coin in coins: 
		for i in range(coin, amount + 1):
			dp[i] = min(dp[i], dp[i - coin] + 1)

	return dp[-1] if dp[-1] != float('inf') else -1

print(coinChange([1, 2, 5], 11))
print(coinChange([2], 3))


'''
518. Coin Change 2
Medium

You are given coins of different denominations and a total amount of money. Write a function to compute the number of combinations that make up that amount. You may assume that you have infinite number of each kind of coin.

Example 1:

Input: amount = 5, coins = [1, 2, 5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
Example 2:

Input: amount = 3, coins = [2]
Output: 0
Explanation: the amount of 3 cannot be made up just with coins of 2.
Example 3:

Input: amount = 10, coins = [10] 
Output: 1
 

Note:

You can assume that:

0 <= amount <= 5000
1 <= coin <= 5000
the number of coins is less than 500
the answer is guaranteed to fit into signed 32-bit integer
'''

# DP Knapsack
# time: O(M*N)  Space: O(N)
def change(amount: int, coins: [int]):
	# dp[i] means: we have dp[i] different ways to combine to sum of i
	dp = [1] + [0] * (amount)

	for coin in coins:
		for i in range(coin, amount + 1):
			dp[i] = dp[i - coin] + dp[i]
	
	return dp[-1]

print(change(5, [1, 2, 5]))
print(change(3, [2]))


'''
139. Word Break
Medium

Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words.

Note:

The same word in the dictionary may be reused multiple times in the segmentation.
You may assume the dictionary does not contain duplicate words.
Example 1:

Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
Example 2:

Input: s = "applepenapple", wordDict = ["apple", "pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
             Note that you are allowed to reuse a dictionary word.
Example 3:

Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
Output: false
'''
##### Full Knapsack with sequence #####

# DP Knapsack problem
# Time: O(M*N)  Space: O(N)
def wordBreak(s: str, wordDict: [str]):
	# dp[i] means if part of string s can be formed with the wordDict. i represents the first i characters 
	# 	being used in the string
	dp = [True] + [False] * len(s)

	# As we can apply one word multiple times, we need to put the loop of words in the inner loop
	for i in range(1, len(s) + 1):
		for word in wordDict:
			l = len(word)
			if i >= l and s[i - l : i] == word:
				dp[i] = dp[i - l] or dp[i]

	return dp[-1]

print(wordBreak("applepenapple", ["apple","pen"]))


'''
377. Combination Sum IV
Medium

Given an integer array with all positive numbers and no duplicates, find the number of possible combinations that add up to a positive integer target.

Example:

nums = [1, 2, 3]
target = 4

The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)

Note that different sequences are counted as different combinations.

Therefore the output is 7.
 

Follow up:
What if negative numbers are allowed in the given array?
How does it change the problem?
What limitation we need to add to the question to allow negative numbers?

Credits:
Special thanks to @pbrother for adding this problem and creating all test cases.
'''
##### Full Knapsack with sequence #####

# The difference between this question and the coin change 2 problem, is that this one requires counting
# of different sequences of result. For ex, (1, 1, 2) and (2, 1, 1) will be different choices.

# To solve this, we pu the loop on knapsack items(nums) in the inner loop

# DP Knapsack
# time: O(M*N)  Space: O(N)
def combinationSum4(nums: [int], target: int):
	# dp[i] means: we have dp[i] different ways to combine to sum of i
	dp = [1] + [0] * (target)

	# Put loop on knapsack items in the inner loop
	for i in range(1, target + 1):
		for n in nums:
			if i >= n:
				dp[i] = dp[i - n] + dp[i]
	
	return dp[-1]

print(combinationSum4([1, 2, 5], 5))
print(combinationSum4([2], 3))


##################################### Stocks Trading ##############

'''
309. Best Time to Buy and Sell Stock with Cooldown
Medium

Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times) with the following restrictions:

You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)
Example:

Input: [1,2,3,0,2]
Output: 3 
Explanation: transactions = [buy, sell, cooldown, buy, sell]
'''

# DP problem
# Time: O(N)  Space: O(1) we simplify the 1-D dp array to using constant number of variables

def maxProfit(prices: [int]):
	# We have 3 states: 
	#	sNotHold: represents the state when we are not holding a stock and do not have a cooldown
	#	sHold: represents the state when we are holding a stock
	#	sNotHoldWithCooldown: represents the state when we are not holding a stock, and have a cooldown
	
	# We have 5 edges:
	#	sNotHold 			 --- Buy  ---> 		sHold
	#	sNotHold 			 --- Rest ---> 		sNotHold
	#	sHold    			 --- Sell --->		sNotHoldWithCooldown
	#	sHold    			 --- Rest --->		sHold
	#	sNotHoldWithCooldown --- Rest --->		sNotHold

	# Initial value of sNotHold should be 0
	# Initial value of sHold and sNotHoldWithCooldown should be float('-inf') as it is not possible
	sNotHold, sHold, sNotHoldWithCooldown = 0, float('-inf'), float('-inf')

	for p in prices:
		sNotHold, sHold, sNotHoldWithCooldown = max(sNotHold, sNotHoldWithCooldown), max(sHold, sNotHold - p), sHold + p

	return max(sNotHold, sNotHoldWithCooldown) 

print('309. Best Time to Buy and Sell Stock with Cooldown')
print(maxProfit([1,2,3,0,2]))


'''
714. Best Time to Buy and Sell Stock with Transaction Fee
Medium

Your are given an array of integers prices, for which the i-th element is the price of a given stock on day i; and a non-negative integer fee representing a transaction fee.

You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction. You may not buy more than 1 share of a stock at a time (ie. you must sell the stock share before you buy again.)

Return the maximum profit you can make.

Example 1:
Input: prices = [1, 3, 2, 8, 4, 9], fee = 2
Output: 8
Explanation: The maximum profit can be achieved by:
Buying at prices[0] = 1
Selling at prices[3] = 8
Buying at prices[4] = 4
Selling at prices[5] = 9
The total profit is ((8 - 1) - 2) + ((9 - 4) - 2) = 8.
Note:

0 < prices.length <= 50000.
0 < prices[i] < 50000.
0 <= fee < 50000.
'''

# DP
# Time: O(N)  Space: O(1)
def maxProfit(prices: [int], fee: int):
	# we have two states: 
	#	sHold: holding a stock, initial value should be -infinity as it is not possible initially
	#	sNotHold: not holding a stock, initial value should be 0

	# 4 edges:
	# sHold   --- sell, and pay transaction fee --->    sNotHold
	# sHold   			--- rest ---> 					sHold
	# sNotHold 			--- buy ---> 					sHold 
	# sNotHold 			--- rest ---> 					sNotHold

	sHold, sNotHold = float('-inf'), 0

	for p in prices:
		sHold, sNotHold = max(sHold, sNotHold - p), max(sNotHold, sHold + p - fee) 

	return sNotHold

print(maxProfit([1, 3, 2, 8, 4, 9], 2))


'''
123. Best Time to Buy and Sell Stock III
hard

Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete at most two transactions.

Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).

Example 1:

Input: [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
             Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
Example 2:

Input: [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
             Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are
             engaging multiple transactions at the same time. You must sell before buying again.
Example 3:

Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
'''

# dp
# Time: O(N)  Space: O(1) as we optimized the space usage from using a full list of dp  

def maxProfit(prices: [int]):
	# We have 5 states: 
	#	t1Hold: holding the first stock bought
	#	t1NotHold: not holding the first stock yet
	#	t2Hold: holding the second stock bought
	#	t2NotHold: not holding the second stock yet
	#	t2Complete: sold the 2nd stock

	# We have 9 edges:
	# t1NotHold --- rest ---> t1NotHold
	# t1NotHold --- buy ---> t1Hold
	# t1Hold --- rest ---> t1Hold
	# t1Hold --- sell ---> t2NotHold
	# t2NotHold --- rest ---> t2NotHold
	# t2NotHold --- buy ---> t2Hold
	# t2Hold --- rest ---> t2Hold
	# t2Hold --- sell ---> t2Complete
	# t2Complete --- rest ---> t2Complete

	t1Hold, t1NotHold, t2Hold, t2NotHold, t2Complete = float('-inf'), 0, float('-inf'), float('-inf'), float('-inf')

	for p in prices:
		t1Hold, t1NotHold = max(t1NotHold - p, t1Hold), t1NotHold
		t2Hold, t2NotHold, t2Complete = max(t2NotHold - p, t2Hold), max(t1Hold + p, t2NotHold), max(t2Hold + p, t2Complete)
	
	return max(t2Complete, t2NotHold) if max(t2Complete, t2NotHold) > 0 else 0

print(maxProfit([3,3,5,0,0,3,1,4]))
print(maxProfit([1,2,3,4,5]))
print(maxProfit([7,6,4,3,1]))


'''
188. Best Time to Buy and Sell Stock IV
Hard

Say you have an array for which the i-th element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete at most k transactions.

Note:
You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

Example 1:

Input: [2,4,1], k = 2
Output: 2
Explanation: Buy on day 1 (price = 2) and sell on day 2 (price = 4), profit = 4-2 = 2.
Example 2:

Input: [3,2,6,5,0,3], k = 2
Output: 7
Explanation: Buy on day 2 (price = 2) and sell on day 3 (price = 6), profit = 6-2 = 4.
             Then buy on day 5 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.		 
'''
# DP 
# this question is similar to '123. Best Time to Buy and Sell Stock III', which allows 2 transactions
# The only difference is that we need to add a loop to loop through all k transactions

# at each transaction i, hold[i] is the max profit until we at the state of holding the ith transaction
# and notHold[i] is the max profit we can get until we are not holding the ith transaction
# states and edges are all similar to question 123, except being dynamically generated.

# Time: O(L*k) where M is length of prices, and k is k 
# Space: O(k) as we need two arrays hold and notHold of length k
def maxProfitKTransaction(k: int, prices: [int]):
	# simple solution
	def unlimitTransactionSolution():
		max_profit = 0
		for i in range(len(prices) - 1):
			max_profit += max(prices[i+1] - prices[i], 0)
		return max_profit

	if k == 0:
		return 0
	# Edge case when k is too big. We can simply solve this problem with the simpliest solution 
	if k >= len(prices) // 2: 
		return unlimitTransactionSolution()

	hold = [float('-inf')] * k
	notHold = [float('-inf')] * (k + 1)
	notHold[0] = 0

	for p in prices:
		# the hold value of first transaction need to be separated as notHold[0] should not be changed
		hold[0] = max(notHold[0] - p, hold[0])
		for i in range(1, k):
			hold[i], notHold[i] = max(notHold[i] - p, hold[i]), max(hold[i - 1] + p, notHold[i])
		# the last notHold value is special as it is to sell the last holded item
		notHold[-1] = max(hold[-1] + p, notHold[-1])

	result = max(notHold)
	return result if result > 0 else 0

print('188. Best Time to Buy and Sell Stock IV')
print(maxProfitKTransaction(2, [3,2,6,5,0,3]))
print(maxProfitKTransaction(2, [3,3,5,0,0,3,1,4]))


############################################################ Editing Strings ############

'''
583. Delete Operation for Two Strings
Medium

Given two words word1 and word2, find the minimum number of steps required to make word1 and word2 the same, where in each step you can delete one character in either string.

Example 1:
Input: "sea", "eat"
Output: 2
Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".
Note:
The length of given words won't exceed 500.
Characters in given words can only be lower-case letters.
'''

# DP 
# Time: O(M*N)  Space: O(N)  M is len of word1, N is len of word2
def minDistance(word1: str, word2: str):
	l1 = len(word1)
	l2 = len(word2)

	# dp[i][j] is the minDistance between two words until the i and j th char 
	# However, we can optimize the dp to a 1-D dp as we only access dp[i - 1]
	dp = [x for x in range(l2 + 1)]
	for i in range(1, l1 + 1):
		tmp = dp[:]
		for j in range(l2 + 1):
			if j == 0:
				tmp[j] = i 
			elif word1[i - 1] == word2[j - 1]:
				tmp[j] = dp[j - 1] 			# dp[i][j] = dp[i-1][j-1]  if same char is met
			else:
				tmp[j] = min(tmp[j - 1], dp[j]) + 1  # dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + 1  if not
		dp = tmp
	return dp[-1]

print(minDistance("sea", "eat"))


'''
72. Edit Distance
Hard

Given two words word1 and word2, find the minimum number of operations required to convert word1 to word2.

You have the following 3 operations permitted on a word:

Insert a character
Delete a character
Replace a character

Example 1:
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

Example 2:
Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')
'''

# DP 

# to calculate dp[i][j], since we can either add, remove, or replace a character, we have 3 possibilities:
# if word1[i] == word2[j]:
#	dp[i][j] = min(dp[i-1][j-1], 	  --> this is if we don't need to do anything
#				   dp[i-1][j] + 1,    --> this is if we need to add a char at word1[i]
#				   dp[i][j-1] + 1)    --> this is if we need to remove a char at word1[i]
# if word1[i] != word2[j]:
#	dp[i][j] = min(dp[i-1][j-1] + 1,  --> this is if we need to replace a char at word1[i]
#				   dp[i-1][j] + 1,    --> this is if we need to add a char at word1[i]
#				   dp[i][j-1] + 1)    --> this is if we need to remove a char at word1[i]
# from above we can see that the only difference is that when word1[i] == word2[j], we don't need to add one
# 	if we get the result from dp[i-1][j-1]

# Time: O(M*N)  Space: O(M*N) can be optimized into O(N) as we only need the 1 row above the current one

def minDistance(word1: str, word2: str):
	dp = [[float('inf')] * (len(word2) + 1) for _ in range(len(word1) + 1)]

	for i in range(len(dp)):
		for j in range(len(dp[0])):
			if i == 0:
				dp[i][j] = j
			elif j == 0:
				dp[i][j] = i
			elif word1[i-1] == word2[j-1]:
				dp[i][j] = min(dp[i-1][j-1], dp[i-1][j] + 1, dp[i][j-1] + 1)
			else:
				dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1

	return dp[-1][-1]

print('72. Edit Distance')
print(minDistance("horse", "ros"))


'''
650. 2 Keys Keyboard
Medium

Initially on a notepad only one character 'A' is present. You can perform two operations on this notepad for each step:

Copy All: You can copy all the characters present on the notepad (partial copy is not allowed).
Paste: You can paste the characters which are copied last time.
 

Given a number n. You have to get exactly n 'A' on the notepad by performing the minimum number of steps permitted. Output the minimum number of steps to get n 'A'.

Example 1:

Input: 3
Output: 3
Explanation:
Intitally, we have one character 'A'.
In step 1, we use Copy All operation.
In step 2, we use Paste operation to get 'AA'.
In step 3, we use Paste operation to get 'AAA'.
 

Note:

The n will be in the range [1, 1000].
'''
# This is more like a Math problem
# Our moves will be in a pattern of something like 'CPPCPPPPCP...' so we can split it into groups 'CPP', 'CPPP', 'CP', ...
# 	with length g_1 = 3, g_2 = 4, g_3 = 2, ...

# We can find out that combining group1 and group2, we can get g_1 * g_2 number of A's. 
# Therefore, the total number of A's is g_1*g_2*g_3*... , and if one group length is composite, say g_i = p * q, we can
# 	again split that group into two smaller groups of length p and q, as p + q is always smaller than p * q 

# This question is essentially to find the prime factorization of N

# Time: O(n) worst case.  Space: O(1)
def minSteps(n: int):
	result = 0
	d = 2
	while n > 1:
		while n % d == 0:
			result += d
			n /= d
		d += 1
	return result

print(minSteps(3))