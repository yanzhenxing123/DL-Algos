from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right



def build_tree(input_list):
    """
    根据完全二叉树数组表示法构建二叉树，数组下标 i 的左右孩子为 2*i+1 和 2*i+2。
    """
    if not input_list:
        return None

    nodes = []
    for val in input_list:
        if val == 'null' or val is None:
            nodes.append(None)
        else:
            nodes.append(TreeNode(int(val)))

    for idx, node in enumerate(nodes):
        if node is None:
            continue
        left_idx = 2 * idx + 1
        right_idx = left_idx + 1
        if left_idx < len(nodes):
            node.left = nodes[left_idx]
        if right_idx < len(nodes):
            node.right = nodes[right_idx]

    return nodes[0]

def find_max_subtree(root):
    """
    找到“子树和最大的子树”，并返回其根节点。
    在计算过程中，如果某个子树的和为负，会将该子树裁剪掉。
    """
    max_sum = float('-inf')
    max_subtree_root = None

    def postorder(node):
        nonlocal max_sum, max_subtree_root
        if not node:
            return 0
        left_sum = postorder(node.left)
        if left_sum < 0:
            node.left = None
            left_sum = 0
        right_sum = postorder(node.right)
        if right_sum < 0:
            node.right = None
            right_sum = 0
        current_sum = node.val + left_sum + right_sum
        if current_sum > max_sum:
            max_sum = current_sum
            max_subtree_root = node
        return current_sum

    postorder(root)
    return max_subtree_root


def level_order_traversal(root):
    if not root:
        return []
    levels = []
    current_level = [root]  # 记录当前层的所有节点（含 null）
    while True:
        level_vals = []
        next_level = []
        has_non_null = False  # 标记当前层是否有非 null 节点
        for node in current_level:
            if node:
                level_vals.append(node.val)
                next_level.append(node.left)
                next_level.append(node.right)
                has_non_null = True
            else:
                level_vals.append('null')
                next_level.append(None)  # null 节点的左右孩子仍记为 null
                next_level.append(None)
        levels.append(level_vals)
        if not has_non_null:
            break  # 若当前层全为 null，停止遍历
        current_level = next_level

    # 合并所有层级，并移除最后一层全 null 节点
    result = []
    for level in levels[:-1]:  # 最后一层全是 null，直接忽略
        result.extend(level)
    while result and result[-1] == 'null':
        result.pop()
    return result



# input_str = input()
# input_list = input_str.replace('[', '').replace(']', '').split(',')
#
# root1 = build_tree(input_list)
# max_subtree1 = find_max_subtree(root1)
# output1 = level_order_traversal(max_subtree1)
# output_str = '[' + ','.join(map(str, output1)) + ']'
# print(output_str)

# input1 = ['3', '2', '5']
# root1 = build_tree(input1)
# max_subtree1 = find_max_subtree(root1)
# output1 = level_order_traversal(max_subtree1)
# print(output1)
#
#
#
#
# input1 = ['-5', '-1', '3', 'null', 'null', '4', '7']
# root1 = build_tree(input1)
# max_subtree1 = find_max_subtree(root1)
# output1 = level_order_traversal(max_subtree1)
# print(output1)







input1 = [-1,'null',1,'null','null',-1,-1,'null','null','null','null',2,1,-3,-1,'null','null','null','null','null','null','null','null',2,1,3,8]
root1 = build_tree(input1)
max_subtree_root = find_max_subtree(root1)
output1 = level_order_traversal(max_subtree_root)
print(output1)

