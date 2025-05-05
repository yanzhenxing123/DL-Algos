from collections import defaultdict
class SKU:
    def __init__(self, sku_id, token_id):
        self.sku_id = sku_id
        self.token_id = token_id
        self.level = None  # 用来记录 SKU 的层级

def hierarchical_grouping_objs(skus, group_size):
    """
    层次聚类后得到不同sku_objs
    """

    def group_by_level(objs, level):
        groups = defaultdict(list)
        for sku in objs:
            cat = tuple(sku.token_id[:level + 1])
            sku.level = level + 1
            groups[cat].append(sku)
        return groups

    def recursive_group(objs, level):
        if (len(objs) <= group_size and level >= 2) or level >= 4:
            return [objs]

        result = []
        grouped = group_by_level(objs, level)
        for group_skus in grouped.values():
            result.extend(recursive_group(group_skus, level + 1))
        return result

    """
    tuple([sku.sku_id for sku in group])
    tuple([sku.token_id for sku in group])
    和加cache没关系
    """

    groups = recursive_group(skus, 0)
    return groups

if __name__ == '__main__':
    # 创建 SKU 对象
    skus = [
        # SKU(sku_id=1, token_id=(1, 0, 3)),
        # SKU(sku_id=2, token_id=(1, 0, 3)),
        SKU(sku_id=3, token_id=(1, 1, 3)),
        SKU(sku_id=4, token_id=(2, 0, 3)),
        SKU(sku_id=5, token_id=(2, 1, 3)),
    ]

    # 调用 hierarchical_grouping_objs
    grouped_skus = hierarchical_grouping_objs(skus, group_size=2)

    # 输出结果
    for group in grouped_skus:
        print([sku.sku_id for sku in group])


