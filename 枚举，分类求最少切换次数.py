import math
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds


# =========================================================
# 0. 已知数据：10种工件轴向占用长度
# =========================================================

# 这里填你们已经算出来的轴向占用长度，单位：mm
lengths = np.array([
    191.547,
    148.826,
    191.547,
    191.547,
    75.000,
    150.000,
    250.000,
    399.924,
    181.033,
    200.000
])

# 每种工件需求量，问题一中都是50件
demand = np.array([50] * 10)

# 标准母材长度，单位：mm
standard_bars = [9000, 10000, 11000, 12000]

# 快速模式：先快速找可行下料方案并统计切换次数。
# 如果必须严格证明“全局最少切换次数”，改成 False，但运行时间会明显变长。
fast_feasible_mode = True
time_limit_per_combination = 3


# =========================================================
# 1. 枚举所有总长度为 target_total 的母材组合
# =========================================================

def enumerate_bar_combinations(target_total):
    """
    枚举所有满足：
    9000*n9 + 10000*n10 + 11000*n11 + 12000*n12 = target_total
    的非负整数解。
    """

    combinations = []

    # 最大根数不需要太大。
    # target_total=99000时，最多也就是11根9m。
    max_count = target_total // min(standard_bars) + 2

    for n9 in range(max_count + 1):
        for n10 in range(max_count + 1):
            for n11 in range(max_count + 1):
                remain = target_total - 9000 * n9 - 10000 * n10 - 11000 * n11

                if remain < 0 or remain % 12000 != 0:
                    continue

                n12 = remain // 12000

                bar_list = (
                    [9000] * n9 +
                    [10000] * n10 +
                    [11000] * n11 +
                    [12000] * n12
                )

                combinations.append({
                    "9m数量": n9,
                    "10m数量": n10,
                    "11m数量": n11,
                    "12m数量": n12,
                    "母材根数": len(bar_list),
                    "总长度/mm": target_total,
                    "母材列表": sorted(bar_list)
                })

    return combinations


def theoretical_min_switch(lengths, demand, target_total):
    """
    理论下界：
    第i类工件总长度如果超过最长母材，就至少要分到多根母材上。
    每根被使用的母材至少出现1类工件，因此：
    最少切换次数 >= 最少出现次数总和 - 母材根数。
    """

    min_type_appearances = int(
        sum(math.ceil(lengths[i] * demand[i] / max(standard_bars)) for i in range(len(lengths)))
    )
    min_bar_count = math.ceil(target_total / max(standard_bars))

    return max(0, min_type_appearances - min_bar_count)


# =========================================================
# 2. 对某一种母材组合，求该组合下的最小切换次数
# =========================================================

def solve_one_combination(lengths, demand, bars, time_limit=3, incumbent_switch=None, fast_mode=True):
    """
    给定一种母材组合 bars，例如：
    [10000, 11000, 11000, ..., 12000]

    求：
    每根母材放哪些工件，才能在满足需求和容量约束下，
    让总切换次数尽可能少。

    决策变量：
    x[k,i]：第k根母材上第i种工件的数量
    z[k,i]：第k根母材上是否出现第i种工件

    核心思想：
    若一根母材上出现的工件种类越少，
    按同类连续加工时切换次数就越少。
    """

    num_bars = len(bars)
    num_types = len(lengths)
    total_waste = float(np.sum(bars) - np.dot(lengths, demand))

    # x变量数量：num_bars * num_types
    # z变量数量：num_bars * num_types
    num_x = num_bars * num_types
    num_z = num_bars * num_types
    num_vars = num_x + num_z

    def xid(k, i):
        """x[k,i]在一维变量中的编号"""
        return k * num_types + i

    def zid(k, i):
        """z[k,i]在一维变量中的编号"""
        return num_x + k * num_types + i

    # -----------------------------------------------------
    # 目标函数：
    # 精确模式：min sum z[k,i]，用于尽量减少切换次数；
    # 快速模式：目标函数全为0，只求可行方案，避免MILP长时间证明最优。
    # -----------------------------------------------------

    c = np.zeros(num_vars)

    if not fast_mode:
        for k in range(num_bars):
            for i in range(num_types):
                c[zid(k, i)] = 1

    # -----------------------------------------------------
    # 变量类型：全部是整数
    # x是非负整数，z是0-1整数
    # -----------------------------------------------------

    integrality = np.ones(num_vars)

    # -----------------------------------------------------
    # 变量上下界
    # -----------------------------------------------------

    lb = np.zeros(num_vars)
    ub = np.zeros(num_vars)

    # x[k,i] 最多不超过该类工件需求量，也不超过当前母材能容纳的最大件数
    for k in range(num_bars):
        for i in range(num_types):
            ub[xid(k, i)] = min(demand[i], math.floor(bars[k] / lengths[i]))

    # z[k,i] 只能是0或1
    for k in range(num_bars):
        for i in range(num_types):
            ub[zid(k, i)] = 1

    A_list = []
    lower_list = []
    upper_list = []

    # -----------------------------------------------------
    # 约束1：每种工件总数量必须满足需求
    # sum_k x[k,i] = demand[i]
    # -----------------------------------------------------

    A_demand = np.zeros((num_types, num_vars))

    for i in range(num_types):
        for k in range(num_bars):
            A_demand[i, xid(k, i)] = 1

    A_list.append(A_demand)
    lower_list.extend(demand)
    upper_list.extend(demand)

    # -----------------------------------------------------
    # 约束2：每根母材不能超长
    # sum_i length[i] * x[k,i] <= bars[k]
    # -----------------------------------------------------

    A_capacity = np.zeros((num_bars, num_vars))

    for k in range(num_bars):
        for i in range(num_types):
            A_capacity[k, xid(k, i)] = lengths[i]

    A_list.append(A_capacity)
    lower_list.extend(np.array(bars) - total_waste)
    upper_list.extend(bars)

    # -----------------------------------------------------
    # 约束3：x 和 z 的关系
    # x[k,i] <= ub[x[k,i]] * z[k,i]
    #
    # 如果 z[k,i]=0，则 x[k,i] 必须是0；
    # 如果 x[k,i]>0，则 z[k,i] 必须是1。
    # -----------------------------------------------------

    A_link_upper = np.zeros((num_bars * num_types, num_vars))

    row = 0
    for k in range(num_bars):
        for i in range(num_types):
            A_link_upper[row, xid(k, i)] = 1
            A_link_upper[row, zid(k, i)] = -ub[xid(k, i)]
            row += 1

    A_list.append(A_link_upper)
    lower_list.extend([-np.inf] * (num_bars * num_types))
    upper_list.extend([0] * (num_bars * num_types))

    # z[k,i] <= x[k,i]
    # 如果z为1，则该类工件在这根母材上至少切1件，减少无意义的搜索分支。
    A_link_lower = np.zeros((num_bars * num_types, num_vars))

    row = 0
    for k in range(num_bars):
        for i in range(num_types):
            A_link_lower[row, xid(k, i)] = 1
            A_link_lower[row, zid(k, i)] = -1
            row += 1

    A_list.append(A_link_lower)
    lower_list.extend([0] * (num_bars * num_types))
    upper_list.extend([np.inf] * (num_bars * num_types))

    if incumbent_switch is not None:
        # 只寻找比当前最好方案更少切换的解；找不到时可快速判定不需要更新。
        A_incumbent = np.zeros((1, num_vars))
        for k in range(num_bars):
            for i in range(num_types):
                A_incumbent[0, zid(k, i)] = 1

        max_z = incumbent_switch + num_bars - 1
        A_list.append(A_incumbent)
        lower_list.append(-np.inf)
        upper_list.append(max_z)

    # -----------------------------------------------------
    # 合并约束
    # -----------------------------------------------------

    A = np.vstack(A_list)
    lower = np.array(lower_list, dtype=float)
    upper = np.array(upper_list, dtype=float)

    constraints = LinearConstraint(A, lower, upper)
    bounds = Bounds(lb, ub)

    # -----------------------------------------------------
    # 求解整数规划
    # -----------------------------------------------------

    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit": time_limit,
            "mip_rel_gap": 0
        }
    )

    if res.x is None:
        return None

    sol = np.rint(res.x).astype(int)

    x = sol[:num_x].reshape(num_bars, num_types)
    z = sol[num_x:].reshape(num_bars, num_types)

    # 计算每根母材使用长度和剩余长度
    used_lengths = x @ lengths
    remains = np.array(bars) - used_lengths

    # 如果一根母材用了 t 种工件，则切换次数是 t-1
    # 总切换次数 = sum_k(max(0, 使用种类数-1))
    switch_counts = []

    for k in range(num_bars):
        type_count = int(z[k].sum())
        switch_counts.append(max(0, type_count - 1))

    total_switch = int(sum(switch_counts))

    return {
        "x": x,
        "z": z,
        "used_lengths": used_lengths,
        "remains": remains,
        "switch_counts": switch_counts,
        "total_switch": total_switch,
        "objective_value": res.fun,
        "solver_success": res.success,
        "solver_message": res.message
    }


# =========================================================
# 3. 搜索所有99000mm组合，找切换次数最少的方案
# =========================================================

def search_best_plan(lengths, demand):
    total_workpiece_length = float(np.dot(lengths, demand))

    # 因为标准母材都是1000mm整数倍，
    # 所以不小于总工件长度的最小1000倍数就是理论最小母材总长度。
    target_total = math.ceil(total_workpiece_length / 1000) * 1000

    print("=" * 60)
    print("全部工件总长度/mm：", round(total_workpiece_length, 3))
    print("理论最小母材总长度/mm：", target_total)
    print("总体利用率/%：", round(total_workpiece_length / target_total * 100, 4))
    print("=" * 60)

    combinations = enumerate_bar_combinations(target_total)
    lower_bound_switch = theoretical_min_switch(lengths, demand, target_total)

    print(f"一共枚举到 {len(combinations)} 种总长度为 {target_total} mm 的母材组合。")
    print(f"理论最少切换次数下界：{lower_bound_switch}")

    best = None
    search_records = []

    combinations = sorted(enumerate(combinations, start=1), key=lambda item: item[1]["母材根数"])

    for idx, comb in combinations:
        bars = np.array(comb["母材列表"], dtype=float)

        sol = solve_one_combination(
            lengths=lengths,
            demand=demand,
            bars=bars,
            time_limit=time_limit_per_combination,
            incumbent_switch=None if best is None else best["solution"]["total_switch"],
            fast_mode=fast_feasible_mode
        )

        if sol is None:
            search_records.append({
                "组合编号": idx,
                "9m数量": comb["9m数量"],
                "10m数量": comb["10m数量"],
                "11m数量": comb["11m数量"],
                "12m数量": comb["12m数量"],
                "母材根数": comb["母材根数"],
                "是否可行": "未找到/超时" if fast_feasible_mode else "否",
                "总切换次数": None
            })
            continue

        print(
            f"组合{idx} 可行：母材根数={comb['母材根数']}，"
            f"切换次数={sol['total_switch']}"
        )

        search_records.append({
            "组合编号": idx,
            "9m数量": comb["9m数量"],
            "10m数量": comb["10m数量"],
            "11m数量": comb["11m数量"],
            "12m数量": comb["12m数量"],
            "母材根数": comb["母材根数"],
            "是否可行": "是",
            "总切换次数": sol["total_switch"]
        })

        candidate = {
            "combo_index": idx,
            "combination": comb,
            "bars": bars,
            "solution": sol
        }

        if best is None:
            best = candidate
        else:
            # 第一比较：总切换次数越少越好
            if sol["total_switch"] < best["solution"]["total_switch"]:
                best = candidate

            # 如果切换次数相同，可以再比较母材根数，根数少一点更便于管理
            elif sol["total_switch"] == best["solution"]["total_switch"]:
                if comb["母材根数"] < best["combination"]["母材根数"]:
                    best = candidate

        if best["solution"]["total_switch"] == lower_bound_switch:
            print(f"已达到理论下界 {lower_bound_switch}，提前停止搜索。")
            break

    search_df = pd.DataFrame(search_records)

    return best, search_df, total_workpiece_length, target_total


# =========================================================
# 4. 根据最优方案生成下料表和逐件切割顺序
# =========================================================

def build_result_tables(best, lengths, demand, total_workpiece_length, target_total, search_df):
    x = best["solution"]["x"]
    bars = best["bars"]
    switch_counts = best["solution"]["switch_counts"]

    plan_rows = []
    detail_rows = []

    num_bars, num_types = x.shape

    for k in range(num_bars):
        bar_id = f"M{k + 1}"
        bar_length = bars[k]

        used_length = float(np.dot(x[k], lengths))
        remain = bar_length - used_length

        used_types = [i for i in range(num_types) if x[k, i] > 0]

        combo_parts = []
        order_parts = []

        current_position = 0.0
        sequence = 1

        # 同类连续加工：按工件编号从小到大排
        for i in used_types:
            count = int(x[k, i])

            combo_parts.append(f"圆管{i + 1}×{count}")
            order_parts.append(f"圆管{i + 1}×{count}")

            for _ in range(count):
                start = current_position
                end = start + lengths[i]

                detail_rows.append({
                    "母材编号": bar_id,
                    "母材长度/mm": int(bar_length),
                    "母材内序号": sequence,
                    "工件编号": i + 1,
                    "工件长度/mm": round(lengths[i], 3),
                    "起始位置/mm": round(start, 3),
                    "终止位置/mm": round(end, 3)
                })

                current_position = end
                sequence += 1

        plan_rows.append({
            "母材编号": bar_id,
            "母材长度/mm": int(bar_length),
            "工件组合": "，".join(combo_parts),
            "加工顺序": " → ".join(order_parts),
            "占用长度/mm": round(used_length, 3),
            "剩余长度/mm": round(remain, 3),
            "单根利用率/%": round(used_length / bar_length * 100, 4),
            "切换次数": switch_counts[k]
        })

    plan_df = pd.DataFrame(plan_rows)
    detail_df = pd.DataFrame(detail_rows)

    combination = best["combination"]

    summary_df = pd.DataFrame([
        {"指标": "求解模式", "数值": "快速可行方案" if fast_feasible_mode else "精确最优"},
        {"指标": "全部工件总长度/mm", "数值": round(total_workpiece_length, 3)},
        {"指标": "母材总长度/mm", "数值": target_total},
        {"指标": "总剩余长度/mm", "数值": round(target_total - total_workpiece_length, 3)},
        {"指标": "总体利用率/%", "数值": round(total_workpiece_length / target_total * 100, 4)},
        {"指标": "总切换次数", "数值": best["solution"]["total_switch"]},
        {"指标": "9m数量", "数值": combination["9m数量"]},
        {"指标": "10m数量", "数值": combination["10m数量"]},
        {"指标": "11m数量", "数值": combination["11m数量"]},
        {"指标": "12m数量", "数值": combination["12m数量"]},
        {"指标": "母材根数", "数值": combination["母材根数"]},
        {"指标": "推荐组合编号" if fast_feasible_mode else "最优组合编号", "数值": best["combo_index"]}
    ])

    return summary_df, plan_df, detail_df


# =========================================================
# 5. 主程序
# =========================================================

def main():
    best, search_df, total_workpiece_length, target_total = search_best_plan(lengths, demand)

    if best is None:
        print("在理论最小母材总长度下没有找到可行方案。")
        print("需要把母材总长度增加1000mm后重新搜索。")
        return

    summary_df, plan_df, detail_df = build_result_tables(
        best=best,
        lengths=lengths,
        demand=demand,
        total_workpiece_length=total_workpiece_length,
        target_total=target_total,
        search_df=search_df
    )

    print("\n最优方案汇总：")
    print(summary_df)

    print("\n母材下料方案：")
    print(plan_df)

    output_file = "问题一_母材组合与切换次数优化.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="汇总", index=False)
        search_df.to_excel(writer, sheet_name="组合搜索结果", index=False)
        plan_df.to_excel(writer, sheet_name="母材下料方案", index=False)
        detail_df.to_excel(writer, sheet_name="逐件切割顺序", index=False)

    print(f"\n结果已保存为：{output_file}")


if __name__ == "__main__":
    main()
