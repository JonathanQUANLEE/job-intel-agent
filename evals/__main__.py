"""评测模块入口：python -m evals 跑召回评测；python -m evals.decision 跑决策评测。

两层评测各管一段：
  evals           召回：该找到的岗位找到了吗（hit@k / MRR / 忠实度）
  evals.decision  决策：agent 行为对不对（追问时机 / 条件继承 / 排序 / 轨迹）
"""

from .run_eval import main

main()
