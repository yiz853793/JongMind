# JongMind

麻将 AI 实验项目。当前核心是一个按雀魂四人麻将规则推进的 Python 发牌姬/裁判进程。

## 已实现

- 136 张牌，含 3 张赤宝牌：`0m`、`0p`、`0s`。
- 14 张王牌、岭上牌、宝牌指示牌、杠宝牌指示牌。
- 东家 14 张起手，南/西/北 13 张起手。
- 四个玩家进程通过 `multiprocessing.Queue` 与发牌姬通信。
- 摸牌、打牌、响应阶段、吃、碰、明杠、暗杠、加杠。
- 荣和、自摸、双/三家荣和的响应裁决。
- 立直宣言、立直棒、一发状态。
- 九种九牌、四风连打、四家立直、四杠散了、荒牌流局。
- 荒牌听牌罚符。
- 役种、符番、赤宝牌、宝牌和点数计算通过 `mahjong` 库处理。
- 打牌模型放在独立的 `jongmind_ai/` 包，当前包含随机弃牌模型、只看向听数模型、牌效模型和神经网络 beginner 模型。

## 架构边界

- `jongmind/game.py`：公共游戏类型和状态视图，例如 `Seat`、`Phase`、`DealerCommand`、`DealerState`。
- `jongmind/dealer.py`：单局裁判/状态机，负责规则推进、合法动作、和牌与流局结算。
- `jongmind/runtime.py`：进程运行时和 dealer/player 消息广播，负责队列通信边界。
- `jongmind_ai/`：AI 模型、特征编码、模型注册表和训练奖励函数。新增模型优先放这里。
- `jongmind/train_policy_random.py`：当前强化学习训练入口，负责对局采样、奖励回传、checkpoint 保存和早停。
- `jongmind/evaluate_models.py`：模型评估入口，负责固定/随机座次的东风战或半庄对局统计。

新增模型的最小路径：

```python
from jongmind_ai import DiscardDecision, register_model


class MyAgent:
    def choose_discard(self, hand, state):
        return DiscardDecision(tile=hand[0], shanten=0, ukeire=0, reason="example")


register_model("my_agent", lambda seed: MyAgent())
```

## 当前边界

这些复杂边界还需要继续精修：

- 抢杠还没有接入加杠响应窗口。
- 食替限制、振听、同巡振听还没有完整实现。
- 流局满贯、包牌、完整半庄连庄/终局判断还没有实现。
- 荒牌听牌已支持副露后的剩余暗手牌形判断；极少数役种/特殊形边界还需要继续补规则测试。

## 命令示例

```python
DealerCommand(kind="draw", seat=Seat.SOUTH)
DealerCommand(kind="discard", seat=Seat.EAST, tile="5m")
DealerCommand(kind="discard", seat=Seat.EAST, tile="9m", riichi=True)
DealerCommand(kind="pass", seat=Seat.SOUTH)
DealerCommand(kind="win", seat=Seat.SOUTH, action="ron")
DealerCommand(kind="win", seat=Seat.EAST, action="tsumo")
DealerCommand(kind="call", seat=Seat.WEST, action="pon", tiles=("5m", "5m"))
DealerCommand(kind="call", seat=Seat.SOUTH, action="chii", tiles=("3m", "4m"))
DealerCommand(kind="call", seat=Seat.NORTH, action="kan", tiles=("5p", "5p", "5p"))
DealerCommand(kind="closed_kan", seat=Seat.EAST, tiles=("1m", "1m", "1m", "1m"))
DealerCommand(kind="added_kan", seat=Seat.EAST, tile="5m")
DealerCommand(kind="abortive_draw", seat=Seat.NORTH)
```

## 运行

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

从随机权重开始，用真实对局奖励训练 `neural_beginner` 对抗 3 个随机玩家：

```powershell
python -m jongmind.train_policy_random --matches 200 --match-type east --target-seat rotate --checkpoint outputs/checkpoints/neural_beginner.pt
```

`--seed` 可以传整数，也可以传 `random` 让程序启动时生成并打印本次基础 seed：

```powershell
python -m jongmind.train_policy_random --matches 200 --seed random --target-seat random --checkpoint outputs/checkpoints/neural_beginner.pt
```

训练器也支持单独指定模型/策略随机种子，`--seed` 控制牌山、对手和座次，`--planner-seed` 控制模型初始化和采样弃牌：

```powershell
python -m jongmind.train_policy_random --matches 200 --seed random --planner-seed random --target-seat random --checkpoint outputs/checkpoints/neural_beginner.pt
```

新手期训练器默认使用和牌优先的三档奖励：一档形状奖励使用同手牌相对弃牌质量，二档单局终局奖励中和牌信号最重，三档整场顺位/点棒奖励较轻。也可以手动调节：

```powershell
python -m jongmind.train_policy_random --matches 200 --shape-scale 1.0 --terminal-scale 1.0 --match-scale 0.03 --checkpoint outputs/checkpoints/neural_beginner.pt
```

训练器默认允许 `neural_beginner` 在反应阶段自己选择 `pass/chii/pon/kan`，并把鸣牌网络一起保存进 checkpoint。若要临时退回门清训练，可加 `--disable-calls`；若想降低/提高鸣牌的局部形状奖励，可调 `--call-shape-scale`。

对随机对手稳定后，可以混入更强对手，并把形状奖励降权，让模型更重视预期点数、实际和牌点数和无役听牌惩罚：

```powershell
python -m jongmind.train_policy_random --matches 1000 --init-checkpoint outputs/checkpoints/neural_beginner.pt --checkpoint outputs/checkpoints/neural_beginner.pt --seed random --planner-seed random --match-type east --target-seat random --opponents tile_efficiency,shanten,open_call --learning-rate 0.0002 --entropy-coef 0.02 --temperature 1.0 --shape-scale 0.5 --call-shape-scale 0.02 --terminal-scale 1.2 --match-scale 0.03 --opponent-win-penalty -0.5 --shape-ce-coef 0.05
```

可以给长训练加早停。早停默认关闭；下面这个例子要求最近 20 场同时满足平均顺位、一位率、四位率、平均点、和牌率等阈值才会保存并停止：

```powershell
python -m jongmind.train_policy_random --matches 1000 --init-checkpoint outputs/checkpoints/neural_beginner.pt --checkpoint outputs/checkpoints/neural_beginner.pt --seed random --planner-seed random --match-type east --target-seat random --opponents tile_efficiency,random,random --learning-rate 0.0002 --entropy-coef 0.02 --temperature 1.0 --shape-scale 0.5 --call-shape-scale 0.02 --terminal-scale 1.2 --match-scale 0.03 --opponent-win-penalty -0.5 --shape-ce-coef 0.05 --early-stop-window 20 --early-stop-max-avg-rank 1.6 --early-stop-min-first-rate 0.65 --early-stop-max-fourth-rate 0.10 --early-stop-min-avg-score 28000 --early-stop-min-win-rate 0.08
```

加入高手后如果三四位和被飞增多，可以提高防守相关惩罚：`--deal-in-penalty` 控制放铳痛感，`--bust-penalty` 控制被飞惩罚，`--placement-rewards` 控制顺位奖惩，`--defense-scale` 控制有人立直/明显开手时危险弃牌的小步惩罚。

如需从已有 checkpoint 继续强化学习：

```powershell
python -m jongmind.train_policy_random --matches 200 --init-checkpoint outputs/checkpoints/neural_beginner.pt --checkpoint outputs/checkpoints/neural_beginner.pt
```

训练后评估：

```powershell
python -m jongmind.evaluate_models --target neural_beginner --opponents random,random,random --matches 100 --match-type east --target-seat rotate
```

更严格的半庄评估可以随机开局座次，之后按东一到南四正常轮庄：

```powershell
python -m jongmind.evaluate_models --target neural_beginner --opponents tile_efficiency,shanten,random --matches 100 --match-type south --target-seat random
```

运行测试：

```powershell
python -m unittest discover -v
```

## 五阶段训练课程

这条路线先让 `neural_beginner` 模仿稳定老师，再用真实对局奖励微调；每个阶段都保留 JSONL 日志，方便回看顺位、和牌率、放铳率、副露率和训练错误。

### 可选老师模型

当前可以直接在 `--teacher`、`--call-teacher`、`--teacher-prior`、`--opponents` 和评估命令里选择这些老师：

- `tile_efficiency`：门清牌效老师，只看向听和有效牌，适合最初入门。
- `tile_efficiency_call`：牌效鸣牌基线，适合做旧版对照，不建议作为最终老师。
- `mjai_manue`：参考 mjai-manue 的“和牌概率/平均打点/危险度/放铳损失”思路，偏弃牌 EV。
- `akochan`：参考 Akochan 的局收支、顺位和攻守权衡，防守和场况更重，适合中后期验证。
- `mahjong_ai`：参考 MahjongAI 的规则启发式，重视役牌、断幺、染手、门清保护和鸣牌阈值，适合做鸣牌老师。
- `expected_value_call`：`mjai_manue` 的兼容别名。

这些老师不是外部项目源码的移植，而是按公开思路在本项目里重新实现的可运行 profile。推荐顺序是先用 `tile_efficiency` 学会基本弃牌，再用 `mjai_manue` 学软弃牌分布，用 `mahjong_ai` 学鸣牌，最后把 `akochan` 放进评估或 teacher-prior 里检查攻守是否更稳。

进阶模仿命令示例：

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; python -m jongmind.train_imitation `
  --teacher mjai_manue `
  --discard-target soft `
  --discard-temperature 0.45 `
  --train-call-teacher `
  --call-teacher mahjong_ai `
  --matches 1000 `
  --match-type east `
  --target-seat random `
  --opponents random,random,shanten `
  --seed random `
  --planner-seed random `
  --checkpoint outputs/checkpoints/neural_beginner.pt `
  --output-log outputs/stage1b_mjai_manue_mahjong_ai.jsonl `
  --learning-rate 8e-5 `
  --batch-size 128 `
  --save-every 50
```

### 阶段 1：模仿学习预训练

用 `tile_efficiency` 做老师，先学基本弃牌；同时用偏置较强的 pass 监督初始化反应网络，避免 RL 一开始乱鸣牌：

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; python -m jongmind.train_imitation `
  --teacher tile_efficiency `
  --matches 2000 `
  --match-type east `
  --target-seat rotate `
  --opponents random,random,shanten `
  --seed random `
  --planner-seed random `
  --checkpoint outputs/checkpoints/neural_beginner.pt `
  --output-log outputs/stage1_imitation.jsonl `
  --train-call-pass `
  --call-pass-bias 3.0 `
  --call-pass-weight 0.25 `
  --learning-rate 1e-4 `
  --batch-size 128 `
  --save-every 50
```

阶段目标：`discard_acc` 稳定上升、训练无 error；如果 call 头过于保守，下一轮降低 `--call-pass-bias` 或 `--call-pass-weight`。

### 阶段 2：模仿学习后评估

先只评估，不更新权重。对随机桌看基本生存能力，对混合桌看是否已经接近老师策略：

```powershell
python -m jongmind.evaluate_models --target neural_beginner --opponents random,random,random --matches 200 --match-type east --target-seat rotate --output-dir outputs/eval_stage2_random
python -m jongmind.evaluate_models --target neural_beginner --opponents tile_efficiency,shanten,random --matches 200 --match-type east --target-seat random --output-dir outputs/eval_stage2_mixed
```

阶段目标：错误数为 0，平均顺位和四位率不要明显劣于 `tile_efficiency` 桌中的随机对手；若副露率异常高，回到阶段 1 增强 pass 监督。

### 阶段 3：从模仿模型开始 RL 微调

从模仿 checkpoint 载入，重置优化器，用较小学习率和较高探索熵进入真实奖励训练：

```powershell
python -m jongmind.train_policy_random `
  --matches 1000 `
  --init-checkpoint outputs/checkpoints/neural_beginner.pt `
  --checkpoint outputs/checkpoints/neural_beginner.pt `
  --reset-optimizer `
  --seed random `
  --planner-seed random `
  --match-type east `
  --target-seat random `
  --opponents random,random,shanten `
  --learning-rate 0.0002 `
  --entropy-coef 0.02 `
  --temperature 1.0 `
  --shape-scale 0.7 `
  --call-shape-scale 0.03 `
  --terminal-scale 1.2 `
  --match-scale 0.03 `
  --opponent-win-penalty -0.5 `
  --teacher-prior mjai_manue `
  --teacher-prior-weight-start 0.25 `
  --teacher-prior-weight-end 0.05 `
  --teacher-prior-decay-matches 600 `
  --teacher-prior-temperature 0.50 `
  --expected-value-weight 0.0025 `
  --teacher-reaction-ce-coef 0.15 `
  --call-action-penalty 0.08 `
  --shape-ce-coef 0.05 `
  --output-log outputs/stage3_rl_from_imitation.jsonl `
  --review-log outputs/stage3_decisions.jsonl
```

阶段目标：和牌率、平均点和平均顺位逐步改善，同时裸验证副露率不要持续抬升；如果副露率超过 25%-30%，优先提高 `--teacher-reaction-ce-coef` 或 `--call-action-penalty`，不要继续沿用已经乱鸣的 checkpoint。

### 阶段 4：逐渐提高对手强度

不要一次把桌子拉满，按课程逐步加压。每档都可以配合 `--validation-every` 做固定验证，过线后再进入下一档：

```powershell
python -m jongmind.train_policy_random `
  --matches 1500 `
  --init-checkpoint outputs/checkpoints/neural_beginner.pt `
  --checkpoint outputs/checkpoints/neural_beginner.pt `
  --seed random `
  --planner-seed random `
  --match-type east `
  --target-seat random `
  --opponents tile_efficiency,random,random `
  --learning-rate 0.0002 `
  --entropy-coef 0.02 `
  --shape-scale 0.5 `
  --call-shape-scale 0.02 `
  --terminal-scale 1.2 `
  --match-scale 0.03 `
  --validation-every 100 `
  --validation-matches 100 `
  --validation-opponents tile_efficiency,shanten,random `
  --validation-precheck-window 20 `
  --early-stop-window 30 `
  --early-stop-max-avg-rank 2.2 `
  --early-stop-min-first-rate 0.30 `
  --early-stop-max-fourth-rate 0.25 `
  --early-stop-min-avg-score 25000 `
  --early-stop-min-win-rate 0.08 `
  --early-stop-max-open-rate 0.25 `
  --validation-max-open-rate 0.25 `
  --output-log outputs/stage4_curriculum.jsonl
```

下一档把 `--opponents` 换成 `mjai_manue,shanten,random`，再换成 `akochan,mjai_manue,random`。若三四位率升高，优先调高 `--deal-in-penalty`、降低 `--temperature`，再考虑加大训练场数。

### 阶段 5：南风场与攻守训练

半庄会放大点棒、连庄和防守选择的影响。进入南风场后提高防守权重与放铳/被飞惩罚，并用半庄验证集早停：

```powershell
python -m jongmind.train_policy_random `
  --matches 2000 `
  --init-checkpoint outputs/checkpoints/neural_beginner.pt `
  --checkpoint outputs/checkpoints/neural_beginner.pt `
  --seed random `
  --planner-seed random `
  --match-type south `
  --target-seat random `
  --opponents tile_efficiency,shanten,open_call `
  --learning-rate 0.00015 `
  --entropy-coef 0.015 `
  --temperature 0.9 `
  --shape-scale 0.4 `
  --call-shape-scale 0.015 `
  --defense-scale 0.6 `
  --terminal-scale 1.4 `
  --match-scale 0.05 `
  --deal-in-penalty -10 `
  --bust-penalty -25 `
  --placement-rewards 2,0,-3,-8 `
  --validation-every 100 `
  --validation-matches 100 `
  --validation-match-type south `
  --validation-opponents tile_efficiency,shanten,open_call `
  --validation-target-seat random `
  --validation-precheck-window 20 `
  --validation-max-avg-rank 2.3 `
  --validation-min-first-rate 0.25 `
  --validation-max-fourth-rate 0.25 `
  --validation-min-avg-score 25000 `
  --validation-min-win-rate 0.07 `
  --output-log outputs/stage5_south_defense.jsonl `
  --review-log outputs/stage5_decisions.jsonl
```

阶段目标不只是更会和牌，而是更少无谓放铳、被飞和四位；用 `outputs/stage5_decisions.jsonl` 复盘有人立直或明显开手后的弃牌危险度。
