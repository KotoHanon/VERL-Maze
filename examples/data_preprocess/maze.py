import argparse
import datasets
import os

SYSTEM_PROMPT = """
你是一个需要通关动态迷宫游戏的玩家。迷宫由4×4的方格构成，坐标系中x轴横向延伸（0到3），y轴纵向延伸（0到3）。
你的智能体始终从(0,0)出发，目标是通过RLUD指令移动到唯一的终点。其中，R：x+1（右移）； L：x-1（左移）； U：y-1（上移）；D：y+1（下移）

注意：有两个动态障碍物分布在不同的列，
它们会在各自初始列进行纵向往复运动——初始向下移动（+y方向），当碰到y=3边界时转为向上移动，碰到y=0时恢复向下。
每次你执行一个动作后，所有障碍物会同步移动1格。例如初始在(2,1)的障碍物将按(2,1)→(2,2)→(2,3)→(2,2)→(2,1)→(2,0)→(2,1)的轨迹循环移动。

你必须确保：
1.智能体移动后不超出迷宫边界；
2.不与任何障碍物当前坐标重合；
3.最终到达终点。
动作序列必须严格用RLUD字符表示（如"RRDD"）。请先详细推演每一步智能体移动后的坐标变化，同步计算障碍物的实时位置，验证是否触发失败条件，
最终输出最短可行路径。将最终动作序列填写在<answer>...</answer>中，例如<answer>RR</answer>。
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/maze')

    args = parser.parse_args()

    train_dataset = datasets.load_dataset("csv",data_files="/root/LLM-Reasoning-Maze/Data/processed_train_data.csv", split="train")
    test_dataset = datasets.load_dataset("csv",data_files="/root/LLM-Reasoning-Maze/Data/processed_test_data.csv", split="train")

    def make_map_fn(split):

        def process_fn(example, idx):
            #map = example.pop("map") # 地图
            question_raw = example.pop("instruct") # 具体位置

            question = SYSTEM_PROMPT + ' ' + question_raw
            answer_raw = example.pop("action_seq") # 动作序列

            data = {
                "data_source": "maze",
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_raw
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                }
            }

            return data

        return process_fn


    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))






