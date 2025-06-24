from harl.common.base_logger import BaseLogger
import numpy as np

class GuanDanLogger(BaseLogger):
    def get_task_name(self):
        """Get task name for logging"""
        return "GuanDan"
    
    def eval_init(self):
        """Initialize evaluation logger"""
        super().eval_init()
        self.eval_episode_cnt = 0
        self.eval_win_cnt = 0
        self.eval_score_cnt = 0
        
    def eval_thread_done(self, tid):
        """Log evaluation results for one thread"""
        super().eval_thread_done(tid)
        self.eval_episode_cnt += 1
        
        # Check if the team won the game
        if self.eval_infos[tid][0]["game_state"] == "Finished":
            rewards = np.sum([r[0] for r in self.one_episode_rewards[tid]], axis=0)
            if rewards > 0:  # Assuming positive reward means win
                self.eval_win_cnt += 1
                
    def eval_log(self, eval_episode):
        """Log evaluation information"""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_win_rate = self.eval_win_cnt / self.eval_episode_cnt if self.eval_episode_cnt > 0 else 0
        
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
            "eval_win_rate": [eval_win_rate],
        }
        
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards) if len(self.eval_episode_rewards) > 0 else 0
        
        print(
            "Evaluation average episode reward is {}, evaluation win rate is {}.\n".format(
                eval_avg_rew, eval_win_rate
            )
        )
        
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew, eval_win_rate])) + "\n"
        )
        self.log_file.flush()