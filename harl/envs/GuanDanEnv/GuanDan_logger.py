from harl.common.base_logger import BaseLogger


class GuanDanLogger(BaseLogger):
    def get_task_name(self):
        return f"{self.env_args['scenario']}-{self.env_args['task']}"
