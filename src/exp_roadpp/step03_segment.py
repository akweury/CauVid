


class Segmentor:
    def __init__(self, ):
        pass

    def run(self, mode,vid, agent_tubes, action_tubes, av_action_tubes):
        if mode == 'ego_action':
            return self.segment_by_ego_actions(vid, agent_tubes, action_tubes, av_action_tubes)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        
    
        


