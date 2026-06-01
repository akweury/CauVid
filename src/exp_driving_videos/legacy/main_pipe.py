

from exp_driving_videos.pipe_utils.matrix2signal import matrix2signal
from exp_driving_videos.pipe_utils.signal2segs import signal2segs
from exp_driving_videos.pipe_utils.percept2matrix import raw2objs
import config 


def get_mini_video_ids():
    folder_path = config.get_dataset_path('driving_mini') / "videos"
    video_ids = [f.stem for f in folder_path.glob("*.mov")]
    return video_ids

def main():   
      
    # get all the video ids 
    video_ids = get_mini_video_ids()[:1]
    
    for vid in video_ids:
        print(f"Processing video: {vid}")
        
        # time series position matrix
        matrix = raw2objs(vid, save_matrices_flag=True)
        # convert the position matrix to primitive signals
        prims, ego_motion = matrix2signal(matrix, vid, visualize_ego=True, save_primitives=True)
        
        # extract ego motion signals
        ego_w_signal = ego_motion[:,2]
        ego_vx_signal = ego_motion[:,0]
        ego_vz_signal = ego_motion[:,1]
        # segment the signals
        ego_w_segs = signal2segs(ego_w_signal)
        ego_vx_segs = signal2segs(ego_vx_signal)
        ego_vz_segs = signal2segs(ego_vz_signal)
        
        
        
        
            
            
if __name__ == "__main__":
    main()

    