

from src.exp_driving_videos.matrix2primitives import matrix2primitives
from src.exp_driving_videos.percept2matrix import percept2matrix
from src.exp_driving_videos.primitive_segmentation import primitive_segmentation
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
        matrix = percept2matrix(vid, save_matrices_flag=True)
        prims, ego_motion = matrix2primitives(matrix, vid, visualize_ego=True, save_primitives=True)
        
        for ap in range(90, 100, 1):
            prims_seg = primitive_segmentation(prims, ego_motion, vid, activity_percentile=ap, min_segment_len=2)
        
            
            
if __name__ == "__main__":
    main()

    