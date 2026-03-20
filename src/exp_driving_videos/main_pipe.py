

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
    video_ids = get_mini_video_ids()
    
    for video_id in video_ids:
        print(f"Processing video: {video_id}")
        matrix = percept2matrix(video_id, save_matrices_flag=True)
        primitives_continued, ego_motion = matrix2primitives(matrix, video_id, visualize_ego=True, save_primitives=True)
        primitives_segmented = primitive_segmentation(primitives_continued, ego_motion, video_id)
if __name__ == "__main__":
    main()

    