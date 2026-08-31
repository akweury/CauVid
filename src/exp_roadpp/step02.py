


def main(input_data):
    print("\n------- Step 02 -------\n")
    use_gt = input_data["use_gt"]
    output_dir = input_data["output_dir"]
    step01_output_dir = input_data["step01_output_dir"]
    mask_iou_th = input_data["mask_iou_th"]
    top_k = input_data["top_k"]
    window_size = input_data["window_size"]
    frame_rate = input_data["frame_rate"]
    if use_gt:
        # Your logic for when ground truth is used
        return
    # Your step 02 processing logic here

    print("\n--------- Step 02 Done ---------------\n")