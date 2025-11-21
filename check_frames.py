import cv2
import os
import glob

def get_frame_count(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return -1
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

def main():
    avi_dir = 'dataset/avi'
    mp4_dir = 'dataset/mp4'
    
    avi_files = glob.glob(os.path.join(avi_dir, '*.avi'))
    print(f"Found {len(avi_files)} AVI files.")
    
    for i, avi_path in enumerate(avi_files[:5]): # Check first 5
        basename = os.path.splitext(os.path.basename(avi_path))[0]
        mp4_path = os.path.join(mp4_dir, basename + '.mp4')
        
        if os.path.exists(mp4_path):
            avi_count = get_frame_count(avi_path)
            mp4_count = get_frame_count(mp4_path)
            print(f"{basename}: AVI={avi_count}, MP4={mp4_count}, Diff={avi_count - mp4_count}")
        else:
            print(f"{basename}: MP4 not found")

if __name__ == "__main__":
    main()
