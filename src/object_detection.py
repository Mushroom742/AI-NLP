import cvlib as cv

'''
Permet de détecter des objets dans une image grâce à YOLOv3
'''
def detect_object(frame):
    bbox, label, conf = cv.detect_common_objects(frame)
    print(bbox, label, conf)

'''
Permet de détecter des objets dans une vidéo, en la séparant en frames
'''
def detect_object_video(video_file):
    frames = cv.get_frames(video_file)
    for f in frames:
        detect_object(f)

if __name__ == "__main__":
    detect_object_video('../data/test/video_test2.mp4')