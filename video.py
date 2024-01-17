import cv2
from pydub import AudioSegment

def play_video(video_path):
    # Read video
    video = cv2.VideoCapture(video_path + ".mp4")

    while True:
        grabbed, frame = video.read()
        frame = cv2.resize(frame, (800, 500))

        if not grabbed:
            print("End of video")
            break

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        current_time=video.get(cv2.CAP_PROP_POS_MSEC)/1000
        print(current_time)

    video.release()
    cv2.destroyAllWindows()

video_path = "dog_cat"
play_video(video_path)
